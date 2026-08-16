"""Preflight for the A1 ingest → gates → formal A2 sequence.

Every check either passes or stops the run. None of them repairs anything: a
preflight that fixes what it finds is a preflight that can talk itself into
starting, and the whole reason this exists is that the expensive mistakes here have
all been silent ones — a duplicate ingest that under-reported its own spend, a
`--limit` that selected different questions than the manifest, a resume that would
have merged answers from two different stores.

Run it alone to see the state without starting anything:

    python scripts/a2_preflight.py

Add --run to continue into ingest, the two gates, and the formal A2 when, and only
when, every check passes.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from llm_long_term_memory.config import ExperimentConfig, Settings  # noqa: E402
from llm_long_term_memory.ingest import fingerprint  # noqa: E402
from llm_long_term_memory.ingest.pipeline import resolved_sessions_per_request  # noqa: E402
from llm_long_term_memory.llm import Limits, QuotaManager  # noqa: E402

PY = str(REPO / ".venv" / "bin" / "python")
DEFAULT_STORE_NAME = "two-stage-p10"
DEFAULT_LABEL = "a2-clean-p10"
DEFAULT_VARIANT = "two_stage_hydrated"
MANIFEST = REPO / "results" / "manifests" / "dev50.json"
PILOT_OUT = REPO / "results" / "raw" / "two_stage_hydrated.jsonl"
EXPECTED_NAMESPACES = 50

# The formal A2 is two arms, because the shipped product includes the conditional
# raw-conversation fallback and `baselines.yaml` does not enable it. The diagnostic
# run made that visible: of its twenty failures, seventeen had retrieved the right
# session and three had not — and all three of those were the single-session-
# assistant cases the archive fallback exists to recover, including the Mayo
# question the demo answers correctly. A headline measured with the fallback off
# would not describe the thing being shipped.
#
# The two configs differ in `fallback.*` and nothing else (`name` and `description`
# are metadata), so the pair is a clean ablation: the delta is the fallback.
ARMS = [
    ("configs/baselines.yaml", "", "memory only, comparable with every earlier run"),
    ("configs/fallback.yaml", "-fallback", "memory first, archive when the answerer asks"),
]
VALID_SOURCE_ROLES = {"user", "assistant", "system"}
QUOTA_TZ = ZoneInfo("America/Los_Angeles")

# A floor, not a target. Extraction v4 measured 36.6% on holdout, and this project
# has no measured run-to-run noise figure for fidelity — only for end-to-end
# accuracy — so a threshold near the baseline would be a number I made up. 30% is
# far enough below to fire only on a real breakage, and the actual value is printed
# either way for a human to judge.
FIDELITY_FLOOR = 0.30
FIDELITY_BASELINE = 0.366


class Stop(Exception):
    """A check failed. Nothing is started, and nothing is repaired."""


@dataclass(frozen=True)
class Target:
    """What this run operates on. Parameterised so that the next extractor
    generation is a command line, not an edit to this file."""

    store_name: str
    variant: str
    label: str

    @property
    def store(self) -> Path:
        return REPO / "stores" / f"{self.store_name}.db"

    def out(self, suffix: str = "") -> Path:
        return REPO / "results" / "raw" / f"{self.variant}.{self.label}{suffix}.jsonl"


@dataclass
class Check:
    name: str
    detail: str


def ok(name: str, detail: str) -> Check:
    print(f"  \033[32mPASS\033[0m  {name}: {detail}")
    return Check(name, detail)


def stop(name: str, detail: str) -> None:
    print(f"  \033[31mSTOP\033[0m  {name}: {detail}")
    raise Stop(f"{name}: {detail}")


# ---------------------------------------------------------------- 1. quota


def _probe_extractor() -> str | None:
    """One real request against the extractor. Returns None on success, else why not."""
    from llm_long_term_memory.config import ExperimentConfig, Settings
    from llm_long_term_memory.llm import Limits, QuotaManager
    from llm_long_term_memory.llm.client import DailyQuotaExhausted, GeminiClient

    settings = Settings()
    cfg = ExperimentConfig.from_yaml(str(REPO / "configs" / "baselines.yaml"))
    quota = QuotaManager(
        state_dir=settings.store_dir / "quota",
        default=Limits(rpm=cfg.quota.rpm, tpm=cfg.quota.tpm, rpd=cfg.quota.rpd),
    )
    quota.load_learned()
    client = GeminiClient(settings.require_api_key(), quota=quota)
    try:
        client.generate(
            model=cfg.models.extractor,
            system="Reply with the single word: ready",
            prompt="ready?",
            role="preflight",
        )
    except DailyQuotaExhausted as exc:
        return f"still exhausted ({exc})"
    except Exception as exc:
        return f"probe failed: {type(exc).__name__}: {exc}"
    return None


def check_quota_really_reset(wait_minutes: int = 0) -> None:
    """Ask the provider, do not infer from the clock.

    The clock says what time it is, not whether the provider agrees. Today it was
    wrong in both directions: a reset asserted from local time that had not
    happened, and a "22.4h" figure that only made sense against the Pacific
    boundary. So this spends one real request and reads the answer.

    `wait_minutes` covers exactly one thing: the provider's reset landing a few
    minutes after midnight Pacific. It re-probes, it does not re-check anything
    else, and it never relaxes a threshold — a run that starts thirty minutes late
    is fine, a run that starts on a store or a quota that is not what it thinks is
    not. Any non-quota failure gives up immediately even inside the window.
    """
    now_pt = datetime.now(QUOTA_TZ)
    midnight = now_pt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    mins = int((midnight - now_pt).total_seconds() // 60)
    deadline = datetime.now().timestamp() + wait_minutes * 60

    while True:
        why = _probe_extractor()
        if why is None:
            ok("extractor quota", f"live probe accepted (PT {now_pt:%H:%M}, {mins} min to reset)")
            return
        if not why.startswith("still exhausted") or datetime.now().timestamp() >= deadline:
            stop(
                "extractor quota",
                f"{why}. Local clock says {mins} min to Pacific midnight, "
                f"but the provider is the authority and it says no.",
            )
        left = int((deadline - datetime.now().timestamp()) // 60)
        print(f"  \033[33mWAIT\033[0m  extractor quota: {why}; re-probing in 5 min ({left} left)")
        time.sleep(300)


# ------------------------------------------------------- 2. no other ingest


def check_no_concurrent_ingest(target: Target) -> None:
    """The failure that has actually cost quota here.

    Two ingests ran at once, each tracked its own usage, and the totals
    under-reported the real spend by about a third. Checked two ways because the
    lock is new and the process scan catches a run started before it existed.
    """
    from llm_long_term_memory.locking import lock_holder

    store = target.store
    holder = lock_holder(store.with_suffix(store.suffix + ".lock"))
    if holder is not None:
        stop("no concurrent ingest", f"pid {holder} holds the ingest lock on {store.name}")

    out = subprocess.run(
        ["/bin/ps", "-Ao", "pid=,command="], capture_output=True, text=True, check=False
    ).stdout
    mine = str(REPO)
    others = [
        line.strip()
        for line in out.splitlines()
        if "ingest run" in line and mine in line and str(Path(sys.argv[0]).name) not in line
    ]
    if others:
        stop("no concurrent ingest", f"{len(others)} ingest process(es) running: {others[0][:90]}")
    ok("no concurrent ingest", "lock free and no ingest process found")


# ------------------------------------------------------------- 3. the store


def corpus_unique_sessions() -> int:
    """How many distinct sessions a complete ingest should leave in the store.

    Not a constant. The first version of this check compared the store against 2,400
    and stopped a completed ingest at 2,348, because 2,400 is the number of session
    *entries* in the corpus and 2,348 is the number of distinct session *IDs* — 51
    IDs recur, since LongMemEval-S uses one session as evidence for more than one
    question. The pipeline counts entries; the store deduplicates by ID. Both
    numbers were right and they were never comparable.

    Deriving it from the corpus means the invariant cannot drift from the data, and
    it holds for any future dataset_limit without anyone remembering to edit a
    constant.
    """
    from llm_long_term_memory.config import ExperimentConfig, Settings
    from llm_long_term_memory.evaluation.datasets import longmemeval as lme

    cfg = ExperimentConfig.from_yaml(str(REPO / "configs" / "baselines.yaml"))
    instances = lme.load("s", Settings().data_dir, limit=getattr(cfg, "dataset_limit", None))
    return len({sess.session_id for inst in instances for sess in inst.sessions})


def store_counts(target: Target) -> tuple[int, int, int]:
    conn = sqlite3.connect(f"file:{target.store}?mode=ro", uri=True)
    try:
        sessions = conn.execute("SELECT COUNT(DISTINCT session_id) FROM turns").fetchone()[0]
        namespaces = conn.execute("SELECT COUNT(DISTINCT user_id) FROM memories").fetchone()[0]
        memories = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    finally:
        conn.close()
    return sessions, namespaces, memories


def store_meta(target: Target, key: str) -> str | None:
    conn = sqlite3.connect(f"file:{target.store}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()
    return row[0] if row else None


def check_target_store(target: Target, expected: int) -> bool:
    """Resume the existing store. Never create, never overwrite.

    A missing file here would mean the store moved or the name is wrong, and the
    recovery for both is a human looking — not this script quietly creating an empty
    database and ingesting the whole corpus into it.

    Returns whether the ingest still has work to do. A complete store is not an
    error: once the corpus is in, the remaining sequence is gates and A2, and this
    script is the thing that runs them. It used to stop here, on the reasoning that
    "nothing to resume" meant a mistake — which was true while ingestion was the
    point and false the moment it finished, leaving the completed store with no way
    to reach the checks that gate its own evaluation.
    """
    store = target.store
    if not store.exists():
        stop("target store", f"{store} does not exist — refusing to create one")
    sessions, namespaces, memories = store_counts(target)
    if sessions == 0:
        stop("target store", f"{store.name} is empty — this is not a resume")
    if sessions > expected:
        stop("target store", f"{sessions} sessions against a corpus of {expected} — not this data")
    complete = sessions == expected
    ok(
        "target store",
        f"{store.name}: {sessions}/{expected} sessions, "
        f"{namespaces} namespaces, {memories:,} memories — "
        + ("complete, ingest will be skipped" if complete else "resuming"),
    )
    ok("store extractor", store_meta(target, "extractor_version") or "unstamped")
    return not complete


# --------------------------------------------- homogeneity, before the formal A2


def expected_fingerprint() -> fingerprint.IngestSpec:
    """What a store written by the current configuration should carry.

    Built from resolved configuration alone — no Gemini client, no encoder, no
    request. A preflight that had to construct the runtime in order to describe it
    would fail whenever the model dependencies were missing, for reasons having
    nothing to do with the store it is checking, and it could reach the network on
    a path whose entire job is to look without touching.

    `sessions_per_request` is resolved through the same helper the ingestion driver
    uses, because the configured batch size is not necessarily the effective one —
    the model's token budget can lower it, and it is the effective value that goes
    into the fingerprint. Reading `cfg.ingest.sessions_per_request` directly here
    would compare a store against a number that never ran.

    The remaining risk — configuration saying one thing while the objects built
    from it do another — is not this function's to carry. `IngestionPipeline.run`
    compares its live objects against this same config-derived spec before spending
    a request, so the two layers together cover what one of them cannot.
    """
    cfg = ExperimentConfig.from_yaml(str(REPO / "configs" / "baselines.yaml"))
    settings = Settings()
    quota = QuotaManager(
        state_dir=settings.store_dir / "quota",
        default=Limits(rpm=cfg.quota.rpm, tpm=cfg.quota.tpm, rpd=cfg.quota.rpd),
    )
    quota.load_learned()
    per_request = resolved_sessions_per_request(
        cfg.ingest.sessions_per_request,
        quota.for_model(cfg.models.extractor).limits.tpm,
    )
    return fingerprint.from_config(cfg, sessions_per_request=per_request)


def check_store_is_homogeneous(target: Target, expected_sessions: int) -> None:
    """Prove the store is the product of one system, by inspecting the rows.

    `meta` is a claim, not evidence: it records the last writer, which is exactly
    why the mixed store described itself as `two-stage-p10-v2` while 68% of its rows
    came from the generation before. The fingerprint and the rows are both checked,
    and the division of labour between them is not symmetric:

    * the fingerprint establishes that the store was written by the configuration
      running now, and for a store first written before the guard existed it is
      stamped on trust — the resume that stamps it has no way to inspect rows
      already on disk;
    * the row checks are the evidence. `scope IS NULL` is a pre-P10 signature, and
      it separates the two real stores cleanly: 4,843 such rows in the mixed one,
      0 in the clean one. A store that lied in `meta` still fails here.

    Deliberately not by `ingested_at` date. A rebuild that spans two days of quota
    is the normal case under this quota, so "all rows share a date" would fail a
    healthy store and pass an unhealthy one that happened to fit inside a day.
    """
    expected_fp = expected_fingerprint().as_dict()
    stored_fp = fingerprint.loads(store_meta(target, "ingest_fingerprint"))
    if not stored_fp:
        stop("homogeneity", "store carries no ingest_fingerprint — it predates the guard")
    moved = fingerprint.differences(stored_fp, expected_fp)
    if moved:
        stop("homogeneity", "store fingerprint is not the current one: " + "; ".join(moved))
    ok("homogeneity: fingerprint", expected_fp["extractor_version"] + " matches the current code")

    conn = sqlite3.connect(f"file:{target.store}?mode=ro", uri=True)
    try:
        scope_null = conn.execute("SELECT COUNT(*) FROM memories WHERE scope IS NULL").fetchone()[0]
        roles = dict(conn.execute("SELECT source_role, COUNT(*) FROM memories GROUP BY 1"))
        sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        namespaces = conn.execute("SELECT COUNT(DISTINCT user_id) FROM memories").fetchone()[0]
    finally:
        conn.close()

    if scope_null:
        stop("homogeneity", f"{scope_null:,} memories have scope NULL — a pre-P10 signature")
    ok("homogeneity: scope", "0 rows with scope NULL")

    invalid = {r: n for r, n in roles.items() if r not in VALID_SOURCE_ROLES}
    if invalid:
        stop("homogeneity", f"invalid source_role values: {invalid}")
    ok("homogeneity: source_role", ", ".join(f"{r} {n:,}" for r, n in sorted(roles.items())))

    if sessions != expected_sessions:
        stop("homogeneity", f"{sessions} sessions, expected {expected_sessions}")
    if namespaces != EXPECTED_NAMESPACES:
        stop("homogeneity", f"{namespaces} namespaces, expected {EXPECTED_NAMESPACES}")
    ok("homogeneity: coverage", f"{sessions} sessions across {namespaces} namespaces")

    # Recorded, never gated. It is a number to watch, not a threshold to pass: the
    # rate was 18.8% of substantive sessions in the mixed store and 15.1% in the
    # clean one, and this project has no evidence for where a healthy line sits.
    # Gating on an invented threshold would stop a run for a number nobody can
    # defend; printing it makes an invisible failure visible, which is the whole
    # gap it was found in.
    from llm_long_term_memory.store import SQLiteMemoryStore

    s = SQLiteMemoryStore(target.store)
    s.initialize()
    try:
        zero, subst = s.zero_yield_sessions()
    finally:
        s.close()
    print(
        f"  \033[36mNOTE\033[0m  zero-memory sessions: {zero:,} of {subst:,} "
        f"({zero / subst:.1%}) — recorded, not gated"
    )


# ------------------------------------------------------- output-file safety


def check_a2_output_is_clean(target: Target) -> None:
    """Each arm writes its own file; the pilot artifact is never touched."""
    for _, suffix, _ in ARMS:
        out = target.out(suffix)
        if out.exists():
            n = len([ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()])
            stop("a2 output", f"{out.name} already holds {n} rows — move it aside first")
    if not MANIFEST.exists():
        stop("a2 manifest", f"{MANIFEST} is missing")
    n_q = len(json.loads(MANIFEST.read_text(encoding="utf-8"))["question_ids"])
    names = ", ".join(target.out(s).name for _, s, _ in ARMS)
    ok("a2 output", f"free: {names}; {PILOT_OUT.name} stays as the pilot record")
    ok("a2 manifest", f"{MANIFEST.name}: {n_q} frozen question ids")


# --------------------------------------------------------------- the stages


def check_arms_differ_only_in_fallback() -> None:
    """The pair is only an ablation if one thing differs.

    Run before the arms rather than asserted in a comment, because `Product - Base`
    is attributable to the fallback exactly as far as this holds. `name` and
    `description` are metadata and are ignored.
    """
    import yaml

    def flat(d, prefix=""):
        out = {}
        for k, v in (d or {}).items():
            key = f"{prefix}.{k}" if prefix else k
            out.update(flat(v, key) if isinstance(v, dict) else {key: v})
        return out

    a = flat(yaml.safe_load((REPO / ARMS[0][0]).read_text(encoding="utf-8")))
    b = flat(yaml.safe_load((REPO / ARMS[1][0]).read_text(encoding="utf-8")))
    ignore = {"name", "description"}
    moved = sorted(k for k in set(a) | set(b) if k not in ignore and a.get(k) != b.get(k))
    unexpected = [k for k in moved if not k.startswith("fallback.")]
    if unexpected:
        stop("comparability", f"the arms differ in more than the fallback: {unexpected}")
    ok("comparability", f"arms differ only in {', '.join(moved)}")


def audit_result(path: Path, manifest_ids: set[str]) -> None:
    """Check a finished arm before anyone reads its number.

    Every one of these has been wrong at least once in this project: a run that
    reported 50 questions over a file holding 30, a `--limit 31` that selected
    different questions than intended, and a resume that would have merged two
    stores into one accuracy.
    """
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    ids = [r["question_id"] for r in rows]
    name = path.name
    if len(ids) != len(set(ids)):
        stop(f"audit:{name}", f"{len(ids)} rows but {len(set(ids))} unique question ids")
    if set(ids) != manifest_ids:
        missing, extra = manifest_ids - set(ids), set(ids) - manifest_ids
        stop(f"audit:{name}", f"does not match dev50: {len(missing)} missing, {len(extra)} extra")
    for field in ("store_fingerprint", "answer_prompt_version", "judge_prompt_version"):
        values = {r.get(field) for r in rows}
        if len(values) != 1:
            stop(f"audit:{name}", f"{field} is not single-valued: {sorted(map(str, values))}")
    correct = sum(r["correct"] for r in rows)
    ok(
        f"audit:{name}",
        f"{len(rows)} rows, dev50 exact, one store and one prompt pair — "
        f"{correct}/{len(rows)} = {correct / len(rows):.1%}",
    )


def run_stage(name: str, args: list[str]) -> None:
    print(f"\n\033[1m=== {name} ===\033[0m", flush=True)
    result = subprocess.run([PY, "-m", "llm_long_term_memory.cli", *args], cwd=REPO)
    if result.returncode != 0:
        stop(name, f"exited {result.returncode}")


def verify_gate_artifact(name: str, path: Path, key: str) -> None:
    """Read the verdict from the artifact, not from the exit code alone.

    A stale artifact is the trap here: if the gate crashed before writing, the file
    on disk is yesterday's pass. Freshness is checked against this process's start.
    """
    if not path.exists():
        stop(f"gate:{name}", f"{path.name} was not written")
    age = datetime.now().timestamp() - path.stat().st_mtime
    if age > 3600:
        stop(f"gate:{name}", f"{path.name} is {age / 60:.0f} min old — not from this run")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get(key):
        stop(f"gate:{name}", f"{key} is {data.get(key)!r} in {path.name}")
    ok(f"gate:{name}", f"{key} true in a fresh {path.name}")


def verify_fidelity_artifact() -> None:
    path = REPO / "results" / "raw" / "fidelity.json"
    if not path.exists():
        stop("gate:fidelity", "fidelity.json was not written")
    age = datetime.now().timestamp() - path.stat().st_mtime
    if age > 3600:
        stop("gate:fidelity", f"fidelity.json is {age / 60:.0f} min old — not from this run")
    overall = json.loads(path.read_text(encoding="utf-8"))["overall"]
    if overall < FIDELITY_FLOOR:
        stop("gate:fidelity", f"overall {overall:.1%} below floor {FIDELITY_FLOOR:.1%}")
    delta = overall - FIDELITY_BASELINE
    note = "" if abs(delta) < 0.03 else f"  \033[33m({delta:+.1%} vs the v4 baseline)\033[0m"
    ok("gate:fidelity", f"overall {overall:.1%} (v4 baseline {FIDELITY_BASELINE:.1%}){note}")


def check_ingest_completed(target: Target, expected: int) -> None:
    sessions, namespaces, memories = store_counts(target)
    print(
        f"\n  after ingest: {sessions}/{expected} sessions, "
        f"{namespaces} namespaces, {memories:,} memories"
    )
    if sessions < expected:
        # No cause is offered. The previous version of this message said "quota
        # probably ran out" without checking, and was wrong — the ingest had
        # completed and the invariant was the thing at fault. An error that guesses
        # points the next reader away from the real problem.
        stop(
            "ingest complete",
            f"{sessions}/{expected} sessions. The gates and A2 have NOT been "
            f"started. Check the ingest output above for why it stopped.",
        )
    ok("ingest complete", f"{sessions} sessions, {namespaces} namespaces, {memories:,} memories")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="continue into ingest, gates and A2 when every check passes",
    )
    parser.add_argument(
        "--store-name",
        default=DEFAULT_STORE_NAME,
        help=f"store stem to resume into (default: {DEFAULT_STORE_NAME})",
    )
    parser.add_argument(
        "--label",
        default=DEFAULT_LABEL,
        help=f"result label, giving <variant>.<label>.jsonl (default: {DEFAULT_LABEL})",
    )
    parser.add_argument(
        "--variant",
        default=DEFAULT_VARIANT,
        help=f"evaluation variant (default: {DEFAULT_VARIANT})",
    )
    parser.add_argument(
        "--wait-for-quota",
        type=int,
        default=0,
        metavar="MINUTES",
        help="re-probe the extractor quota for up to this long before giving up. "
        "Covers the provider's reset landing a few minutes late; it relaxes no "
        "other check.",
    )
    args = parser.parse_args()
    target = Target(store_name=args.store_name, variant=args.variant, label=args.label)

    print(f"\033[1mpreflight\033[0m  {target.store_name} -> {len(ARMS)} A2 arms")
    try:
        expected = corpus_unique_sessions()
        ok("corpus", f"{expected:,} unique session ids — the completion target")
        check_quota_really_reset(args.wait_for_quota)
        check_no_concurrent_ingest(target)
        needs_ingest = check_target_store(target, expected)
        check_a2_output_is_clean(target)

        # Homogeneity is a read-only check, so it runs here rather than only inside
        # --run: a complete store can be inspected without committing to spending
        # anything on it. An incomplete one cannot be — the check compares against
        # the corpus total — so it is deferred to after the ingest, where it has
        # always run.
        if not needs_ingest:
            print("\n\033[1m=== homogeneity ===\033[0m")
            check_store_is_homogeneous(target, expected)
    except Stop:
        print("\n\033[31mPreflight failed. Nothing started.\033[0m")
        return 1

    if not args.run:
        todo = "ingest, gates and A2" if needs_ingest else "gates and A2"
        print(f"\n\033[33mChecks pass. Re-run with --run to start {todo}.\033[0m")
        return 0

    try:
        if needs_ingest:
            run_stage("ingest (resume)", ["ingest", "run", "--store-name", target.store_name])
            check_ingest_completed(target, expected)

            # Homogeneity before the gates, because a store that is not one system
            # makes everything downstream unattributable — the incident this whole
            # sequence exists to prevent recurring.
            print("\n\033[1m=== homogeneity ===\033[0m")
            check_store_is_homogeneous(target, expected)

        # Both gates before A2, and a failure in either stops here. They cost a few
        # requests each against ~250 for the run they gate, which is the whole
        # argument for running them first.
        #
        # Their exit codes are only trustworthy as of this commit: temporal-gate
        # printed "Gate closed" and exited 0, and fidelity had no threshold at all,
        # so an unattended sequence would have walked straight through both. The
        # artifacts are re-read below rather than trusting the exit code alone.
        run_stage("gate: temporal", ["ingest", "temporal-gate"])
        verify_gate_artifact(
            "temporal", REPO / "results" / "raw" / "temporal-gate.json", "gate_open"
        )
        run_stage(
            "gate: fidelity",
            ["ingest", "fidelity", "--holdout", "--min-score", str(FIDELITY_FLOOR)],
        )
        verify_fidelity_artifact()

        # `variant` is positional and there is no --out: --label decides the
        # filename, leaving the pilot file alone. A labelled file is excluded from
        # the default results table, so promoting either arm into the published
        # table is a separate, deliberate step.
        #
        # Two arms, same store, same manifest, same answerer and judge. Checked
        # rather than asserted, because Product - Base is attributable to the
        # fallback exactly as far as that holds.
        print("\n\033[1m=== comparability ===\033[0m")
        check_arms_differ_only_in_fallback()

        manifest_ids = set(json.loads(MANIFEST.read_text(encoding="utf-8"))["question_ids"])
        for config, suffix, why in ARMS:
            run_stage(
                f"formal A2 [{config.split('/')[-1]}] — {why}",
                [
                    "eval",
                    "run",
                    target.variant,
                    "--config",
                    config,
                    "--store-name",
                    target.store_name,
                    "--questions",
                    str(MANIFEST),
                    "--label",
                    target.label + suffix,
                ],
            )
            # Audited before the number is read, not after it is quoted.
            audit_result(target.out(suffix), manifest_ids)
    except Stop:
        print("\n\033[31mStopped. Later stages were not started.\033[0m")
        return 1

    print("\n\033[32mDone.\033[0m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
