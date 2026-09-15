"""Re-score the extraction batch-size experiment from its archived extractions.

The experiment is one parameter — how many sessions share an extraction request —
against one ruler, `ingest.fidelity`, on one held-out cohort. The result is the
largest single effect measured in this project: **37.3% at the shipped batch of 15
against 78.4% at a batch of 1**, on the same sessions, same prompts, same model.
The long-cited "extraction is lossy, 36.6%" is therefore not a property of the
extractor. It is a property of the batch size the free tier's 500 requests/day
forced.

`chronomem ingest fidelity --batch N` measures one arm and overwrites
`results/raw/fidelity.json`, so running the curve leaves only its last row behind.
The arms survive in the CLI's extraction cache under `stores/`, which is ignored
and machine-local. This promotes them to a committed archive and re-scores from
it, so the curve is reproducible without the cache, without the API and without
quota.

    python3 tools/batch_size_curve.py archive   # stores/fidelity-cache/ -> the archive
    python3 tools/batch_size_curve.py score     # the archive -> the curve, zero calls
    python3 tools/batch_size_curve.py restore   # the archive -> stores/fidelity-cache/

An arm is addressed by the fingerprint the CLI derives from the live prompts, the
model and the batch size — recomputed here rather than hardcoded. If a prompt is
edited the fingerprint moves, `archive` stops finding the cache and `score` reports
the arms as no longer comparable to the current extractor. That is the intended
behaviour: these numbers describe the prompts they were produced under, and a
curve silently re-attributed to a different prompt would be worse than no curve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

ARCHIVE = REPO / "results/raw/extraction-batch-size.extractions.json"
OUT = REPO / "results/analysis/extraction-batch-size.json"
CACHE_DIR = REPO / "stores/fidelity-cache"

CONFIG = "configs/v2.yaml"
SESSIONS = 60
HOLDOUT = True

# (arm, sessions per request, grounded). The grounded arm extracts one session per
# request by construction, so it is compared against `batch1` and not against the
# shipped 15 — otherwise two variables would be wearing one number.
ARMS: list[tuple[str, int, bool]] = [
    ("batch15", 15, False),
    ("batch8", 8, False),
    ("batch5", 5, False),
    ("batch3", 3, False),
    ("batch1", 1, False),
    ("grounded", 1, True),
]


def _fingerprint(batch: int, grounded: bool, model: str) -> str:
    """The CLI's cache key, recomputed from the prompts as they are right now."""
    from llm_long_term_memory.ingest.extract_facts import _PROMPT as FACTS_PROMPT
    from llm_long_term_memory.ingest.extract_facts import FACTS_SYSTEM
    from llm_long_term_memory.ingest.keying import _PROMPT as KEYING_PROMPT
    from llm_long_term_memory.ingest.keying import KEYING_SYSTEM

    if grounded:
        from llm_long_term_memory.ingest.grounded import GroundedExtractor

        prompt = "".join(GroundedExtractor(None, model).prompt_texts())
    else:
        prompt = FACTS_SYSTEM + FACTS_PROMPT + KEYING_SYSTEM + KEYING_PROMPT
    # `two_stage` is True for every arm here; the single-stage extractor is v1.
    return hashlib.sha1(
        (prompt + model + str(batch) + str(True) + str(grounded)).encode()
    ).hexdigest()[:12]


def _cohort() -> list:
    """The scored sessions, rebuilt the way `chronomem ingest fidelity` picks them.

    Deterministic: `split_dev_test` is seeded, and the first `SESSIONS` sessions of
    the held-out pool are taken in instance order.
    """
    from llm_long_term_memory.config import ExperimentConfig
    from llm_long_term_memory.evaluation.datasets import longmemeval as lme

    cfg = ExperimentConfig.from_yaml(REPO / CONFIG)
    every = lme.load(cfg.dataset_variant, REPO / "data")
    dev, test = lme.split_dev_test(every)
    picked = []
    for inst in test if HOLDOUT else dev:
        for sess in inst.sessions:
            picked.append(sess)
            if len(picked) >= SESSIONS:
                return picked
    return picked


def _cohort_digest(sessions: list) -> str:
    """Hashes the text the ruler reads, not the ids.

    Two corpora can agree on session ids and disagree on their turns; it is the
    turns that set the denominator, so it is the turns that are hashed.
    """
    from llm_long_term_memory.ingest.fidelity import user_assertions

    h = hashlib.sha256()
    for sess in sessions:
        h.update(sess.session_id.encode())
        h.update(b"\x00")
        h.update(user_assertions(sess).encode())
        h.update(b"\x1e")
    return h.hexdigest()


def _model() -> str:
    from llm_long_term_memory.config import ExperimentConfig

    return ExperimentConfig.from_yaml(REPO / CONFIG).models.extractor


def archive() -> int:
    model = _model()
    sessions = _cohort()
    ids = [s.session_id for s in sessions]
    out: dict = {
        "note": (
            "Archived extractions for the batch-size curve. One cohort of held-out "
            "sessions, one ruler, one parameter moving. Re-score with "
            "`python3 tools/batch_size_curve.py score` — no API calls."
        ),
        "config": CONFIG,
        "model": model,
        "sessions": SESSIONS,
        "holdout": HOLDOUT,
        "session_ids": ids,
        "cohort_sha256": _cohort_digest(sessions),
        "arms": {},
    }
    missing = []
    for arm, batch, grounded in ARMS:
        fp = _fingerprint(batch, grounded, model)
        path = CACHE_DIR / f"{fp}.json"
        if not path.exists():
            missing.append(f"{arm} ({fp})")
            continue
        cached = json.loads(path.read_text(encoding="utf-8"))
        out["arms"][arm] = {
            "sessions_per_request": batch,
            "grounded": grounded,
            "fingerprint": fp,
            "extractions": {sid: cached[sid] for sid in ids if sid in cached},
        }
    if missing:
        print("no cached extraction for: " + ", ".join(missing), file=sys.stderr)
        print(
            "(a prompt edit moves the fingerprint; the arm would have to be re-run)",
            file=sys.stderr,
        )
        return 1
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"→ {ARCHIVE.relative_to(REPO)} ({len(out['arms'])} arms, {SESSIONS} sessions)")
    return 0


def _requests(pairs: list, batch: int, grounded: bool) -> int:
    """What the arm actually cost, counted rather than assumed.

    The two-stage extractor spends one request on facts and a second on keying —
    except on a chunk where stage A returned nothing, which never reaches stage B.
    Multiplying chunks by two would overstate the batch-15 arm, whose chunks are the
    ones that come back empty. The grounded extractor is one request per session.
    """
    if grounded:
        return len(pairs)
    total = 0
    for i in range(0, len(pairs), batch):
        total += 1 if all(not mems for _, mems in pairs[i : i + batch]) else 2
    return total


def _outcomes(pairs: list) -> list[tuple[str, str, str, bool]]:
    """One row per stated specific: did it survive into any memory from its session?

    `score_sessions` aggregates these into recalls. Keeping them un-aggregated is
    what makes the arms *paired* — the same 134 specifics, asked of each arm — so
    the comparison is a within-item test rather than two proportions that happen to
    have been measured on the same corpus.
    """
    from llm_long_term_memory.ingest.fidelity import _extract_facets, user_assertions

    rows = []
    for session, memories in pairs:
        blob = " || ".join(f"{m.content} {m.object or ''}" for m in memories).lower()
        for facet, values in _extract_facets(user_assertions(session)).items():
            for value in values:
                rows.append((session.session_id, facet, value, value in blob))
    return rows


def _mcnemar(a: list, b: list) -> dict:
    """Exact two-sided McNemar on the discordant specifics.

    Concordant items carry no information about a difference, so the test is a sign
    test on the rest. Exact rather than chi-square: the discordant counts here run
    to single figures between adjacent arms, where the asymptotic form is not to be
    trusted.
    """
    from math import comb

    assert [r[:3] for r in a] == [r[:3] for r in b], "arms are not paired"
    gained = sum(1 for x, y in zip(a, b, strict=True) if not x[3] and y[3])
    lost = sum(1 for x, y in zip(a, b, strict=True) if x[3] and not y[3])
    n = gained + lost
    p = 1.0
    if n:
        tail = sum(comb(n, k) for k in range(min(gained, lost) + 1))
        p = min(1.0, 2 * tail / 2**n)
    return {"gained": gained, "lost": lost, "p": p}


def _load_archive() -> dict:
    if not ARCHIVE.exists():
        raise SystemExit(f"{ARCHIVE.relative_to(REPO)} is missing; run `archive` first")
    return json.loads(ARCHIVE.read_text(encoding="utf-8"))


def score(write: bool = True) -> dict:
    from llm_long_term_memory.ingest.fidelity import score_sessions
    from llm_long_term_memory.store import Memory

    data = _load_archive()
    sessions = _cohort()
    by_id = {s.session_id: s for s in sessions}
    digest = _cohort_digest(sessions)
    model = _model()

    rows: list[dict] = []
    outcomes: dict[str, list] = {}
    for arm, batch, grounded in ARMS:
        held = data["arms"].get(arm)
        if held is None:
            continue
        pairs = []
        for sid in data["session_ids"]:
            mems = [
                Memory(**{**d, "event_time": None, "valid_from": None, "valid_to": None})
                for d in held["extractions"].get(sid, [])
            ]
            pairs.append((by_id[sid], mems))
        report = score_sessions(pairs)
        outcomes[arm] = _outcomes(pairs)
        rows.append(
            {
                "arm": arm,
                "sessions_per_request": batch,
                "grounded": grounded,
                "fingerprint_recorded": held["fingerprint"],
                "fingerprint_now": _fingerprint(batch, grounded, model),
                "overall": report.overall,
                "memories": report.memories,
                "memories_per_session": report.memories_per_session,
                "chunks": -(-SESSIONS // batch),
                "requests": _requests(pairs, batch, grounded),
                "per_facet": {f: s.recall for f, s in report.per_facet.items()},
                "stated": {f: s.stated for f, s in report.per_facet.items()},
                "retained": {f: s.retained for f, s in report.per_facet.items()},
            }
        )

    # Adjacent steps say where the curve stops paying; the two spanning comparisons
    # say what the whole move is worth and what requiring citations cost.
    comparisons = [
        ("batch15", "batch8"),
        ("batch8", "batch5"),
        ("batch5", "batch3"),
        ("batch3", "batch1"),
        ("batch15", "batch1"),
        ("batch1", "grounded"),
    ]
    pairwise = [
        {"from": a, "to": b, **_mcnemar(outcomes[a], outcomes[b])}
        for a, b in comparisons
        if a in outcomes and b in outcomes
    ]

    stale = [r["arm"] for r in rows if r["fingerprint_recorded"] != r["fingerprint_now"]]
    result = {
        "note": (
            "Extraction fidelity against sessions per extraction request. Re-scored "
            "offline from results/raw/extraction-batch-size.extractions.json by "
            "tools/batch_size_curve.py; no API calls, no quota."
        ),
        "config": data["config"],
        "model": data["model"],
        "sessions": data["sessions"],
        "holdout": data["holdout"],
        "cohort_sha256_recorded": data["cohort_sha256"],
        "cohort_sha256_now": digest,
        "cohort_matches": digest == data["cohort_sha256"],
        "prompts_match": not stale,
        "stale_arms": stale,
        "specifics": len(next(iter(outcomes.values()))) if outcomes else 0,
        "arms": rows,
        "pairwise": pairwise,
    }
    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    width = max(len(r["arm"]) for r in rows)
    print(f"{'arm'.ljust(width)}  batch  overall  mem/sess  requests")
    for r in rows:
        print(
            f"{r['arm'].ljust(width)}  {r['sessions_per_request']:>5}  "
            f"{r['overall']:>6.1%}  {r['memories_per_session']:>8.2f}  {r['requests']:>8}"
        )
    print()
    for c in pairwise:
        print(
            f"{c['from']} -> {c['to']}: +{c['gained']} / -{c['lost']} specifics, p = {c['p']:.3g}"
        )
    if not result["cohort_matches"]:
        print(
            "\ncohort digest does not match the archive: different corpus, numbers not comparable",
            file=sys.stderr,
        )
    if stale:
        print(f"\nprompts have moved since these arms ran: {', '.join(stale)}", file=sys.stderr)
    if write:
        print(f"\n→ {OUT.relative_to(REPO)}")
    return result


def restore() -> int:
    """Refill the CLI's cache so `chronomem ingest fidelity --batch N` costs nothing."""
    data = _load_archive()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for arm, held in data["arms"].items():
        path = CACHE_DIR / f"{held['fingerprint']}.json"
        path.write_text(json.dumps(held["extractions"]), encoding="utf-8")
        print(f"{arm} → {path.relative_to(REPO)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=("archive", "score", "restore"))
    args = ap.parse_args()
    if args.command == "archive":
        return archive()
    if args.command == "restore":
        return restore()
    score()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
