"""Run the held-out hundred, once, on a store that is provably complete.

    python scripts/heldout_run.py          # check the gates, start nothing
    python scripts/heldout_run.py --run    # evaluate, if every gate passes

Five gates, and each of them has a specific way this could go wrong:

  1. the freeze still holds        a held-out number describes the frozen system
                                   or it describes nothing
  2. ingestion is complete         a partial store looks exactly like an extraction
                                   failure, and would make S1 dominance a
                                   self-fulfilling result
  3. the store is homogeneous      the mixed-store incident, which passed every
                                   structural check while being two systems
  4. the output file is free       so a second run cannot quietly overwrite the
                                   first and become "the" held-out result
  5. this is the first run         the whole value of the set is that it is unseen,
                                   and a re-run after reading the first is not a
                                   held-out measurement

Gate 5 is the one that cannot be enforced by code — a human can always delete the
result and run again. What this can do is make it deliberate rather than
accidental, and leave a record either way.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from llm_long_term_memory.config import Settings  # noqa: E402
from llm_long_term_memory.evaluation.datasets import longmemeval as lme  # noqa: E402
from llm_long_term_memory.ingest import namespaced_sessions  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))

STORE = "heldout100"
CONFIG = "configs/fallback.yaml"
MANIFEST = REPO / "results" / "manifests" / "heldout100.json"
VARIANT = "two_stage_hydrated"
LABEL = "heldout100"
OUT = REPO / "results" / "raw" / f"{VARIANT}.{LABEL}.jsonl"
VALID_ROLES = {"user", "assistant", "system"}


class Stop(Exception):
    pass


def ok(name: str, detail: str) -> None:
    print(f"  \033[32mPASS\033[0m  {name}: {detail}")


def stop(name: str, detail: str):
    print(f"  \033[31mSTOP\033[0m  {name}: {detail}")
    raise Stop(f"{name}: {detail}")


def expected_sessions(settings) -> int:
    ids = set(json.loads(MANIFEST.read_text(encoding="utf-8"))["question_ids"])
    by_id = {i.question_id: i for i in lme.load("s", settings.data_dir)}
    pairs = namespaced_sessions([by_id[q] for q in sorted(ids)])
    return len({s.session_id for _, s in pairs})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    settings = Settings()
    db = settings.store_dir / f"{STORE}.db"

    print(f"\033[1mheld-out\033[0m  {STORE} · {CONFIG} · one run, never repeated\n")
    try:
        # 1 — the freeze
        r = subprocess.run(
            [str(REPO / ".venv" / "bin" / "python"), str(REPO / "scripts" / "freeze.py")],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            stop("freeze", "the system has moved:\n" + r.stdout.strip())
        ok("freeze", "the system matches results/frozen/p10-final/freeze.json")

        # 2 — ingestion complete
        if not db.exists():
            stop("ingestion", f"{db.name} does not exist")
        want = expected_sessions(settings)
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            have = conn.execute("SELECT COUNT(DISTINCT session_id) FROM turns").fetchone()[0]
            namespaces = conn.execute("SELECT COUNT(DISTINCT user_id) FROM memories").fetchone()[0]
            memories = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            scope_null = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE scope IS NULL"
            ).fetchone()[0]
            roles = dict(conn.execute("SELECT source_role, COUNT(*) FROM memories GROUP BY 1"))
            stored_fp = conn.execute(
                "SELECT value FROM meta WHERE key='ingest_fingerprint'"
            ).fetchone()
        finally:
            conn.close()
        if have < want:
            stop(
                "ingestion",
                f"{have}/{want} sessions. A partial store is indistinguishable from an "
                f"extraction failure, so the run would measure its own incompleteness. "
                f"Resume: lltm ingest run --config {CONFIG} --store-name {STORE} "
                f"--questions {MANIFEST.name}",
            )
        ok("ingestion", f"{have}/{want} sessions, {namespaces} namespaces, {memories:,} memories")

        # 3 — homogeneous, and written by the frozen extractor
        frozen = json.loads(
            (REPO / "results" / "frozen" / "p10-final" / "freeze.json").read_text(encoding="utf-8")
        )
        if not stored_fp:
            stop("homogeneity", "the store carries no ingest_fingerprint")
        moved = [
            f"{k}: {v} -> {json.loads(stored_fp[0]).get(k)}"
            for k, v in frozen["extractor"].items()
            if json.loads(stored_fp[0]).get(k) != v
        ]
        if moved:
            stop("homogeneity", "not written by the frozen extractor: " + "; ".join(moved))
        ok("homogeneity: extractor", frozen["extractor"]["extractor_version"] + ", frozen match")
        if scope_null:
            stop("homogeneity", f"{scope_null:,} memories have scope NULL — a pre-P10 signature")
        ok("homogeneity: scope", "0 rows with scope NULL")
        bad = {r: n for r, n in roles.items() if r not in VALID_ROLES}
        if bad:
            stop("homogeneity", f"invalid source_role values: {bad}")
        ok("homogeneity: source_role", ", ".join(f"{r} {n:,}" for r, n in sorted(roles.items())))

        # 4 and 5 — the output, and whether this has already happened
        if OUT.exists():
            n = len([x for x in OUT.read_text(encoding="utf-8").splitlines() if x.strip()])
            stop(
                "first run",
                f"{OUT.name} already holds {n} rows. The held-out set is worth what it "
                f"is worth because it was unseen; re-running after reading the first "
                f"result is a development measurement wearing its name. Move the file "
                f"aside deliberately if that is really what you want.",
            )
        ok("first run", f"{OUT.name} does not exist")
        n_q = len(json.loads(MANIFEST.read_text(encoding="utf-8"))["question_ids"])
        ok("manifest", f"{MANIFEST.name}: {n_q} frozen question ids")
    except Stop:
        print("\n\033[31mGates failed. Nothing was run.\033[0m")
        return 1

    if not args.run:
        print("\n\033[33mEvery gate passes. Re-run with --run to spend the one shot.\033[0m")
        return 0

    print("\n\033[1m=== held-out evaluation ===\033[0m", flush=True)
    result = subprocess.run(
        [
            str(REPO / ".venv" / "bin" / "python"),
            "-m",
            "llm_long_term_memory.cli",
            "eval",
            "run",
            VARIANT,
            "--config",
            CONFIG,
            "--store-name",
            STORE,
            "--questions",
            str(MANIFEST),
            "--label",
            LABEL,
        ],
        cwd=REPO,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
