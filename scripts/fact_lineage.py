"""Trace every fact a failing question needed, layer by layer, until it disappears.

    python scripts/fact_lineage.py            # all failures, to stdout
    python scripts/fact_lineage.py 80ec1f4f   # one question, in full

`source_session_recalled` is 94% and every one of the fourteen dev50 failures has
it set. That statistic says the folder holding the answer was opened. It does not
say the right sheet of paper came out of it, and the difference is where these
questions are being lost.

So this dumps the four layers a fact has to survive, for the gold evidence
sessions of one question:

    1. raw            the original turns — is the fact in the conversation at all?
    2. structured     memories extracted from those turns — did the fact survive?
    3. context        the memories the answerer actually received — was it supplied?
    4. answer         what came out

The classification is deliberately **not** automated. An earlier attempt derived
required facts from the gold answer's wording and was worthless: the gold for one
question is "15 days", a number that appears nowhere in the conversation because
it is the sum of a five-day trip and a ten-day one. Matching gold text against
context measures whether the answer was already written down, which for a computed
answer is never. Fourteen questions is small enough to read.

This is a benchmark diagnostic and cannot run in production, where nothing knows
which facts an answer required. Production keeps watching zero-yield,
memories/session, yield by batch position and fallback rate; atomic-fact survival
is an offline measurement made against gold.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from llm_long_term_memory.config import ExperimentConfig, Settings  # noqa: E402
from llm_long_term_memory.evaluation.datasets import longmemeval as lme  # noqa: E402

STORE = REPO / "stores" / "two-stage-p10.db"
RESULT = REPO / "results" / "raw" / "two_stage_hydrated.a2-clean-p10-fallback-v2.jsonl"


def load():
    rows = [json.loads(x) for x in RESULT.read_text(encoding="utf-8").splitlines() if x.strip()]
    cfg = ExperimentConfig.from_yaml(str(REPO / "configs" / "baselines.yaml"))
    instances = {
        i.question_id: i
        for i in lme.load(cfg.dataset_variant, Settings().data_dir, limit=cfg.dataset_limit)
    }
    return rows, instances


def report(row: dict, inst, conn: sqlite3.Connection, full: bool) -> None:
    qid = row["question_id"]
    gold_sessions = list(inst.answer_session_ids)
    supplied = {m["memory_id"] for m in (row["notes"].get("retrieval") or [])}

    print("=" * 78)
    print(f"{qid}  [{row['question_type']}]   asked on {inst.question_date}")
    print(f"Q    {inst.question}")
    print(f"GOLD {inst.answer}")
    print(f"GOT  {row['hypothesis']}")
    print(
        f"fallback={row['notes'].get('fallback_level')} "
        f"status={row['notes'].get('answer_status')} "
        f"top_k={row['notes'].get('top_k')} gold_sessions={gold_sessions}"
    )

    for sid in gold_sessions:
        print(f"\n--- gold session {sid}")
        turns = conn.execute(
            "SELECT turn_index, role, content FROM turns WHERE session_id=? ORDER BY turn_index",
            (sid,),
        ).fetchall()
        print(f"  [1] raw: {len(turns)} turns")
        if full:
            for i, role, content in turns:
                print(f"      {i:2d} {role:9s} {content}")
        else:
            for i, role, content in turns:
                print(f"      {i:2d} {role:9s} {content[:150]}")

        mems = conn.execute(
            "SELECT id, user_id, content FROM memories WHERE source_session_id=? ORDER BY id",
            (sid,),
        ).fetchall()
        mine = [m for m in mems if m[1] == qid]
        print(
            f"  [2] structured: {len(mine)} memories in this namespace "
            f"({len(mems)} across all namespaces)"
        )
        for mid, _, content in mine:
            mark = "GIVEN " if mid in supplied else "  --  "
            print(f"      {mark} {content}")

        given = sum(1 for mid, ns, _ in mems if ns == qid and mid in supplied)
        print(f"  [3] context: {given} of this session's {len(mine)} memories reached the answerer")


def main() -> int:
    rows, instances = load()
    wanted = sys.argv[1:] if len(sys.argv) > 1 else None
    fails = [r for r in rows if not r["correct"]]
    if wanted:
        fails = [r for r in fails if r["question_id"] in wanted]

    conn = sqlite3.connect(f"file:{STORE}?mode=ro", uri=True)
    try:
        for row in fails:
            report(row, instances[row["question_id"]], conn, full=bool(wanted))
            print()
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
