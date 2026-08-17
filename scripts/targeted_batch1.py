"""Re-extract the sessions that lost a known fact, one session per request.

    python scripts/targeted_batch1.py          # plan and cost it, no calls
    python scripts/targeted_batch1.py --run    # spend the quota

The batch-size pilot measured that extraction at fifteen sessions per request
yields 2.7 memories per session against 12.0 at one, and that a session's position
in the batch causes it. What it could not say is whether any of the lost material
was material anyone would ask about. Average yield is not evidence that the
*answers* were in what went missing.

The fact-lineage audit supplies that. Seven of the fourteen dev50 failures lose a
named fact at extraction, and each one is written down here. So this re-extracts
only the gold evidence sessions behind those seven, at one session per request,
with the same model, prompt, schema and decoding, and asks a question fixed before
the run:

    not "did batch 1 produce more memories"
    but "did batch 1 produce *this* fact"

`WANTED` is the pre-registration. The pattern is a search aid and never the
verdict: every memory the session produces is printed, because three earlier
automated classifications in this project were confidently wrong and all three
were string matching standing in for reading.

**No store is written.** The clean P10 store is an input.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from llm_long_term_memory.config import ExperimentConfig, Settings  # noqa: E402
from llm_long_term_memory.evaluation.datasets import longmemeval as lme  # noqa: E402

STORE = REPO / "stores" / "two-stage-p10.db"
OUT = REPO / "results" / "raw" / "targeted-batch1.json"

# question -> (the fact production lost, a search aid, the sessions it lives in).
# Fixed before the run. Sessions are named rather than taken from
# `answer_session_ids` wholesale so the diagnostic pays only for what it needs.
WANTED = {
    "edced276": (
        "the Hawaii trip lasted 10 days",
        r"10[\s-]?day|ten[\s-]?day",
        ["answer_60e8941a_1"],
    ),
    "4adc0475": ("the user had two assists", r"assist", ["answer_6efce493_2"]),
    "37f165cf": ("the second novel was 416 pages", r"416", ["answer_6b9b2b1e_1"]),
    "73d42213": ("the clinic took two hours to reach", r"two hours|2 hours", ["answer_1881e7db_2"]),
    "c9f37c46": ("the user attended an open mic night", r"open mic", ["answer_cdba3d9f_2"]),
    "gpt4_7abb270c": (
        "the sixth museum visit",
        r"museum|gallery",
        ["answer_7093d898_2", "answer_7093d898_5"],
    ),
    "80ec1f4f": (
        "the February gallery visit, with its date",
        r"art cube|gallery|february|15th",
        ["answer_990c8992_2"],
    ),
}


def production_memories(conn, namespace: str, session_id: str) -> list[str]:
    return [
        c
        for (c,) in conn.execute(
            "SELECT content FROM memories WHERE source_session_id=? AND user_id=?",
            (session_id, namespace),
        )
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="spend quota; otherwise plan only")
    args = parser.parse_args()

    settings = Settings()
    cfg = ExperimentConfig.from_yaml(str(REPO / "configs" / "baselines.yaml"))
    instances = {
        i.question_id: i
        for i in lme.load(cfg.dataset_variant, settings.data_dir, limit=cfg.dataset_limit)
    }
    sessions_by_id = {s.session_id: s for i in instances.values() for s in i.sessions}

    jobs = [(q, s) for q, (_, _, sids) in WANTED.items() for s in sids]
    print(f"targeted batch=1: {len(jobs)} sessions, ~{len(jobs) * 2} extractor requests")
    for q, (fact, _, sids) in WANTED.items():
        print(f"  {q:14s} {fact:44s} {sids}")
    if not args.run:
        print("\nRe-run with --run to start.")
        return 0

    from llm_long_term_memory.ingest import TwoStageExtractor
    from llm_long_term_memory.llm import Limits, QuotaManager, UsageTracker
    from llm_long_term_memory.llm.client import DailyQuotaExhausted, GeminiClient

    quota = QuotaManager(
        state_dir=settings.store_dir / "quota",
        default=Limits(rpm=cfg.quota.rpm, tpm=cfg.quota.tpm, rpd=cfg.quota.rpd),
    )
    quota.load_learned()
    usage = UsageTracker()
    client = GeminiClient(settings.require_api_key(), quota=quota, usage=usage)
    extractor = TwoStageExtractor(client, cfg.models.extractor)

    conn = sqlite3.connect(f"file:{STORE}?mode=ro", uri=True)
    records = json.loads(OUT.read_text(encoding="utf-8"))["records"] if OUT.exists() else []
    done = {(r["question_id"], r["session_id"]) for r in records}

    try:
        for question_id, session_id in jobs:
            if (question_id, session_id) in done:
                continue
            fact, pattern, _ = WANTED[question_id]
            try:
                outcome = extractor.extract([sessions_by_id[session_id]])
            except DailyQuotaExhausted as exc:
                print(f"\n\033[33mdaily quota reached: {exc}\033[0m — progress saved")
                break
            produced = [m.content for m in outcome.memories if m.source_session_id == session_id]
            before = production_memories(conn, question_id, session_id)
            hit = [c for c in produced if re.search(pattern, c, re.I)]
            records.append(
                {
                    "question_id": question_id,
                    "session_id": session_id,
                    "wanted": fact,
                    "production_memories": before,
                    "batch1_memories": produced,
                    "pattern_hits": hit,
                }
            )
            print(f"\n=== {question_id} / {session_id} — wanted: {fact}")
            print(f"  production (batch 15): {len(before)} memories")
            for c in before:
                print(f"      {c}")
            print(f"  batch 1: {len(produced)} memories" + ("  <-- pattern hit" if hit else ""))
            for c in produced:
                mark = ">>" if c in hit else "  "
                print(f"   {mark} {c}")
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(
                json.dumps(
                    {
                        "note": (
                            "Targeted batch=1 re-extraction of the gold sessions behind the "
                            "seven dev50 failures that lose a named fact at extraction. Same "
                            "model, prompt, schema and decoding as production; only the batch "
                            "size differs. No store is written. `pattern_hits` is a search aid "
                            "— read `batch1_memories`."
                        ),
                        "usage": usage.summary().get("total_requests"),
                        "records": records,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
    finally:
        conn.close()

    print(f"\n→ {OUT}")
    recovered = sum(1 for r in records if r["pattern_hits"])
    print(
        f"pattern matched in {recovered} of {len(records)} sessions — read the text before "
        f"calling any of them recovered"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
