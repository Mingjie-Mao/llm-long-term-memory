"""Run — or, by default, only cost — the query-time count candidate on reviewed questions.

Two arms over the same reviewed questions and the same bounded source pool:

* `control`  — the existing deterministic path: count the members the generator proposed.
* `candidate` — `runtime.evidence_count`: review every supplied source, quote each member,
  let code do the counting, and refuse when the pool cannot settle the answer.

Nothing here chooses a gold answer. The gold is the frozen human review; a run that has
no frozen review refuses to start. The default is a dry run: it builds every prompt,
reports the input size, and makes zero provider calls, so the price is known before the
pre-registered comparison is authorised.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from llm_long_term_memory.runtime.evidence_count import PROMPT_VERSION, Source, answer, make_prompt

REPO = Path(__file__).resolve().parent.parent
GOLD = REPO / "results/review/count-v1/gold.json"


def sources_of(question: dict) -> list[Source]:
    return [
        Source(id=s["id"], text=s["text"], role=s["role"], recorded_at=s.get("recorded_at"))
        for s in question["sources"]
    ]


def control_answer(question: dict) -> int:
    """The arm being improved on: count what the generator proposed, unreviewed."""
    return len({e.casefold() for e in question.get("proposed_entities", [])})


def dry_run(gold: dict) -> dict:
    rows = []
    for question in gold["questions"]:
        prompt = make_prompt(question["question"], sources_of(question))
        rows.append(
            {
                "id": question["id"],
                "sources": len(question["sources"]),
                # Same rough 4-chars-per-token rule the client uses to reserve quota.
                "estimated_input_tokens": len(prompt) // 4,
                "gold_answer": question["answer"],
            }
        )
    return {
        "mode": "dry_run",
        "provider_calls": 0,
        "prompt_version": PROMPT_VERSION,
        "gold_sha256": gold["gold_sha256"],
        "questions": len(rows),
        "estimated_input_tokens": sum(r["estimated_input_tokens"] for r in rows),
        "rows": rows,
    }


def execute(gold: dict, model: str) -> dict:
    from llm_long_term_memory.config import load_config
    from llm_long_term_memory.llm.client import GeminiClient

    config = load_config()
    client = GeminiClient(config)
    rows = []
    for question in gold["questions"]:
        result, completion = answer(question["question"], sources_of(question), client, model)
        rows.append(
            {
                "id": question["id"],
                "gold_answer": question["answer"],
                "control_answer": control_answer(question),
                "candidate_status": result.status,
                "candidate_answer": result.count,
                "candidate_members": list(result.members),
                "citations": list(result.citations),
                "usage": getattr(completion, "usage", None),
            }
        )
    answered = [r for r in rows if r["candidate_status"] == "answered"]
    return {
        "mode": "execute",
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "gold_sha256": gold["gold_sha256"],
        "questions": len(rows),
        "candidate_refused": len(rows) - len(answered),
        "candidate_exact": sum(1 for r in answered if r["candidate_answer"] == r["gold_answer"]),
        "control_exact": sum(1 for r in rows if r["control_answer"] == r["gold_answer"]),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=GOLD)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="spend quota; refused unless COUNT_PREREG names the registered plan",
    )
    args = parser.parse_args()
    if not args.gold.exists():
        print("STOP: no frozen human review; run count_review.py freeze first")
        return 2
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    if args.execute and not os.environ.get("COUNT_PREREG"):
        print("STOP: set COUNT_PREREG to the registered plan id before spending quota")
        return 2
    report = execute(gold, args.model) if args.execute else dry_run(gold)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
