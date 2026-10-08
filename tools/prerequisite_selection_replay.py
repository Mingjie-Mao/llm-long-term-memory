"""Verify v17 calculator/gap contracts against saved v15 selections; zero calls.

This does NOT test the new page selector or report new answer accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]

from analysis_io import rows_by_question, sha256_file  # noqa: E402
from grounded_operand_replay import restore_ledger  # noqa: E402

from llm_long_term_memory.conversation import AnswerRequest  # noqa: E402
from llm_long_term_memory.evaluation.datasets.longmemeval import load  # noqa: E402
from llm_long_term_memory.runtime.grounded_answering import (  # noqa: E402
    FocusSessionEvidenceLedger,
    GroundedAnswererV18,
    SessionEvidenceLedger,
)
from llm_long_term_memory.store import SQLiteMemoryStore  # noqa: E402


def measure():
    answers_path = REPO / "results/raw/grounded-reader-v15.answers.jsonl"
    acceptance = json.loads(
        (REPO / "results/analysis/grounded-reader-v15.acceptance-final.json").read_text(
            encoding="utf-8"
        )
    )
    assert acceptance["inputs"][str(answers_path.relative_to(REPO))] == sha256_file(answers_path)
    inventory = json.loads(
        (REPO / "results/analysis/grounded-reader-v15.execution.json").read_text(encoding="utf-8")
    )
    store_path = REPO / "stores/train150.db"
    assert inventory["files"]["stores/train150.db"] == sha256_file(store_path)
    expected = set()
    for name in ("reasoning-errors", "correct-sample10"):
        expected.update(
            json.loads(
                (REPO / f"results/manifests/train150-raw-v1-{name}.json").read_text(
                    encoding="utf-8"
                )
            )["question_ids"]
        )
    rows = rows_by_question(answers_path)
    assert set(rows) == expected and len(rows) == 19
    instances = {i.question_id: i for i in load("s", REPO / "data")}
    store = SQLiteMemoryStore(store_path, read_only=True)
    store.initialize()
    engine = object.__new__(GroundedAnswererV18)
    engine.store, engine.chars_per_token = store, 4.6
    output = []
    try:
        for qid, row in rows.items():
            instance = instances[qid]
            request = AnswerRequest(instance.question, instance.question_date, qid)
            focus = ()
            for saved in row["answer"]["notes"]["grounded_calls"]:

                def ledger_type(*, sources, max_tokens, focus_ids=focus):
                    return (
                        FocusSessionEvidenceLedger(
                            sources=sources, max_tokens=max_tokens, focus_ids=focus_ids
                        )
                        if focus_ids
                        else SessionEvidenceLedger(sources=sources, max_tokens=max_tokens)
                    )

                ledger = restore_ledger(store, qid, saved["evidence"], ledger_type=ledger_type)
                parsed = engine.parse_verdict(saved["provider_response"], ledger, request)
                result = engine.validate_verdict(parsed, ledger, request)
                assert (
                    result.computed == saved["computed"] and result.detail == saved["calculation"]
                )
                focus = tuple(saved.get("calculation", {}).get("focus_sources", []))
            text = (
                (result.answer if result.computed else parsed.answer.strip())
                if engine.can_answer(parsed, result)
                else "I do not know."
            )
            assert text == row["answer"]["text"]
            gap = engine.completion_notes(ledger, request, parsed, result, text)["evidence_gap"]
            if qid == "f9e8c073":
                assert gap["missing_fields"] == ["report_period", "overlap_between_reports"]
            elif qid == "gpt4_2f8be40d":
                assert gap["missing_fields"] == ["event_identity", "event_date", "report_period"]
            elif text != "I do not know.":
                assert gap is None
            output.append(
                {"question_id": qid, "original_answer_preserved": True, "evidence_gap": gap}
            )
    finally:
        store.close()
    return {
        "experiment_class": "saved-selection-regression",
        "provider_calls": 0,
        "questions": 19,
        "pass": True,
        "rows": output,
        "inputs": {"saved_readers": sha256_file(answers_path), "store": sha256_file(store_path)},
        "generator_sha256": sha256_file(Path(__file__)),
        "limitations": "Same saved source pools/choices, NOT new page-selection or QA accuracy.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="prerequisite-selection-v2")
    args = parser.parse_args()
    output = REPO / f"results/analysis/{args.label}.train19.json"
    if output.exists():
        raise SystemExit("Refusing to overwrite prior evidence")
    result = measure()
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pass": result["pass"], "questions": 19, "provider_calls": 0}))
