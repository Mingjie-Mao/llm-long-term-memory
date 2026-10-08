"""Replay v8 selections under v9 against original hash-verified train150 sources."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import rows_by_question, sha256_file, write_report  # noqa: E402
from grounded_operand_replay import restore_ledger  # noqa: E402

from llm_long_term_memory.conversation import AnswerRequest  # noqa: E402
from llm_long_term_memory.runtime.grounded_answering import (  # noqa: E402
    FocusSessionEvidenceLedger,
    GroundedAnswererV9,
    GroundedAnswererV10,
    GroundedAnswererV11,
    GroundedAnswererV12,
    GroundedAnswererV13,
    GroundedAnswererV14,
    GroundedAnswererV15,
    SessionEvidenceLedger,
)


def main():
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.evaluation.datasets.longmemeval import load
    from llm_long_term_memory.store.sqlite import SQLiteMemoryStore

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, choices=(9, 10, 11, 12, 13, 14, 15), default=9)
    parser.add_argument("--suffix", choices=("", "r2"), default="")
    args = parser.parse_args()
    suffix = "-" + args.suffix if args.suffix else ""
    output = REPO / f"results/analysis/grounded-selection-replay-v{args.version}{suffix}.json"
    markdown = output.with_suffix(".md")
    if output.exists() or markdown.exists():
        raise SystemExit("refusing overwrite")
    raw = REPO / f"results/raw/grounded-reader-v{args.version - 1}.answers.jsonl"
    inventory = REPO / f"results/analysis/grounded-reader-v{args.version - 1}.execution.json"
    identity = sha256_file(inventory)
    frozen = json.loads(inventory.read_text(encoding="utf-8"))
    store_path = REPO / "stores/train150.db"
    if sha256_file(store_path) != frozen["files"]["stores/train150.db"]:
        raise ValueError("store changed since original reader")
    rows = rows_by_question(raw)
    allowed = set()
    for name in ("reasoning-errors", "correct-sample10"):
        allowed.update(
            json.loads(
                (REPO / f"results/manifests/train150-raw-v1-{name}.json").read_text(
                    encoding="utf-8"
                )
            )["question_ids"]
        )
    if set(rows) != allowed:
        raise ValueError("expected the exact registered19 exposed train150 questions")
    instances = {i.question_id: i for i in load("s", Settings().data_dir)}
    engine = object.__new__(
        {
            9: GroundedAnswererV9,
            10: GroundedAnswererV10,
            11: GroundedAnswererV11,
            12: GroundedAnswererV12,
            13: GroundedAnswererV13,
            14: GroundedAnswererV14,
            15: GroundedAnswererV15,
        }[args.version]
    )
    results = []
    store = SQLiteMemoryStore(store_path, read_only=True)
    engine.store = store
    engine.chars_per_token = 4.6
    try:
        for qid, row in rows.items():
            if row["inventory_sha256"] != identity:
                raise ValueError("reader identity changed")
            instance = instances[qid]
            request = AnswerRequest(
                instance.question, instance.question_date, instance.store_namespace
            )
            attempts = []
            previous_focus = ()
            for saved in row["answer"]["notes"]["grounded_calls"]:

                def ledger_type(*, sources, max_tokens, focus_ids=previous_focus):
                    return (
                        FocusSessionEvidenceLedger(
                            sources=sources, max_tokens=max_tokens, focus_ids=focus_ids
                        )
                        if focus_ids
                        else SessionEvidenceLedger(sources=sources, max_tokens=max_tokens)
                    )

                ledger = restore_ledger(
                    store,
                    instance.store_namespace,
                    saved["evidence"],
                    ledger_type=ledger_type,
                )
                original_context = ledger.audit()["context_sha256"]
                enrichment = (
                    engine.extend_ledger(ledger, request, [], None) if args.version == 13 else {}
                )
                parsed = engine.parse_verdict(saved["provider_response"], ledger, request)
                calculation = engine.validate_verdict(parsed, ledger, request)
                focus = engine.render_context(ledger, request, 1, calculation)
                assert len(focus) <= ledger.max_tokens * ledger.chars_per_token
                attempts.append(
                    {
                        "original_context_sha256": original_context,
                        "replayed_context_sha256": ledger.audit()["context_sha256"],
                        "enrichment": enrichment,
                        "old_cause": saved.get("calculation", {}).get("cause"),
                        "old_computed": saved["computed"],
                        "new_cause": calculation.detail.get("cause"),
                        "new_computed": calculation.computed,
                        "new_code_answer": calculation.answer,
                        "new_can_answer": engine.can_answer(parsed, calculation),
                        "focus_sources": calculation.detail.get("focus_sources", []),
                        "focused_context_tokens": int(len(focus) / ledger.chars_per_token),
                        "canonicalization": calculation.detail.get("canonicalization", []),
                    }
                )
                previous_focus = tuple(saved.get("calculation", {}).get("focus_sources", []))
            results.append({"question_id": qid, "cohort": row["cohort"], "attempts": attempts})
    finally:
        store.close()
    result = {
        "provider_calls": 0,
        "experiment_class": "saved-selection development/regression replay",
        "interpretation": (
            "Saved selections with registered pre-read bridge after original SHA verification. "
            "Not new model answers or accuracy."
            if args.version == 13
            else "Unchanged saved source pool and selections; not new model answers or accuracy."
        ),
        "input_sha256": sha256_file(raw),
        "inventory_sha256": identity,
        "source_sha256": sha256_file(
            REPO / "src/llm_long_term_memory/runtime/grounded_answering.py"
        ),
        "generator_sha256": sha256_file(Path(__file__)),
        "restore_tool_sha256": sha256_file(REPO / "tools/grounded_operand_replay.py"),
        "rows": results,
    }

    def render(data):
        lines = [
            f"# V{args.version} saved-selection replay",
            "",
            data["interpretation"],
            "Provider calls: 0.",
            "",
        ]
        for row in data["rows"]:
            lines.append(f"- {row['question_id']}: `{json.dumps(row['attempts'])}`")
        return "\n".join(lines) + "\n"

    print(write_report(result, render, json_out=output, md_out=markdown))


if __name__ == "__main__":
    main()
