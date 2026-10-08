"""Zero-provider replay of saved operands against hash-verified train150 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import rows_by_question, sha256_file, write_report  # noqa: E402

from llm_long_term_memory.runtime.grounded_answering import (  # noqa: E402
    EvidenceLedger,
    EvidenceSource,
    GroundedVerdict,
    calculate,
    calculate_v2,
)


def restore_ledger(
    store, user_id: str, audit: dict, *, ledger_type=EvidenceLedger
) -> EvidenceLedger:
    memories = {m.id: m for m in store.iter_all(user_id)}
    sources = []
    for metadata in audit["sources"]:
        session = store.get_session(metadata["session_id"]) if metadata["session_id"] else None
        if session is not None and session.user_id != user_id:
            raise ValueError("source provenance belongs to another tenant")
        if metadata["kind"] == "memory":
            memory = memories.get(metadata["source_id"])
            if memory is None or memory.user_id != user_id:
                raise ValueError("memory missing or belongs to another tenant")
            text = memory.content
        elif metadata["kind"] == "raw":
            if session is None:
                raise ValueError("raw session missing")
            turns = {t.id: t for t in store.turns_for_session(session.id)}
            text = turns[metadata["source_id"]].content
        else:
            raise ValueError("unsupported source kind")
        if hashlib.sha256(text.encode()).hexdigest() != metadata["text_sha256"]:
            raise ValueError("saved source text changed")
        fields = {k: v for k, v in metadata.items() if k != "text_sha256"}
        sources.append(EvidenceSource(**fields, text=text))
    ledger = ledger_type(sources=sources, max_tokens=audit["max_tokens"])
    if ledger.audit()["context_sha256"] != audit["context_sha256"]:
        raise ValueError("saved context changed")
    return ledger


def analyse(raw_path: Path, store_path: Path, data_dir: Path) -> dict:
    from llm_long_term_memory.evaluation.datasets.longmemeval import load
    from llm_long_term_memory.store.sqlite import SQLiteMemoryStore

    rows = rows_by_question(raw_path)
    allowed = json.loads(
        (REPO / "results/manifests/train150-raw-v1-reasoning-errors.json").read_text(
            encoding="utf-8"
        )
    )["question_ids"]
    if not set(rows) <= set(allowed) or store_path.stem != "train150":
        raise ValueError("only exposed train150 diagnostic operands may be replayed")
    instances = {i.question_id: i for i in load("s", data_dir)}
    store = SQLiteMemoryStore(store_path, read_only=True)
    results = []
    try:
        for qid, row in rows.items():
            instance = instances[qid]
            saved = row["notes"]["grounded_calls"][-1]
            ledger = restore_ledger(store, instance.store_namespace, saved["evidence"])
            verdict = GroundedVerdict.model_validate(saved["verdict"])
            old = calculate(verdict, ledger, instance.question_date)
            new = calculate_v2(verdict, ledger, instance.question_date)
            results.append(
                {
                    "question_id": qid,
                    "operation": verdict.operation,
                    "v1": {"computed": old.computed, "answer": old.answer, "detail": old.detail},
                    "v2": {"computed": new.computed, "answer": new.answer, "detail": new.detail},
                    "context_sha256": saved["evidence"]["context_sha256"],
                }
            )
    finally:
        store.close()
    return {
        "experiment_class": "saved-operand-regression-replay",
        "provider_calls": 0,
        "rows": results,
        "newly_computable": sum(r["v2"]["computed"] and not r["v1"]["computed"] for r in results),
        "input_sha256": sha256_file(raw_path),
        "source_sha256": sha256_file(
            REPO / "src/llm_long_term_memory/runtime/grounded_answering.py"
        ),
        "generator_sha256": sha256_file(Path(__file__)),
        "interpretation": "Unchanged saved operands; no new model answers or accuracy estimate.",
    }


def render(result: dict) -> str:
    lines = [
        "# Saved operand replay v2",
        "",
        result["interpretation"],
        f"Provider calls: 0. Newly computable: {result['newly_computable']}.",
        "",
        "| Question | Operation | v1 | v2 |",
        "|---|---|---|---|",
    ]
    for row in result["rows"]:
        values = [row[k]["answer"] or row[k]["detail"].get("cause", "") for k in ("v1", "v2")]
        lines.append(f"| {row['question_id']} | {row['operation']} | {values[0]} | {values[1]} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.json_out, args.md_out):
        if path.exists() or any(part in {"frozen", "sealed"} for part in path.resolve().parts):
            raise SystemExit("refusing to overwrite or write frozen/sealed evidence")
    result = analyse(
        REPO / "results/raw/two_stage_raw_primary_grounded.grounded-diagnostic-v1-rep1.jsonl",
        REPO / "stores/train150.db",
        REPO / "data",
    )
    print(write_report(result, render, json_out=args.json_out, md_out=args.md_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
