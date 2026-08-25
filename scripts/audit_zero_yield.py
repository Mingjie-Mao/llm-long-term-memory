"""Write a reproducible zero-yield audit for a LongMemEval ingest store."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from llm_long_term_memory.config import Settings  # noqa: E402
from llm_long_term_memory.evaluation.datasets import longmemeval as lme  # noqa: E402
from llm_long_term_memory.evaluation.manifest import load_manifest  # noqa: E402
from llm_long_term_memory.ingest.pipeline import IngestProgress  # noqa: E402
from llm_long_term_memory.ingest.zero_yield import (  # noqa: E402
    audit_zero_yield,
    memory_counts,
)
from llm_long_term_memory.ingest.zero_yield_report import (  # noqa: E402
    render_zero_yield_markdown,
    summarize_batch_pilot,
)
from llm_long_term_memory.store import SQLiteMemoryStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default="train150")
    parser.add_argument("--questions", default="results/manifests/train150.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument(
        "--pilot",
        type=Path,
        default=Path("results/raw/batch-position-pilot.json"),
    )
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    settings = Settings()
    manifest = load_manifest(args.questions)
    by_id = {item.question_id: item for item in lme.load(manifest.variant, settings.data_dir)}
    instances = [by_id[question_id] for question_id in manifest.question_ids]
    progress = IngestProgress.load(settings.store_dir / f"{args.store}-ingest.json")
    store = SQLiteMemoryStore(settings.store_dir / f"{args.store}.db")
    store.initialize()
    try:
        counts = memory_counts(store, {item.question_id for item in instances})
    finally:
        store.close()

    # The production batch size is part of the store fingerprint.  Reading the
    # resolved value avoids silently analysing positions under the configured value
    # if a model TPM limit lowered it at runtime.
    store = SQLiteMemoryStore(settings.store_dir / f"{args.store}.db")
    store.initialize()
    try:
        fingerprint = json.loads(store.get_meta("ingest_fingerprint") or "{}")
    finally:
        store.close()
    batch_size = int(fingerprint.get("sessions_per_request", 15))
    report = audit_zero_yield(instances, progress, counts, batch_size=batch_size)
    payload = report.to_dict()
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key not in {"cases", "all_cases"}},
            indent=2,
        )
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"details: {args.output}")
    if args.require_complete and not report.complete:
        print("refusing final classification: ingestion is incomplete")
        return 2
    if args.markdown:
        try:
            pilot_payload = json.loads(args.pilot.read_text(encoding="utf-8"))
            pilot = summarize_batch_pilot(pilot_payload)
            markdown = render_zero_yield_markdown(
                payload,
                pilot,
                audit_source=str(args.output or "stdout"),
                pilot_source=str(args.pilot),
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"refusing zero-yield markdown: {exc}")
            return 2
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown, encoding="utf-8")
        print(f"summary: {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
