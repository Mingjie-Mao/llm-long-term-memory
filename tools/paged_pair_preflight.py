"""Aggregate dev100 page/call budget; no gold/errors inspected, zero API calls."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
from analysis_io import sha256_file  # noqa: E402
from paired_manifest_scope import select_manifest  # noqa: E402


def main():
    from llm_long_term_memory import cli
    from llm_long_term_memory.commands.common import manifest_instances
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.evaluation.manifest import load_manifest, require_claim
    from llm_long_term_memory.runtime.evidence_pages import archive_pages
    from llm_long_term_memory.runtime.grounded_answering import personal_archive_review

    manifest_path = REPO / "results/manifests/dev100.json"
    manifest = load_manifest(manifest_path)
    require_claim(manifest, "regression")
    instances = select_manifest(manifest_instances(manifest, Settings()), manifest.question_ids)
    _, _, runner, _, _ = cli._build(
        "two_stage_raw_primary_grounded_v18",
        "configs/fallback.yaml",
        "dev100",
        client_override=object(),
        read_only_store=True,
    )
    counts, tokens, complete, routed = [], [], 0, 0
    try:
        for instance in instances:
            review = personal_archive_review(instance.question)
            pages = (
                archive_pages(
                    runner.store, instance.store_namespace, roles=("user",), split_oversized=True
                )
                if review
                else None
            )
            routed += review
            complete += pages.complete if pages else True
            counts.append(len(pages.pages) if pages else 0)
            tokens.append(sum(int(len(p.render()) / 4.6) for p in pages.pages) if pages else 0)
    finally:
        runner.store.close()
    result = {
        "experiment_class": "development-context-only-preflight",
        "provider_calls": 0,
        "questions": len(counts),
        "user_review_questions": routed,
        "complete_archives": complete,
        "pages_per_candidate_run": sum(counts),
        "median_pages_all_questions": statistics.median(counts),
        "median_review_source_tokens": statistics.median(tokens),
        "max_pages": max(counts),
        "nominal_calls_upper_before_client_retries": 3 * (sum(counts) + 4 * len(counts)),
        "client_attempt_upper": 6 * (sum(counts) + 4 * len(counts)),
        "expected_reader_calls": (
            "600 answers plus three candidate page reviews, plus selective rereads/failures"
        ),
        "manifest_sha256": sha256_file(manifest_path),
        "source_sha256": sha256_file(
            REPO / "src/llm_long_term_memory/runtime/grounded_answering.py"
        ),
        "generator_sha256": sha256_file(Path(__file__)),
        "limitations": (
            "No question-level gold or errors inspected. Source-token estimates exclude "
            "prompt/schema/provider tokenization."
        ),
    }
    out = REPO / "results/analysis/paged-paired-dev100-v18-v1.preflight-r2.json"
    if out.exists():
        raise ValueError("preflight namespace exists")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
