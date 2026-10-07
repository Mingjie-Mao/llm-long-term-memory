"""One public train150 page diagnostic, preserving raw invalid responses and usage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
from analysis_io import sha256_file  # noqa: E402


def main():
    from llm_long_term_memory import cli
    from llm_long_term_memory.conversation import AnswerRequest
    from llm_long_term_memory.evaluation.datasets.longmemeval import load
    from llm_long_term_memory.runtime.evidence_pages import (
        PageSelectionError,
        archive_pages,
        select_page,
    )
    from llm_long_term_memory.store import external_session_id

    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--label", default="page-selection-diagnostic-v1")
    args = parser.parse_args()
    if not args.label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in args.label):
        raise ValueError("invalid namespace")
    _, settings, runner, _, usage = cli._build(
        "two_stage_raw_primary_grounded_v18",
        "configs/fallback.yaml",
        "train150",
        client_override=None if args.execute else object(),
        read_only_store=True,
    )
    try:
        dataset = (settings.data_dir / "longmemeval_s_cleaned.json").resolve()
        if (
            sha256_file(dataset)
            != "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
        ):
            raise ValueError("public corpus changed")
        instance = next(i for i in load("s", settings.data_dir) if i.question_id == "b46e15ed")
        page = archive_pages(
            runner.store, instance.question_id, roles=("user",), split_oversized=True
        ).pages[0]
        original = {
            (s.session_id, n): (t.role, t.content)
            for s in instance.sessions
            for n, t in enumerate(s.turns)
        }
        for source in page.sources:
            if original[(external_session_id(source.session_id), source.turn_index)] != (
                source.role,
                source.text,
            ):
                raise ValueError("not exact public source")
        request = AnswerRequest(instance.question, instance.question_date, instance.question_id)
        identity = {
            "public_dataset_sha256": sha256_file(dataset),
            "question_id": instance.question_id,
            "experiment_class": "exposed-train150-one-page-failure-diagnostic",
            "model": runner.model,
            "page": page.audit(),
            "prereg_sha256": sha256_file(REPO / "results/prereg-page-diagnostic-v1.md"),
            "r2_prereg_sha256": sha256_file(REPO / "results/prereg-page-diagnostic-v2.md")
            if args.label.endswith("-v2")
            else None,
            "sources": {
                p: sha256_file(REPO / p)
                for p in (
                    "src/llm_long_term_memory/runtime/evidence_pages.py",
                    "src/llm_long_term_memory/runtime/grounded_answering.py",
                    "tools/page_selection_diagnostic.py",
                )
            },
        }
        if not args.execute:
            print(json.dumps({"provider_calls": 0, "identity": identity}))
            return 0
        output = REPO / f"results/analysis/{args.label}.json"
        if output.exists():
            raise ValueError("namespace exists")
        runner.client.max_retries = 2
        runner.client.max_transport_retries = 2
        try:
            try:
                chosen, response, verdict = select_page(runner.client, runner.model, page, request)
                result = {
                    "accepted": True,
                    "chosen": chosen,
                    "verdict": verdict.model_dump(),
                    "provider_response": response.text,
                }
            except PageSelectionError as exc:
                result = {
                    "accepted": False,
                    "cause": str(exc),
                    "provider_response": exc.response.text,
                }
        finally:
            usage.save(REPO / f"results/raw/{args.label}.usage.json")
        if any(sha256_file(REPO / p) != sha for p, sha in identity["sources"].items()):
            raise ValueError("source changed during diagnostic")
        output.write_text(
            json.dumps({"identity": identity, "result": result, "usage": usage.summary()}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result))
    finally:
        runner.store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
