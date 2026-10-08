"""Prepare the nine registered public-benchmark reader requests locally, without APIs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import sha256_file  # noqa: E402
from grounded_context_replay import StubClient  # noqa: E402


class PreviewClient(StubClient):
    def __init__(self):
        self.requests = []

    def generate(self, **kwargs):
        self.requests.append(
            {k: v.model_json_schema() if k == "schema" else v for k, v in kwargs.items()}
        )
        if "observations" in kwargs["schema"].model_fields:
            from types import SimpleNamespace

            match = re.search(r"^\[(E\d+)\].*?\n(.+)", kwargs["prompt"], re.M)
            item = {"source": match[1], "interpretation": "Preview stub", "decision": "context"}
            if "operands" not in kwargs["schema"].model_fields:
                item = json.dumps(item | {"quote": match[2]})
            text = kwargs["schema"](
                observations=[item],
                reviewed_sources=[match[1]],
                scope_complete=True,
                status="answer",
                answer="stub",
            ).model_dump_json()
            return SimpleNamespace(text=text, input_tokens=0, output_tokens=0, api_latency_ms=0)
        return super().generate(**kwargs)


def main():
    from llm_long_term_memory import cli
    from llm_long_term_memory.commands.common import manifest_instances
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.conversation import AnswerRequest
    from llm_long_term_memory.evaluation.manifest import load_manifest

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--version", type=int, choices=(5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), default=5
    )
    parser.add_argument("--with-regression", action="store_true")
    args = parser.parse_args()
    if args.out.exists() or any(p in {"frozen", "sealed"} for p in args.out.resolve().parts):
        raise SystemExit("refusing overwrite or immutable evidence")
    path = REPO / "results/manifests/train150-raw-v1-reasoning-errors.json"
    manifest = load_manifest(path)
    settings = Settings()
    instances = {i.question_id: i for i in manifest_instances(manifest, settings)}
    ids = list(manifest.question_ids)
    manifest_paths = [path]
    if args.with_regression:
        regression_path = REPO / "results/manifests/train150-raw-v1-correct-sample10.json"
        regression = load_manifest(regression_path)
        instances.update({i.question_id: i for i in manifest_instances(regression, settings)})
        ids.extend(regression.question_ids)
        manifest_paths.append(regression_path)
        if len(ids) != len(set(ids)):
            raise SystemExit("overlapping cohorts")
    stub = PreviewClient()
    _, _, runner, _, _ = cli._build(
        f"two_stage_raw_primary_grounded_v{args.version}",
        "configs/fallback.yaml",
        "train150",
        client_override=stub,
        read_only_store=True,
    )
    rows = []
    try:
        for qid in ids:
            instance = instances[qid]
            first_request = len(stub.requests)
            runner.answer_request(
                AnswerRequest(instance.question, instance.question_date, instance.store_namespace)
            )
            rows.append({"question_id": qid, "reader_request": stub.requests[first_request]})
    finally:
        runner.store.close()
    result = {
        "provider_requests": 0,
        "purpose": "Local review of first reader calls, not new model answers or an execution.",
        "dataset_origin": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned",
        "outbound_destination_if_authorized": "original Gemini API",
        "answerer": "gemini-3.5-flash-lite",
        "judge": "gemma-4-31b-it",
        "reader_call_limit": 2 * len(ids),
        "judge_call_limit": len(ids),
        "limits_exclude_internal_provider_retries": True,
        "preview_does_not_predict_recovery_or_judge_requests": True,
        "manifest_sha256": sha256_file(path),
        "manifests": {str(p.relative_to(REPO)): sha256_file(p) for p in manifest_paths},
        "source_sha256": sha256_file(
            REPO / "src/llm_long_term_memory/runtime/grounded_answering.py"
        ),
        "generator_sha256": sha256_file(Path(__file__)),
        "rows": rows,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"provider_requests": 0, "preview_questions": len(rows), "output": str(args.out)}
        )
    )


if __name__ == "__main__":
    main()
