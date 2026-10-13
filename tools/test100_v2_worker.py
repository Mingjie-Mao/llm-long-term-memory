"""Isolated historical v2 reader; frozen inference with current transport limits."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "stores/experiments/test100-v2-source-restore-v1"
STEM = "original-test100-v2-t4-regression-v1"
sys.path[:0] = [str(ROOT / "src"), str(REPO / "tools")]
from analysis_io import rows_by_question, sha256_file  # noqa: E402


class Capture:
    def __init__(self, delegate):
        self.delegate, self.calls = delegate, []

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def generate(self, **kwargs):
        response = self.delegate.generate(**kwargs)
        schema = kwargs.get("schema")
        request = {k: v for k, v in kwargs.items() if k != "schema"}
        request["schema"] = schema.model_json_schema() if schema else None
        self.calls.append({"request": request, "response": response.text})
        return response


def main():
    from llm_long_term_memory import cli, store
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.evaluation.datasets import longmemeval as lme
    from llm_long_term_memory.llm import client
    from llm_long_term_memory.locking import exclusive

    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--rep", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    restored = json.loads(
        (REPO / "results/analysis/test100-v2-source-restore-v1.json").read_text(encoding="utf-8")
    )
    for rel, receipt in restored["restored_files"].items():
        if sha256_file(ROOT / rel) != receipt["sha256"]:
            raise ValueError("historical v2 source changed")
    if sha256_file(ROOT / "captured-v2.yaml") != restored["config_sha256"]:
        raise ValueError("historical v2 config changed")

    class ReadOnlyStore(store.SQLiteMemoryStore):
        def initialize(self):
            self._conn.execute("PRAGMA query_only=ON")

    store.SQLiteMemoryStore = ReadOnlyStore
    if not args.execute:

        class OfflineClient:
            def __init__(self, *a, **kw):
                pass

            def generate(self, **kwargs):
                return client.Completion(
                    '{"status":"answer","answer":"offline stub","notes":"composition only"}',
                    kwargs["model"],
                    0,
                    0,
                    0,
                    1,
                    0,
                )

        client.GeminiClient = OfflineClient
        Settings.require_api_key = lambda self: "offline-unused"
    _, settings, runner, _, usage = cli._build(
        "two_stage_fallback", str(ROOT / "captured-v2.yaml"), "test100"
    )
    try:
        runner.client.max_retries = runner.client.max_transport_retries = 2
        if args.execute:
            from google import genai
            from google.genai import types

            # Operational adapter only: the historical SDK client had no timeout.
            # Inference kwargs, source code, config, model and prompts remain old v2.
            runner.client._client = genai.Client(
                api_key=settings.require_api_key(), http_options=types.HttpOptions(timeout=180000)
            )
        runner.client = Capture(runner.client)
        manifest_path = REPO / "results/manifests/test100.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ids = manifest["question_ids"]
        if len(ids) != len(set(ids)) or len(ids) != 100:
            raise ValueError("exact original100 manifest required")
        loaded = {i.question_id: i for i in lme.load("s", REPO / "data")}
        instances = [loaded[qid] for qid in ids]
        if not args.execute:
            contexts = [runner.answer(i).context_tokens for i in instances]
            out = REPO / f"results/analysis/{STEM}.baseline-preflight.json"
            result = {
                "provider_calls": usage.total_requests,
                "questions": 100,
                "context_tokens": contexts,
                "restored_inventory_sha256": sha256_file(
                    REPO / "results/analysis/test100-v2-source-restore-v1.json"
                ),
                "worker_sha256": sha256_file(Path(__file__)),
                "manifest_sha256": sha256_file(manifest_path),
            }
            if out.exists() and json.loads(out.read_text(encoding="utf-8")) != result:
                raise ValueError("immutable preflight changed")
            if not out.exists():
                out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"provider_calls": usage.total_requests, "questions": 100}))
            return 0
        inventory_path = REPO / f"results/analysis/{STEM}.execution.json"
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        identity = sha256_file(inventory_path)
        for rel, digest in inventory["files"].items():
            if sha256_file(REPO / rel) != digest:
                raise ValueError(f"original-test100 frozen input changed: {rel}")
        base = REPO / f"results/raw/{STEM}-control-rep{args.rep}"
        rp, up = base.with_suffix(".readers.jsonl"), base.with_suffix(".usage.json")
        with exclusive(rp, what="historical v2 original-test100 readers"):
            saved = rows_by_question(rp) if rp.exists() else {}
            if saved and not up.exists():
                raise ValueError("missing baseline usage")
            if not set(saved) <= set(ids) or any(
                r["inventory_sha256"] != identity or r["cohort"] != "control"
                for r in saved.values()
            ):
                raise ValueError("foreign baseline reader")
            if up.exists():
                usage.records[:] = usage._load_records(up)
            with rp.open("a", encoding="utf-8") as sink:
                for i in instances:
                    if i.question_id in saved:
                        continue
                    if runner.client.quota.for_model(runner.model).remaining_today < 6:
                        print(
                            json.dumps(
                                {
                                    "status": "QUOTA_WAIT",
                                    "arm": "control",
                                    "rep": args.rep,
                                    "readers": len(saved),
                                }
                            ),
                            flush=True,
                        )
                        return 2
                    runner.client.calls.clear()
                    start, before = time.perf_counter(), len(usage.records)
                    try:
                        answer = runner.answer(i)
                    finally:
                        usage.save(up)
                    answer.notes.update(
                        {
                            "execution_identity": {
                                "inventory_sha256": identity,
                                "arm": "control",
                                "answerer": runner.model,
                            },
                            "actual_answer_calls": list(runner.client.calls),
                            "end_to_end_wall_ms": (time.perf_counter() - start) * 1000,
                            "whole_question_usage": {"requests": len(usage.records) - before},
                        }
                    )
                    row = {
                        "question_id": i.question_id,
                        "cohort": "control",
                        "grading_status": "UNGRADED",
                        "inventory_sha256": identity,
                        "answer_prompt_version": "memory-aware-v2",
                        "answer": asdict(answer),
                    }
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    sink.flush()
                    saved[i.question_id] = row
                    print(f"test100 control rep{args.rep} readers {len(saved)}/100", flush=True)
            return 0
    finally:
        runner.store.close()


if __name__ == "__main__":
    raise SystemExit(main())
