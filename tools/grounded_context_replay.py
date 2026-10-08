"""Replay the actual candidate context with a stub reader; never calls a provider.

Only train150 may expose per-question reach here. dev100 uses --context-only and
does not inspect its gold evidence or answers. Reports are new development artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import sha256_file, write_report  # noqa: E402

BASELINE = "two_stage_raw_primary"
CANDIDATE = "two_stage_raw_primary_grounded"


class StubClient:
    calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        extra = {}
        if kwargs.get("schema") and "operands" in kwargs["schema"].model_fields:
            match = re.search(r"^\[(E\d+)\]", kwargs["prompt"], re.M)
            if match:
                extra["observations"] = [
                    {"source": match[1], "interpretation": "Stub", "decision": "context"}
                ]
        text = (
            kwargs["schema"](
                reviewed_sources=re.findall(r"^\[(E\d+)\]", kwargs["prompt"], re.M),
                scope_complete=True,
                status="answer",
                answer="stub",
                operation="lookup",
                **extra,
            ).model_dump_json()
            if kwargs.get("schema") and "reviewed_sources" in kwargs["schema"].model_fields
            else json.dumps({"status": "answer", "answer": "stub"})
        )
        return SimpleNamespace(text=text, input_tokens=0, output_tokens=0, api_latency_ms=0)


def analyse(
    store_name: str, manifest_path: Path, context_only: bool, candidate_variant: str = CANDIDATE
) -> dict:
    from llm_long_term_memory import cli
    from llm_long_term_memory.conversation import AnswerRequest
    from llm_long_term_memory.evaluation.datasets import longmemeval as lme
    from llm_long_term_memory.evaluation.manifest import load_manifest
    from llm_long_term_memory.retrieve.excerpts import archive_excerpts
    from llm_long_term_memory.store import external_session_id

    if not context_only and (store_name != "train150" or manifest_path.name != "train150.json"):
        raise ValueError("gold reach is permitted only for the exposed train150 manifest")
    manifest = load_manifest(manifest_path)
    client = StubClient()
    runners = {}
    try:
        for variant in (BASELINE, candidate_variant):
            cfg, settings, runner, _, _ = cli._build(
                variant,
                "configs/fallback.yaml",
                store_name,
                client_override=client,
                read_only_store=True,
            )
            runners[variant] = runner
        instances = {i.question_id: i for i in lme.load(manifest.variant, settings.data_dir)}
        rows, sizes, drops = [], {arm: [] for arm in runners}, []
        for qid in manifest.question_ids:
            instance = instances[qid]
            request = AnswerRequest(
                instance.question, instance.question_date, instance.store_namespace
            )
            answers = {arm: runner.answer_request(request) for arm, runner in runners.items()}
            for arm, answer in answers.items():
                sizes[arm].append(answer.context_tokens)
            audit = answers[candidate_variant].notes["grounded_calls"][0]["evidence"]
            drops.append(len(audit["dropped"]))
            if context_only:
                continue
            gold = {
                (s.session_id, i)
                for s in instance.sessions
                for i, t in enumerate(s.turns)
                if t.has_answer
            }
            if not gold:
                continue
            runner = runners[BASELINE]
            raw = archive_excerpts(runner.store, qid, instance.question, 4000)
            baseline = {(external_session_id(t.session_id), t.turn_index) for t in raw.turns}
            candidate = {
                (external_session_id(s["session_id"]), s["turn_index"])
                for s in audit["sources"]
                if s["kind"] == "raw"
            }
            # A selected id is not evidence if its stored body differs from the dataset.
            stored = {
                (external_session_id(t.session_id), t.turn_index): t.content
                for sid in runner.store.session_ids_for_user(qid)
                for t in runner.store.turns_for_session(sid)
            }
            original = {
                (s.session_id, i): t.content
                for s in instance.sessions
                for i, t in enumerate(s.turns)
            }
            mismatch = any(stored.get(key) != original[key] for key in gold)
            rows.append(
                {
                    "question_id": qid,
                    "type": instance.question_type,
                    "baseline": gold <= baseline and not mismatch,
                    "candidate": gold <= candidate and not mismatch,
                    "content_mismatch": mismatch,
                }
            )
        context = {
            arm: {
                "median": statistics.median(values),
                "p90": sorted(values)[int(len(values) * 0.9)],
                "max": max(values),
            }
            for arm, values in sizes.items()
        }
        result = {
            "experiment_class": "development-offline",
            "store": store_name,
            "manifest": str(manifest_path.relative_to(REPO)),
            "manifest_sha256": sha256_file(manifest_path),
            "questions": len(manifest.question_ids),
            "context_only": context_only,
            "context": context,
            "candidate_questions_with_dropped_sources": sum(d > 0 for d in drops),
            "provider_requests": 0,
            "provider_tokens": 0,
            "stub_calls": client.calls,
            "models": cfg.models.model_dump(),
            "config_sha256": sha256_file(REPO / "configs/fallback.yaml"),
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            "python": platform.python_version(),
            "candidate_prompt": runners[candidate_variant].answer_prompt_version,
            "candidate_source_sha256": sha256_file(
                REPO / "src/llm_long_term_memory/runtime/grounded_answering.py"
            ),
            "index_ids_sha256": hashlib.sha256(
                json.dumps(list(runners[candidate_variant].raw_primary_turn_index.ids)).encode()
            ).hexdigest(),
        }
        if rows:
            n = len(rows)
            wins = sum(r["candidate"] and not r["baseline"] for r in rows)
            losses = sum(r["baseline"] and not r["candidate"] for r in rows)
            result.update(
                {
                    "gold_questions": n,
                    "all_gold_turn_coverage": {
                        arm: sum(r[arm] for r in rows) / n for arm in ("baseline", "candidate")
                    },
                    "paired": {"wins": wins, "losses": losses},
                    "rows": rows,
                    "offline_gate": {
                        "coverage_gain_at_least_2pp": (wins - losses) / n >= 0.02,
                        "no_content_mismatch": not any(r["content_mismatch"] for r in rows),
                        "median_context_at_most_6000": context[candidate_variant]["median"] <= 6000,
                    },
                }
            )
        return result
    finally:
        for runner in runners.values():
            runner.store.close()


def render(result: dict) -> str:
    lines = [
        f"# Grounded candidate offline replay — {result['store']}",
        "",
        "Development evidence; stub reader, zero provider requests. Reach is not accuracy.",
        "",
        "| arm | median context | p90 | max |",
        "|---|---:|---:|---:|",
    ]
    for arm, values in result["context"].items():
        lines.append(f"| {arm} | {values['median']} | {values['p90']} | {values['max']} |")
    if "all_gold_turn_coverage" in result:
        lines += [
            "",
            f"All-gold-turn coverage: {result['all_gold_turn_coverage']}; "
            f"paired {result['paired']}.",
            "",
            f"Offline gates: {result['offline_gate']}.",
        ]
    lines += [
        "",
        "Questions with context-budget omissions: "
        f"{result['candidate_questions_with_dropped_sources']}.",
        "Both arms use configs/fallback.yaml and its unchanged model assignments.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--context-only", action="store_true")
    parser.add_argument(
        "--candidate",
        default=CANDIDATE,
        choices=(
            CANDIDATE,
            "two_stage_raw_primary_grounded_v2",
            "two_stage_raw_primary_grounded_v3",
            "two_stage_raw_primary_grounded_v4",
            "two_stage_raw_primary_grounded_v5",
            "two_stage_raw_primary_grounded_v6",
            "two_stage_raw_primary_grounded_v7",
            "two_stage_raw_primary_grounded_v8",
            "two_stage_raw_primary_grounded_v9",
            "two_stage_raw_primary_grounded_v10",
            "two_stage_raw_primary_grounded_v11",
            "two_stage_raw_primary_grounded_v12",
            "two_stage_raw_primary_grounded_v13",
            "two_stage_raw_primary_grounded_v14",
            "two_stage_raw_primary_grounded_v15",
            "two_stage_raw_primary_grounded_v16",
            "two_stage_raw_primary_grounded_v17",
            "two_stage_raw_primary_grounded_v18",
        ),
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.json_out, args.md_out):
        if path.exists() or any(p in {"frozen", "sealed"} for p in path.parts):
            raise SystemExit(f"refusing to overwrite or write immutable evidence: {path}")
    result = analyse(args.store, args.manifest.resolve(), args.context_only, args.candidate)
    print(write_report(result, render, json_out=args.json_out, md_out=args.md_out), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
