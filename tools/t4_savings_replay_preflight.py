"""Freeze replay eligibility by exact requests/outputs, never by prior correctness."""

from __future__ import annotations

import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
from analysis_io import rows_by_question, sha256_file  # noqa: E402
from run_grounded_paired import combine  # noqa: E402
from run_paged_paired import save_json  # noqa: E402
from run_recursive_summary_cost import instances_for  # noqa: E402
from t4_event_count_acceptance import GAP_GOLD  # noqa: E402
from t4_savings_repair import SavingsRepairRunner  # noqa: E402
from t4_savings_replay import (  # noqa: E402
    ReplayMismatch,
    SavedReplyClient,
    grade_request_sha,
    row_sha,
)
from t4_selection_acceptance import CapturingClient  # noqa: E402
from t4_selection_preflight import OfflineClient  # noqa: E402

SOURCE = "t4-scoped-selection-train21-v3"
OUT = REPO / "results/analysis/t4-scoped-selection-train21-v4.replay-preflight-r2.json"


def historical():
    ip = REPO / f"results/analysis/{SOURCE}.execution.json"
    inventory = json.loads(ip.read_text(encoding="utf-8"))
    if any(sha256_file(REPO / p) != h for p, h in inventory["files"].items()):
        raise ValueError("historical source/config/data changed")
    if inventory["faithful_gap_targets"] != GAP_GOLD:
        raise ValueError("historical auxiliary gold changed")
    gate = json.loads((REPO / f"results/analysis/{SOURCE}.gate.json").read_text(encoding="utf-8"))
    if gate["inventory_sha256"] != sha256_file(ip) or any(
        sha256_file(REPO / p) != h for p, h in gate["inputs"].items()
    ):
        raise ValueError("historical gate bindings changed")
    rp, gp, xp = [
        REPO / f"results/raw/{SOURCE}.{s}"
        for s in ("readers.jsonl", "grades.jsonl", "gap-grades.jsonl")
    ]
    readers, grades, gaps = [rows_by_question(p) for p in (rp, gp, xp)]
    for i in instances_for("train150"):
        combine(
            i,
            readers[i.question_id],
            grades[i.question_id],
            sha256_file(ip),
            readers[i.question_id]["cohort"],
        )
        if i.question_id in gaps:
            combine(
                replace(i, answer=GAP_GOLD[i.question_id]),
                readers[i.question_id],
                gaps[i.question_id],
                sha256_file(ip),
                readers[i.question_id]["cohort"],
            )
    return (
        readers,
        grades,
        gaps,
        {str(p.relative_to(REPO)): sha256_file(p) for p in (ip, rp, gp, xp)},
    )


def proof_for(i, reader, grade, source_files):
    requests = {"original": grade_request_sha(i, reader["answer"]["text"])}
    if i.question_id in GAP_GOLD:
        requests["gap"] = grade_request_sha(
            replace(i, answer=GAP_GOLD[i.question_id]), reader["answer"]["text"]
        )
    return {
        "source_reader_sha256": row_sha(reader),
        "source_files": source_files,
        "judge_request_sha256": requests,
        "judge_prompt_version": grade["judge_prompt_version"],
        "kind": "historical exact requests/output/source audit; zero fresh generations",
    }


def main():
    from llm_long_term_memory import cli
    from llm_long_term_memory.conversation import AnswerRequest

    old, grades, _, files = historical()
    _, _, base, _, _ = cli._build(
        "two_stage_raw_primary_t4",
        "configs/fallback.yaml",
        "train150",
        client_override=OfflineClient(),
        read_only_store=True,
    )
    runner = SavingsRepairRunner.from_t4(base)
    eligible, fresh, rows = [], [], []
    try:
        for i in instances_for("train150"):
            q = i.question_id
            request = AnswerRequest(i.question, i.question_date, i.store_namespace)
            saved = SavedReplyClient(OfflineClient(), old[q])
            runner.client = CapturingClient(saved)
            try:
                answer = runner.answer_request(request)
                saved.verify_output(answer, old[q])
                if row_sha(answer.notes["selection_repair"]) != row_sha(
                    old[q]["answer"]["notes"]["selection_repair"]
                ):
                    raise ReplayMismatch("source/calculation audit changed")
                eligible.append(q)
                proof = proof_for(i, old[q], grades[q], files)
            except ReplayMismatch:
                fresh.append(q)
                runner.client = CapturingClient(OfflineClient())
                answer = runner.answer_request(request)
                proof = None
            rows.append(
                {
                    "question_id": q,
                    "replay_eligible": proof is not None,
                    "proof": proof,
                    "context_tokens": answer.context_tokens,
                    "calculation": answer.notes["selection_repair"].get("calculation"),
                }
            )
    finally:
        base.store.close()
    result = {
        "provider_calls": 0,
        "experiment_class": "exposed-train150-exact-request-replay-preflight",
        "selection_basis": "exact requests/output/source audit; never prior correctness",
        "replay_ids": eligible,
        "fresh_ids": fresh,
        "questions": 21,
        "rows": rows,
        "median_context": statistics.median(r["context_tokens"] for r in rows),
        "inputs": files
        | {
            str(p.relative_to(REPO)): sha256_file(p)
            for p in (
                Path(__file__),
                REPO / "tools/t4_savings_replay.py",
                REPO / "tools/t4_savings_repair.py",
            )
        },
        "limitations": "Historical replay, not new model samples; full comparison must be fresh.",
    }
    if OUT.exists() and json.loads(OUT.read_text(encoding="utf-8")) != result:
        raise ValueError("immutable replay plan changed")
    if not OUT.exists():
        save_json(OUT, result)
    print(
        json.dumps(
            {
                "provider_calls": 0,
                "replays": len(eligible),
                "fresh_ids": fresh,
                "median_context": result["median_context"],
            }
        )
    )


if __name__ == "__main__":
    main()
