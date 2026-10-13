"""Zero-call first-loss and scoped-repair replay on the fixed exposed train21."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
from analysis_io import rows_by_question, sha256_file  # noqa: E402
from run_paged_paired import save_json  # noqa: E402
from run_recursive_summary_cost import instances_for  # noqa: E402
from t4_savings_repair import SavingsRepairRunner  # noqa: E402
from t4_selection_acceptance import CapturingClient  # noqa: E402


class OfflineClient:
    def generate(self, **kwargs):
        from llm_long_term_memory.llm.client import Completion

        return Completion(
            '{"notes":"offline composition only","status":"answer","answer":"stub"}',
            kwargs["model"],
            0,
            0,
            0,
            1,
            0,
        )


def replay():
    from llm_long_term_memory import cli
    from llm_long_term_memory.conversation import AnswerRequest
    from llm_long_term_memory.ingest.extract import _parse_date
    from llm_long_term_memory.retrieve.excerpts import archive_excerpts
    from llm_long_term_memory.retrieve.time_notes import annotate

    source = REPO / "results/raw/t4-scoped-selection-train21-v3.readers.jsonl"
    old = rows_by_question(source)
    _, _, runner, _, _ = cli._build(
        "two_stage_raw_primary_t4",
        "configs/fallback.yaml",
        "train150",
        client_override=CapturingClient(OfflineClient()),
        read_only_store=True,
    )
    candidate = SavingsRepairRunner.from_t4(runner)
    rows, diagnoses = [], []
    causes = {
        "09ba9854": (
            "S5",
            "User train quote10 marked OUTSIDE in a price comparison; secondary fallback drops correct first-pass context.",
        ),
        "0ddfec37_abs": ("S5", "Different collection category used despite raw category evidence."),
        "51a45a95": (
            "S5",
            "Action place confused with coupon origin; USER same-session shopping present.",
        ),
        "eaca4986": (
            "S4b",
            "Second assistant artifact ranked but dropped during USER-preferred packing.",
        ),
        "f9e8c073": (
            "S0",
            "Unspecified and last-year quantity reports do not establish disjoint periods.",
        ),
        "gpt4_731e37d7": (
            "S0",
            "Reported completed March event conflicts with February session; year unresolved.",
        ),
        "gpt4_2f8be40d": (
            "S0",
            "Named and indirect/undated reports leave the total's identity/time scope unclear.",
        ),
    }
    try:
        for i in instances_for("train150"):
            request = AnswerRequest(i.question, i.question_date, i.store_namespace)
            answer = candidate.answer_request(request)
            rows.append(
                {
                    "question_id": i.question_id,
                    "context_tokens": answer.context_tokens,
                    "selection_audit": answer.notes["selection_repair"],
                    "program_result": answer.text if answer.text != "stub" else None,
                }
            )
            if i.question_id not in causes:
                continue
            ranked = runner.store.search_turns(request.user_id, request.question, limit=500)
            packed = archive_excerpts(
                runner.store,
                request.user_id,
                request.question,
                4000,
                asked_on=_parse_date(request.asked_on),
                time_notes=True,
                prefer_user=True,
                time_window=True,
            )
            refs = []
            for s in i.sessions:
                for n, t in enumerate(s.turns):
                    if not t.has_answer:
                        continue
                    internal = [r for r in ranked if r.content == t.content]
                    text = (
                        annotate(t.content, _parse_date(s.date))[0]
                        if t.role == "user"
                        else t.content
                    )
                    prompts = [
                        c["request"]["prompt"]
                        for c in old[i.question_id]["answer"]["notes"]["actual_answer_calls"]
                    ]
                    refs.append(
                        {
                            "session_id": s.session_id,
                            "turn": n,
                            "role": t.role,
                            "raw_sha256": __import__("hashlib")
                            .sha256(t.content.encode())
                            .hexdigest(),
                            "ranked_positions": [ranked.index(r) + 1 for r in internal],
                            "packed": any(r.id in {p.id for p in packed.turns} for r in internal),
                            "complete_text_in_original_request": any(text in p for p in prompts),
                        }
                    )
            stage, cause = causes[i.question_id]
            if i.question_id == "eaca4986" and not any(
                r["ranked_positions"] and not r["packed"] for r in refs
            ):
                raise ValueError("song composition diagnosis is not reproduced")
            diagnoses.append(
                {
                    "question_id": i.question_id,
                    "historical_first_failure": stage,
                    "cause": cause,
                    "question": i.question,
                    "old_answer": old[i.question_id]["answer"]["text"],
                    "reference_trace": refs,
                }
            )
    finally:
        runner.store.close()
    return {
        "provider_calls": 0,
        "experiment_class": "exposed-train150-offline-diagnostic",
        "questions": len(rows),
        "median_context": statistics.median(r["context_tokens"] for r in rows),
        "diagnoses": diagnoses,
        "rows": rows,
        "inputs": {
            str(p.relative_to(REPO)): sha256_file(p)
            for p in (
                source,
                REPO / "tools/t4_selection_repair.py",
                REPO / "tools/t4_event_count_repair.py",
                REPO / "tools/t4_savings_repair.py",
                Path(__file__),
                REPO / "stores/train150.db",
                REPO / "results/manifests/train150.json",
            )
        },
        "limitations": "Stub generation tests composition/provenance only; not accuracy. "
        "Historical first losses describe base t4; all prior rules/results retained. "
        "Full USER audit restores wedding gap in a new protocol.",
    }


if __name__ == "__main__":
    out = REPO / "results/analysis/t4-scoped-selection-train21-v4.preflight-r2.json"
    result = replay()
    if out.exists() and json.loads(out.read_text(encoding="utf-8")) != result:
        raise ValueError("preflight changed; choose a fresh output namespace")
    if not out.exists():
        save_json(out, result)
    print(json.dumps({k: result[k] for k in ("provider_calls", "questions", "median_context")}))
