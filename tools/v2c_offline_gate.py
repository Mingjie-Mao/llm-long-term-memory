"""Audit v2c's deterministic plan on the frozen v2b gate without API calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import read_json, rows_by_question  # noqa: E402

MANIFEST = REPO / "results/manifests/v2c-gate8.json"
CONTROL = REPO / "results/raw/two_stage_memory_only.v2b-gate16.jsonl"
OUT_JSON = REPO / "results/analysis/v2c-offline-gate.json"
OUT_MD = REPO / "results/analysis/v2c-offline-gate.md"
STORE = REPO / "stores/v2b-gate16.db"

EXPECTED = {
    "58ef2f1c": "detail",
    "6a1eabeb": "update",
    "gpt4_45189cb4": "timeline",
    "778164c6": "detail",
}
CONTROLS = {"45dc21b6", "6613b389", "a96c20ee"}


def analyse() -> dict:
    from llm_long_term_memory.evaluation.datasets.longmemeval import load
    from llm_long_term_memory.evaluation.runners.v2c import plan
    from llm_long_term_memory.retrieve.fallback import RawFallback
    from llm_long_term_memory.store import SQLiteMemoryStore

    manifest = read_json(MANIFEST)
    ids = manifest["question_ids"]
    instances = {item.question_id: item for item in load(manifest["variant"], REPO / "data")}
    rows = rows_by_question(CONTROL)
    store = SQLiteMemoryStore(STORE, read_only=True)
    store.initialize()
    try:
        outcomes = []
        for qid in ids:
            selected = [
                memory
                for item in rows[qid]["notes"]["retrieval"]
                if (memory := store.get(item["memory_id"])) is not None
            ]
            outcome = plan(instances[qid].question, selected)
            detail_evidence = (
                RawFallback(store).recover_local_detail(
                    instances[qid].store_namespace, instances[qid].question, selected
                )
                if outcome.detail_hydration_reason
                else None
            )
            outcomes.append(
                {
                    "question_id": qid,
                    "question_type": instances[qid].question_type,
                    "suppressed_update_ids": outcome.suppressed_update_ids,
                    "timeline_ids": outcome.timeline_ids,
                    "detail_hydration_reason": outcome.detail_hydration_reason,
                    "detail_source_turns": [
                        f"{turn.session_id}:{turn.turn_index}"
                        for turn in (detail_evidence.turns if detail_evidence else [])
                    ],
                    "detail_source_has_valentines_day": bool(
                        detail_evidence
                        and any("Valentine's Day" in turn.content for turn in detail_evidence.turns)
                    ),
                    "detail_source_has_mango_salsa": bool(
                        detail_evidence
                        and any("Mango Salsa" in turn.content for turn in detail_evidence.turns)
                    ),
                    "selected_before": len(selected),
                    "selected_after": len(outcome.memories),
                    "timeline": outcome.context_note,
                }
            )
    finally:
        store.close()

    by_id = {row["question_id"]: row for row in outcomes}
    checks = {
        "detail_target_routed": bool(by_id["58ef2f1c"]["detail_hydration_reason"]),
        "detail_target_source_contains_exact_day": bool(
            by_id["58ef2f1c"]["detail_source_has_valentines_day"]
        ),
        "recommendation_target_routed": bool(by_id["778164c6"]["detail_hydration_reason"]),
        "recommendation_source_contains_exact_name": bool(
            by_id["778164c6"]["detail_source_has_mango_salsa"]
        ),
        "update_target_repaired": bool(by_id["6a1eabeb"]["suppressed_update_ids"]),
        "timeline_target_has_three_events": len(by_id["gpt4_45189cb4"]["timeline_ids"]) == 3,
        "controls_have_no_update_suppression": not any(
            by_id[qid]["suppressed_update_ids"] for qid in CONTROLS
        ),
        "controls_have_no_timeline_plan": not any(by_id[qid]["timeline_ids"] for qid in CONTROLS),
    }
    return {
        "calls": 0,
        "questions": len(ids),
        "checks": checks,
        "pass": all(checks.values()),
        "expected_routes": EXPECTED,
        "rows": outcomes,
    }


def render(result: dict) -> str:
    lines = [
        "# v2c zero-call plan gate",
        "",
        "**No model/API calls were made.**",
        "",
        f"Decision: **{'PASS' if result['pass'] else 'STOP'}**",
        "",
    ]
    for name, passed in result["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines += ["", "## Routes", ""]
    for row in result["rows"]:
        routes = []
        if row["suppressed_update_ids"]:
            routes.append(f"update suppress {len(row['suppressed_update_ids'])}")
        if row["timeline_ids"]:
            routes.append(f"timeline {len(row['timeline_ids'])}")
        if row["detail_hydration_reason"]:
            routes.append("detail hydration")
        lines.append(f"- `{row['question_id']}`: {', '.join(routes) if routes else 'unchanged'}")
    return "\n".join(lines) + "\n"


def main() -> None:
    result = analyse()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    OUT_MD.write_text(render(result))
    print(render(result), end="")


if __name__ == "__main__":
    main()
