"""Produce the preregistered v2c.8 reasoning-48 confirmation report."""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import read_json, read_jsonl  # noqa: E402

ROWS = REPO / "results/raw/two_stage_v2c.reasoning48-v2c8.jsonl"
USAGE = REPO / "results/raw/two_stage_v2c.reasoning48-v2c8.usage.json"
INGEST_USAGE = REPO / "results/raw/v2c-reasoning48.ingest.usage.json"
ARCHIVED = REPO / "results/validation/v3-answer-pilot-aggregate.json"
OUT_JSON = REPO / "results/analysis/v2c-reasoning48-final.json"
OUT_MD = REPO / "results/analysis/v2c-reasoning48-final.md"

INVALID = "778164c6"
FAILURE_CLASS = {
    "false_premise": {"0ddfec37_abs", "gpt4_70e84552_abs"},
    "composition_or_arithmetic": {
        "a4996e51",
        "a9f6b44c",
        "d905b33f",
        "gpt4_2f8be40d",
        "gpt4_74aed68e",
        "gpt4_7a0daae1",
    },
    "source_detail_selection": {"58ef2f1c", "dd2973ad"},
    "retrieval_miss": {"e48988bc"},
}
THRESHOLDS = {
    "overall": 0.75,
    "knowledge-update": 0.875,
    "temporal-reasoning": 0.75,
    "multi-session": 0.50,
    "single-session-assistant": 0.75,
    "single-session-preference": 0.75,
    "single-session-user": 0.75,
    "median_context_tokens_max": 900,
    "source_session_recall": 0.95,
}


def ratio(rows: list[dict], field: str = "correct") -> float:
    return sum(bool(row[field]) for row in rows) / len(rows) if rows else 0.0


def analyse() -> dict:
    rows = read_jsonl(ROWS)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[row["question_type"]].append(row)
    type_accuracy = {kind: ratio(items) for kind, items in sorted(by_type.items())}
    median_context = statistics.median_high([row["context_tokens"] for row in rows])
    recall = sum(row["source_session_recalled"] for row in rows) / len(rows)
    configured_accuracy = ratio(rows)
    valid_rows = [row for row in rows if row["question_id"] != INVALID]

    checks = {
        "overall": configured_accuracy >= THRESHOLDS["overall"],
        **{kind: type_accuracy[kind] >= THRESHOLDS[kind] for kind in type_accuracy},
        "median_context_tokens": median_context <= THRESHOLDS["median_context_tokens_max"],
        "source_session_recall": recall >= THRESHOLDS["source_session_recall"],
    }

    archived = read_json(ARCHIVED)
    archived_arms = {arm["arm"]: arm for arm in archived["arms"]}
    wrong_ids = {row["question_id"] for row in rows if not row["correct"]}
    classified = set().union(*FAILURE_CLASS.values())
    if wrong_ids != classified:
        raise RuntimeError(
            f"failure taxonomy mismatch: unclassified={wrong_ids - classified}, "
            f"stale={classified - wrong_ids}"
        )

    usage = read_json(USAGE)["summary"]
    ingest_usage = read_json(INGEST_USAGE)["summary"]
    return {
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "questions": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "configured_judge_accuracy": configured_accuracy,
        "invalid_items": [INVALID],
        "valid_item_accuracy": ratio(valid_rows),
        "type_accuracy": type_accuracy,
        "median_context_tokens": median_context,
        "source_session_recall": recall,
        "fallback_levels": dict(Counter(row["notes"]["fallback_level"] for row in rows)),
        "checks": checks,
        "thresholds": THRESHOLDS,
        "failure_taxonomy": {name: sorted(ids) for name, ids in FAILURE_CLASS.items()},
        "comparison": {
            "v2_control_majority_accuracy": archived_arms["v2-control"]["majority_accuracy"],
            "v3_reasoned_majority_accuracy": archived_arms["v3-reasoned"]["majority_accuracy"],
            "delta_vs_v2_control_majority": (
                configured_accuracy - archived_arms["v2-control"]["majority_accuracy"]
            ),
            "delta_vs_v3_reasoned_majority": (
                configured_accuracy - archived_arms["v3-reasoned"]["majority_accuracy"]
            ),
            "note": "v2c.8 is one run; archived comparators are three-run majority results.",
        },
        "usage": {"ingest_increment": ingest_usage, "qa": usage},
    }


def render(result: dict) -> str:
    def pct(value: float) -> str:
        return f"{100 * value:.1f}%"

    lines = [
        "# v2c.8 reasoning-48 confirmation",
        "",
        f"Decision: **{result['decision']}**",
        "",
        f"- configured judge: **{result['correct']}/{result['questions']} "
        f"({pct(result['configured_judge_accuracy'])})**",
        f"- valid-item accuracy, excluding audited invalid `778164c6`: "
        f"**{pct(result['valid_item_accuracy'])}**",
        f"- median context: **{result['median_context_tokens']} tokens**",
        f"- source-session recall: **{pct(result['source_session_recall'])}**",
        "",
        "## Registered checks",
        "",
    ]
    for name, passed in result["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines += ["", "## Accuracy by type", ""]
    for kind, accuracy in result["type_accuracy"].items():
        lines.append(f"- `{kind}`: {pct(accuracy)}")
    comparison = result["comparison"]
    lines += [
        "",
        "## Historical directional comparison",
        "",
        f"- archived v2-control majority: {pct(comparison['v2_control_majority_accuracy'])}",
        f"- archived v3-reasoned majority: {pct(comparison['v3_reasoned_majority_accuracy'])}",
        f"- v2c.8 single run: {pct(result['configured_judge_accuracy'])}",
        "",
        "These are directional, not a paired significance claim: v2c.8 has one run, "
        "while each archived comparator is a three-run majority.",
        "",
        "## Eleven judged failures",
        "",
    ]
    labels = {
        "false_premise": "false-premise handling",
        "composition_or_arithmetic": "composition/arithmetic",
        "source_detail_selection": "source-detail selection",
        "retrieval_miss": "retrieval miss",
    }
    for name, ids in result["failure_taxonomy"].items():
        lines.append(f"- {labels[name]}: {len(ids)} — " + ", ".join(f"`{qid}`" for qid in ids))
    lines += [
        "",
        "No candidate changes or reruns are made from these 48 rows. The next honest "
        "measurement is a separately frozen final set.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    result = analyse()
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    OUT_MD.write_text(render(result))
    print(render(result), end="")


if __name__ == "__main__":
    main()
