"""Strict manifest and execution-identity checks for the registered grounded gate."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from pathlib import Path

from analysis_io import rows_by_question


def evaluate(
    paths: dict[str, list[Path]],
    types: dict[str, str],
    identity: str,
    candidate_prompt: str = "memory-grounded-v1",
) -> dict:
    if set(paths) != {"baseline", "candidate"} or any(len(p) != 3 for p in paths.values()):
        raise ValueError("exactly three runs per arm are required")
    if len(types) != 100:
        raise ValueError("the registered manifest must contain exactly 100 unique questions")
    runs = {}
    for arm, files in paths.items():
        runs[arm] = []
        for path in files:
            rows = rows_by_question(path)
            if set(rows) != set(types):
                raise ValueError(f"incomplete or foreign question ids: {path}")
            for qid, row in rows.items():
                if type(row["correct"]) is not bool or row["question_type"] != types[qid]:
                    raise ValueError(f"invalid correctness or question type: {path}, {qid}")
                expected = "memory-aware-v2" if arm == "baseline" else candidate_prompt
                if row.get("answer_prompt_version") != expected:
                    raise ValueError(f"wrong prompt version: {path}, {qid}")
                run_id = row["notes"].get("execution_identity", {})
                if (
                    run_id.get("inventory_sha256") != identity
                    or run_id.get("arm") != arm
                    or run_id.get("answerer") != "gemini-3.5-flash-lite"
                ):
                    raise ValueError(f"mixed execution identities: {path}, {qid}")
                context = row["context_tokens"]
                if type(context) not in {int, float} or not math.isfinite(context) or context < 0:
                    raise ValueError(f"invalid context size: {path}, {qid}")
            runs[arm].append(rows)
    deltas = {
        q: sum(r[q]["correct"] for r in runs["candidate"])
        - sum(r[q]["correct"] for r in runs["baseline"])
        for q in types
    }
    net = sum(deltas.values()) / 3
    by_type = {t: sum(deltas[q] for q in types if types[q] == t) / 3 for t in set(types.values())}
    context = statistics.median(r[q]["context_tokens"] for r in runs["candidate"] for q in types)
    rules = {
        "mean_net_at_least_4": net >= 4,
        "no_type_loses_more_than_2": all(v >= -2 for v in by_type.values()),
        "median_context_at_most_6000": context <= 6000,
    }
    return {
        "experiment_class": "regression",
        "mean_net": net,
        "mean_scores": {
            arm: sum(sum(row["correct"] for row in r.values()) for r in reps) / 3
            for arm, reps in runs.items()
        },
        "by_type_mean_net": by_type,
        "type_counts": dict(Counter(types.values())),
        "median_context": context,
        "rules": rules,
        "pass": all(rules.values()),
        "questions_up_down": {
            "up": sum(d > 0 for d in deltas.values()),
            "down": sum(d < 0 for d in deltas.values()),
        },
    }
