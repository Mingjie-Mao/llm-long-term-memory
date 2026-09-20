"""Verify that the v2b 16-question gate was selected without batch-8 outcomes.

This is deliberately a verifier, not a general sampler.  Its constants are the
pre-registered rule, so changing the rule makes the checked-in manifest fail.
It performs no network or model calls.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import read_jsonl  # noqa: E402
sys.path.insert(0, str(REPO / "src"))

SOURCE = REPO / "results/manifests/v3-reasoning48.json"
TARGET = REPO / "results/manifests/v2b-gate16.json"
BASELINES = tuple(
    REPO / f"results/sealed/v3-answer-pilot/v2-control.rep{i}.jsonl" for i in range(1, 4)
)
DATA = REPO / "data"
SELECTOR_TAG = "20260916-v2b-gate16-v1"
QUOTAS = {
    "temporal-reasoning": 4,
    "multi-session": 4,
    "knowledge-update": 2,
    "single-session-user": 2,
    "single-session-assistant": 2,
    "single-session-preference": 2,
}


def expected_ids() -> tuple[list[str], dict[str, int], dict[str, str]]:
    from llm_long_term_memory.evaluation.datasets.longmemeval import load
    from llm_long_term_memory.evaluation.manifest import load_manifest

    pool = load_manifest(SOURCE)
    pool_ids = set(pool.question_ids)
    types = {
        inst.question_id: inst.question_type
        for inst in load(pool.variant, DATA)
        if inst.question_id in pool_ids
    }
    if set(types) != pool_ids:
        raise AssertionError("source manifest contains ids absent from the local dataset")

    votes: dict[str, list[bool]] = defaultdict(list)
    for path in BASELINES:
        rows = read_jsonl(path)
        seen = set()
        for row in rows:
            qid = row["question_id"]
            if qid in pool_ids:
                if qid in seen:
                    raise AssertionError(f"duplicate {qid} in {path}")
                votes[qid].append(bool(row["correct"]))
                seen.add(qid)
        if seen != pool_ids:
            raise AssertionError(f"{path} does not contain the complete 48-question pool")

    majority_correct = {qid: sum(values) >= 2 for qid, values in votes.items()}

    def order(qid: str) -> str:
        payload = f"{SELECTOR_TAG}|{types[qid]}|{int(majority_correct[qid])}|{qid}"
        return hashlib.sha256(payload.encode()).hexdigest()

    selected: list[str] = []
    for question_type, quota in QUOTAS.items():
        bucket = [qid for qid in pool.question_ids if types[qid] == question_type]
        wrong = sorted((qid for qid in bucket if not majority_correct[qid]), key=order)
        right = sorted((qid for qid in bucket if majority_correct[qid]), key=order)
        used_wrong = min(len(wrong), quota // 2)
        chosen = wrong[:used_wrong]
        chosen.extend(right[: quota - len(chosen)])
        if len(chosen) < quota:
            chosen.extend(wrong[used_wrong : used_wrong + quota - len(chosen)])
        if len(chosen) != quota:
            raise AssertionError(f"not enough {question_type} questions for quota {quota}")
        selected.extend(chosen)

    return sorted(selected), {qid: sum(votes[qid]) for qid in pool_ids}, types


def verify() -> dict:
    from llm_long_term_memory.evaluation.manifest import load_manifest

    manifest = load_manifest(TARGET)
    expected, correct_counts, types = expected_ids()
    actual = list(manifest.question_ids)
    if actual != expected:
        raise AssertionError(
            "v2b-gate16 does not match its registered selector\n"
            f"expected: {expected}\nactual:   {actual}"
        )
    if manifest.seed != 20260916:
        raise AssertionError(f"unexpected manifest seed: {manifest.seed}")

    type_counts = Counter(types[qid] for qid in actual)
    if dict(type_counts) != QUOTAS:
        raise AssertionError(f"type quotas moved: {dict(type_counts)}")
    right = sum(correct_counts[qid] >= 2 for qid in actual)
    result = {
        "questions": len(actual),
        "majority_right_controls": right,
        "majority_wrong_targets": len(actual) - right,
        "type_counts": dict(type_counts),
        "selector_tag": SELECTOR_TAG,
    }
    return result


def main() -> int:
    result = verify()
    print(json.dumps(result, indent=2, sort_keys=True))
    print("v2b-gate16 manifest matches the pre-registered deterministic selector")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
