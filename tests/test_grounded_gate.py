from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from grounded_gate import evaluate


@pytest.fixture
def fixture(tmp_path):
    types = {f"q{i}": "multi-session" if i < 50 else "temporal-reasoning" for i in range(100)}
    paths = {arm: [] for arm in ["baseline", "candidate"]}
    for arm, score in [("baseline", 80), ("candidate", 84)]:
        for rep in range(3):
            path = tmp_path / f"{arm}-{rep}.jsonl"
            rows = [
                dict(
                    question_id=q,
                    question_type=t,
                    correct=i < score,
                    context_tokens=5700,
                    answer_prompt_version="memory-aware-v2"
                    if arm == "baseline"
                    else "memory-grounded-v1",
                    notes={
                        "execution_identity": {
                            "inventory_sha256": "frozen",
                            "arm": arm,
                            "answerer": "gemini-3.5-flash-lite",
                        }
                    },
                )
                for i, (q, t) in enumerate(types.items())
            ]
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            paths[arm].append(path)
    return paths, types


def mutate(paths, change):
    path = paths["candidate"][0]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    change(rows)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_exactly_four_mean_points_pass_without_rounding_loss(fixture):
    paths, types = fixture
    result = evaluate(paths, types, "frozen")
    assert result["mean_net"] == 4
    assert result["pass"]


def test_v2_gate_requires_its_registered_prompt_on_every_repeat(fixture):
    paths, types = fixture
    for path in paths["candidate"]:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            row["answer_prompt_version"] = "memory-grounded-v2"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    assert evaluate(paths, types, "frozen", candidate_prompt="memory-grounded-v2")["pass"]
    mutate(paths, lambda rows: rows[0].update(answer_prompt_version="memory-grounded-v1"))
    with pytest.raises(ValueError, match="prompt version"):
        evaluate(paths, types, "frozen", candidate_prompt="memory-grounded-v2")


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("correct", "false", "correctness"),
        ("correct", 1, "correctness"),
        ("question_type", "foreign", "question type"),
        ("question_id", "foreign", "question ids"),
        ("answer_prompt_version", "memory-aware-v2", "prompt version"),
        ("context_tokens", -1, "context size"),
        ("context_tokens", float("nan"), "context size"),
    ],
)
def test_malformed_or_foreign_rows_are_refused(fixture, field, value, match):
    paths, types = fixture
    mutate(paths, lambda rows: rows[0].update({field: value}))
    with pytest.raises(ValueError, match=match):
        evaluate(paths, types, "frozen")


def test_duplicate_ids_are_not_silently_collapsed(fixture):
    paths, types = fixture
    mutate(paths, lambda rows: rows.append(rows[0]))
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(paths, types, "frozen")


def test_incomplete_runs_are_not_evaluated(fixture):
    paths, types = fixture
    mutate(paths, lambda rows: rows.pop())
    with pytest.raises(ValueError, match="incomplete"):
        evaluate(paths, types, "frozen")


@pytest.mark.parametrize(
    "field,value",
    [("inventory_sha256", "changed"), ("arm", "baseline"), ("answerer", "strong-model")],
)
def test_config_or_model_identity_mix_is_refused(fixture, field, value):
    paths, types = fixture
    mutate(paths, lambda rows: rows[0]["notes"]["execution_identity"].update({field: value}))
    with pytest.raises(ValueError, match="identities"):
        evaluate(paths, types, "frozen")


def test_one_lucky_repeat_does_not_pass_the_average_gate(fixture):
    paths, types = fixture
    for path in paths["candidate"][1:]:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for i, row in enumerate(rows):
            row["correct"] = i < 80
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert not evaluate(paths, types, "frozen")["pass"]
