"""What a question set can detect, computed before the experiment rather than after.

Two mechanism experiments were registered against reasoning-48, run, and returned nets
of +0, +0 and -1, and both were then found to be below the set's resolution. The data
that predicts it was already on disk. These tests pin the arithmetic and, more
importantly, pin that the tool would have refused those designs.
"""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
REASONING48 = REPO / "results/raw/two_stage_v2c.reasoning48-v2c8.jsonl"


def _module():
    path = REPO / "tools/resolution.py"
    spec = importlib.util.spec_from_file_location("resolution_for_tests", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(tmp_path: Path, name: str, verdicts: dict[str, tuple[str, bool]]) -> Path:
    path = tmp_path / f"{name}.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "question_id": qid,
                    "question_type": kind,
                    "correct": correct,
                    "context_tokens": 100,
                }
            )
            for qid, (kind, correct) in verdicts.items()
        )
        + "\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("label", "family"),
    [
        ("heldout100-rep2", "heldout100"),
        ("v2b-gate16-repeat2", "v2b-gate16"),
        ("gate8-v2c3-rep1", "gate8-v2c3"),
        ("heldout100", "heldout100"),
    ],
)
def test_repeat_labels_group_onto_what_they_are_repeats_of(label, family):
    """The marker is inside the label, not after the variant.

    A first pass matched `-repeat2` but not `-rep2`, which silently dropped the only
    hundred-question repeat set and left the rates being computed from two nine-question
    ones — where a single flip reads as 50%.
    """
    assert _module()._REPEAT.sub("", label) == family


def test_a_configuration_that_disagrees_with_itself_is_counted_per_type(tmp_path):
    module = _module()
    first = _rows(tmp_path, "arm.set", {"q1": ("temporal-reasoning", True), "q2": ("x", True)})
    second = _rows(
        tmp_path, "arm.set-rep2", {"q1": ("temporal-reasoning", False), "q2": ("x", True)}
    )

    measured = module.measure({"arm.set": [first, second]})

    assert measured["rates"]["temporal-reasoning"]["rate"] == 1.0
    assert measured["rates"]["x"]["rate"] == 0.0
    assert measured["sources"]["arm.set"]["disagreed"] == 1


def test_the_projection_scales_the_measured_rates_by_the_target_composition():
    module = _module()
    rates = {
        "temporal-reasoning": {"unstable": 4, "n": 35, "rate": 4 / 35},
        "multi-session": {"unstable": 2, "n": 41, "rate": 2 / 41},
    }

    projected = module.project(rates, Counter({"temporal-reasoning": 10, "multi-session": 10}))

    assert projected["expected_disagreeing_questions"] == pytest.approx(10 * 4 / 35 + 10 * 2 / 41)
    assert projected["net_noise_sd"] == pytest.approx((10 * 4 / 35 + 10 * 2 / 41) ** 0.5)


def test_a_type_with_no_repeat_measurement_is_named_rather_than_assumed_stable():
    """Counting it as perfectly stable makes the answer an underestimate, so it is said."""
    module = _module()
    rates = {"temporal-reasoning": {"unstable": 4, "n": 35, "rate": 4 / 35}}

    projected = module.project(rates, Counter({"temporal-reasoning": 10, "brand-new": 10}))

    assert projected["types_with_no_repeat_measurement"] == ["brand-new"]


def test_a_perfectly_stable_set_can_register_any_difference():
    module = _module()
    rates = {"x": {"unstable": 0, "n": 50, "rate": 0.0}}

    projected = module.project(rates, Counter({"x": 48}))

    assert projected["expected_disagreeing_questions"] == 0
    assert projected["smallest_interpretable_net"] == 0


@pytest.mark.skipif(not REASONING48.is_file(), reason="the v2c rows are not in this checkout")
def test_the_tool_would_have_refused_both_experiments_that_were_run():
    """The retrospective the two quota days bought.

    v2e returned a net of +0, v5.0's fixed arm +0 and its planned arm -1. Every one of
    those is inside the noise this set carries, and the repeat runs that say so predate
    all three.
    """
    module = _module()
    families = module._repeat_families()
    measured = module.measure(families)

    import sys

    sys.path.insert(0, str(REPO / "tools"))
    from analysis_io import read_jsonl

    composition = Counter(row["question_type"] for row in read_jsonl(REASONING48))
    projected = module.project(measured["rates"], composition)

    assert projected["questions"] == 48
    assert projected["smallest_interpretable_net"] >= 3, (
        "if this drops to 2 the retrospective changes and the decision records need "
        "revisiting, because they say those runs could not have detected what they "
        "were registered for"
    )
    for observed_net in (0, 0, -1):
        assert abs(observed_net) < projected["smallest_interpretable_net"]
