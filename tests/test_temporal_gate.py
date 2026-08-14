"""The gate that decides whether an ingest is worth its quota.

The scoring is what these test. A gate that reports the wrong number is worse than
no gate: it authorises spending on a pipeline that has not earned it.
"""

from __future__ import annotations

from chronomem.ingest.keying import Keying, UpdateOp
from chronomem.ingest.temporal_gate import (
    THRESHOLDS,
    score,
    stability_between,
    unusable,
)
from chronomem.ingest.temporal_pairs import ALL_PAIRS, COEXISTENCE, REPLACEMENT


def k(key: str, op: UpdateOp = UpdateOp.COEXISTS) -> Keying:
    return Keying(temporal_key=key, update_op=op, object="x")


def test_a_keyer_that_always_replaces_fails_on_false_supersede():
    """The reason the benchmark holds more coexistence pairs than replacements: a
    system that says `replaces` every time would score perfect recall."""
    keyings = [(k("attr"), k("attr", UpdateOp.REPLACES)) for _ in ALL_PAIRS]
    report = score(ALL_PAIRS, keyings)

    assert report.replacement_recall == 1.0
    assert report.false_supersede == 1.0
    assert not report.gate_open, "perfect recall must not open the gate on its own"


def test_a_keyer_that_never_replaces_fails_on_recall_but_is_safe():
    keyings = [(k("attr"), k("attr")) for _ in ALL_PAIRS]
    report = score(ALL_PAIRS, keyings)

    assert report.false_supersede == 0.0
    assert report.replacement_recall == 0.0
    assert not report.gate_open


def test_key_consistency_ignores_the_coexistence_pairs():
    """Two coexisting facts may legitimately share a key or not — scoring them as
    consistency failures would penalise correct behaviour."""
    keyings = []
    for pair in ALL_PAIRS:
        if pair.expected == "coexists":
            keyings.append((k("one"), k("another")))  # deliberately different
        else:
            keyings.append((k("same"), k("same", UpdateOp(pair.expected))))
    report = score(ALL_PAIRS, keyings)

    assert report.key_consistency == 1.0


def test_a_split_key_shows_up_as_inconsistency():
    """The rule-based failure: "uses TensorFlow" keyed `uses_tool` and "switched to
    PyTorch" keyed `prefers`, so no supersede could ever fire."""
    keyings = []
    for pair in ALL_PAIRS:
        op = UpdateOp(pair.expected)
        keyings.append((k("before_key"), k("after_key", op)))
    report = score(ALL_PAIRS, keyings)

    assert report.key_consistency == 0.0
    assert report.replacement_recall == 1.0, "ops can be right while keys are wrong"
    assert not report.gate_open


def test_a_perfect_keyer_opens_the_gate():
    keyings = [(k("attr"), k("attr", UpdateOp(p.expected))) for p in ALL_PAIRS]
    report = score(ALL_PAIRS, keyings, stability=1.0)

    assert report.key_consistency == 1.0
    assert report.replacement_recall == 1.0
    assert report.false_supersede == 0.0
    assert report.gate_open


def test_one_false_supersede_in_ten_exceeds_the_threshold():
    """1.5% is tight on purpose: retiring a fact that is still true removes it from
    every later query, and there are only ten coexistence pairs to fail."""
    keyings = []
    for i, pair in enumerate(ALL_PAIRS):
        op = (
            UpdateOp.REPLACES if pair.expected == "coexists" and i == 8 else UpdateOp(pair.expected)
        )
        keyings.append((k("attr"), k("attr", op)))
    report = score(ALL_PAIRS, keyings, stability=1.0)

    assert report.false_supersede > THRESHOLDS["false_supersede"]
    assert not report.gate_open


def test_stability_counts_key_and_op_together():
    first = [k("a", UpdateOp.REPLACES), k("b")]
    same = [k("a", UpdateOp.REPLACES), k("b")]
    drifted_key = [k("a", UpdateOp.REPLACES), k("c")]
    drifted_op = [k("a", UpdateOp.COEXISTS), k("b")]

    assert stability_between(first, same) == 1.0
    assert stability_between(first, drifted_key) == 0.5
    assert stability_between(first, drifted_op) == 0.5


def test_unkeyable_facts_are_counted():
    assert unusable([k("attr"), k("states", UpdateOp.NONE), k("", UpdateOp.NONE)]) == 2


def test_the_benchmark_is_weighted_toward_the_expensive_mistake():
    """More pairs that must not replace than pairs that must, so `false supersede`
    has a real denominator."""
    assert len(COEXISTENCE) > len(REPLACEMENT)


def test_report_is_serializable_as_a_reproducible_gate_artifact():
    report = score(
        ALL_PAIRS,
        [(k("attr"), k("attr", UpdateOp(pair.expected))) for pair in ALL_PAIRS],
        stability=1.0,
    )

    artifact = report.to_dict()
    assert artifact["gate_open"]
    assert artifact["thresholds"] == THRESHOLDS
    assert len(artifact["outcomes"]) == len(ALL_PAIRS)
