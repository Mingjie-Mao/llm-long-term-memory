"""A gate finer than the instrument's noise must be refused, including in the past.

The point of this check is that it would have stopped decisions that were actually made,
so the tests below are written as those decisions rather than as abstract thresholds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

gate = pytest.importorskip("check_gate_is_resolvable")

# The measured replicate pair: 38 of 142 verdicts changed between two runs of one
# configuration whose retrieval was byte-identical.
MEASURED_RATE = 27 / 142


def test_the_retired_one_probe_regression_rule_is_refused_on_every_stratum():
    """This is the rule v4.1 was judged against. On a 30-probe stratum the instrument
    reproduces a 5-probe swing by resampling the same arm, so a 1-probe threshold
    decided nothing."""
    for stratum in (30, 33, 49, 142):
        needed = gate.minimum_detectable_effect(MEASURED_RATE, stratum, repeats=1, sigmas=2.0)
        assert needed > 1, f"a 1-probe gate should not be readable on {stratum} probes"


def test_the_registered_v4_effect_does_not_clear_its_own_instrument():
    """v4.0-flat was selected on net +9 over 142 probes. Its replicate differs from it by
    net -7. The selection margin is inside the noise, which is why this file exists."""
    needed = gate.minimum_detectable_effect(MEASURED_RATE, 142, repeats=1, sigmas=2.0)
    assert needed > 9


def test_repeats_lower_the_bar_and_more_repeats_lower_it_further():
    """v2 and v3 ran three repeats per arm and their results replicate. That is the
    mechanism, not a coincidence."""
    one = gate.minimum_detectable_effect(MEASURED_RATE, 142, repeats=1, sigmas=2.0)
    three = gate.minimum_detectable_effect(MEASURED_RATE, 142, repeats=3, sigmas=2.0)
    five = gate.minimum_detectable_effect(MEASURED_RATE, 142, repeats=5, sigmas=2.0)
    assert one > three > five


def test_a_larger_stratum_needs_a_larger_net_but_a_smaller_share():
    """Net probes grow as sqrt(n) while the stratum grows as n, so pooling helps —
    stated explicitly because 'use a bigger stratum' is one of the tool's remedies."""
    small = gate.minimum_detectable_effect(MEASURED_RATE, 30, repeats=1, sigmas=2.0)
    large = gate.minimum_detectable_effect(MEASURED_RATE, 142, repeats=1, sigmas=2.0)
    assert large > small
    assert large / 142 < small / 30


def test_an_even_number_of_repeats_is_refused_rather_than_rounded():
    """Majority voting on an even number of runs has no majority. Silently rounding
    would report a bar the protocol cannot actually deliver."""
    with pytest.raises(ValueError):
        gate.minimum_detectable_effect(MEASURED_RATE, 142, repeats=2, sigmas=2.0)


def test_a_perfectly_stable_instrument_needs_no_margin():
    assert gate.minimum_detectable_effect(0.0, 142, repeats=1, sigmas=2.0) == 0.0


def test_the_rate_comes_from_the_recorded_pair_not_from_a_constant():
    noise = {"verdict_changed": 38, "wins": 10, "losses": 17, "n": 142}
    assert gate.disagreement_rate(noise) == pytest.approx(0.1901, abs=1e-4)


def test_switching_between_wrong_and_abstained_does_not_move_accuracy():
    noise = {"verdict_changed": 10, "wins": 0, "losses": 0, "n": 10}
    assert gate.disagreement_rate(noise) == 0


def test_homogeneous_repeat_model_can_overstate_the_improvement():
    # Half the probes always succeed; half are fair coins. The latter stay fair
    # after any odd-majority vote, so true disagreement stays 0.5 * 0.5 = 0.25.
    assert gate._flip_rate_after_repeats(0.25, 3) < 0.25


def test_the_committed_diagnosis_still_refuses_the_rule_it_retired():
    """Replay against the real artifact when it is present."""
    diagnosis = REPO / "results/analysis/v4-probe-diagnosis.json"
    if not diagnosis.is_file():
        pytest.skip("v4-probe-diagnosis.json is a local research artifact")
    noise = json.loads(diagnosis.read_text(encoding="utf-8"))["noise_floor"]
    rate = gate.disagreement_rate(noise)

    # Whatever the exact rate, a one-probe threshold on the smallest v4 stratum must not
    # be readable — that is the finding, and it should fail loudly if the numbers move.
    assert gate.minimum_detectable_effect(rate, 30, repeats=1, sigmas=2.0) > 1
