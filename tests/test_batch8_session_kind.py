"""Session-kind classification and standardisation in the batch-8 attribution."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from batch8_session_kind_attribution import session_kind, standardise


def test_session_kinds_follow_the_haystack_source_prefix():
    assert session_kind("answer_4be1b6b4_2") == "evidence"
    assert session_kind("sharegpt_mMdyu3t_0") == "sharegpt"
    assert session_kind("ultrachat_12345") == "ultrachat"
    assert session_kind("c9b1c309_4") == "simulated"
    assert session_kind("fb3ec1ff") == "simulated"


def test_standardise_reweights_per_kind_recall_by_another_composition():
    per_kind = {"simulated": {"recall": 0.6}, "sharegpt": {"recall": 0.1}}
    assert standardise(per_kind, {"simulated": 3, "sharegpt": 1}) == (0.6 * 3 + 0.1) / 4


def test_standardise_skips_kinds_this_cohort_never_saw():
    per_kind = {"simulated": {"recall": 0.6}}
    assert standardise(per_kind, {"simulated": 2, "sharegpt": 5}) == 0.6
