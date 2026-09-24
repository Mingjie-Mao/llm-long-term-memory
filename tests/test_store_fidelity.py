"""The paired comparison `tools/store_fidelity.py` reports between two store readings."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from store_fidelity import pair


def test_pair_counts_only_flips_over_shared_specifics():
    before = {"a": True, "b": False, "c": False, "d": True}
    after = {"a": True, "b": True, "c": False, "e": True}

    assert pair(before, after) == {
        "shared": 3,
        "gained": 1,
        "lost": 0,
        "only_before": 1,
        "only_after": 1,
    }


def test_pair_sees_a_loss():
    assert pair({"a": True}, {"a": False})["lost"] == 1
