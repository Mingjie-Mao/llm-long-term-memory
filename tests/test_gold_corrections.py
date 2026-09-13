"""The correction overlay must never edit the probe set, and must only ever remove members.

`synthesis-probes.json` is hashed into `v4-probe-split.json`, the held-out split that is
the only unseen check v4 has. An overlay that wrote back to the probe set would destroy it
silently — every row would look unchanged and the hash would move. So the property under
test is not "the corrected answers are right"; it is "the instrument was corrected without
being rewritten".
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

apply_corrections = pytest.importorskip("apply_gold_corrections")

PROBES = REPO / "results/analysis/synthesis-probes.json"
CORRECTIONS = REPO / "results/analysis/count-probe-gold-corrections.json"


def test_a_correction_only_removes_members_never_adds_one():
    """An overlay that could raise a gold answer could invent evidence, which is the one
    thing a corrected instrument must not be able to do."""
    probes = {
        "count_0": {
            "probe_id": "count_0",
            "kind": "count",
            "answer": 3,
            "evidence_memory_ids": ["a", "b", "c"],
        }
    }
    out = apply_corrections.corrected_gold(probes, {"count_0": {"b"}})

    assert out["count_0"]["original"] == 3
    assert out["count_0"]["corrected"] == 2
    assert set(out["count_0"]["members"]) <= {"a", "b", "c"}
    assert "b" not in out["count_0"]["members"]


def test_no_exclusions_leaves_the_gold_untouched():
    probes = {
        "count_0": {
            "probe_id": "count_0",
            "kind": "count",
            "answer": 2,
            "evidence_memory_ids": ["a", "b"],
        }
    }
    out = apply_corrections.corrected_gold(probes, {})

    assert out["count_0"]["corrected"] == out["count_0"]["original"] == 2
    assert out["count_0"]["dropped"] == 0


def test_only_count_probes_are_touched():
    """The adjudication is about set membership. A duration or a comparison has no members
    to exclude, and silently rewriting one would change a stratum nobody audited."""
    probes = {
        "duration_0": {
            "probe_id": "duration_0",
            "kind": "duration",
            "answer": 18,
            "evidence_memory_ids": ["a", "b"],
        }
    }
    with pytest.raises(ValueError, match="non-count"):
        apply_corrections.corrected_gold(probes, {"duration_0": {"a"}})


def test_parsed_is_read_as_text_because_rows_store_it_that_way():
    assert apply_corrections.as_int("8") == 8
    assert apply_corrections.as_int(None) is None
    assert apply_corrections.as_int("eight") is None


def test_the_committed_adjudication_is_well_formed():
    if not CORRECTIONS.is_file():
        pytest.skip("the adjudication is a local research artifact")
    adjudication = json.loads(CORRECTIONS.read_text(encoding="utf-8"))

    # It must not claim to be a human adjudication, because it is not one.
    assert "NOT a human adjudication" in adjudication["adjudicated_by"]
    assert adjudication["adjudicated_before_any_corrected_score_was_computed"] is True
    assert adjudication["provider_calls"] == 0

    members = adjudication["members"]
    assert len(members) == adjudication["counts"]["adjudicated"]
    for member in members:
        assert member["verdict"] in {"keep", "exclude"}
        # `disputed` came out of the review: two calls where the question's wording
        # supports both answers. They keep their original verdict rather than being
        # changed, because flipping one of them would make the headline stronger.
        assert member["confidence"] in {"high", "medium", "disputed"}
        # Every call carries the rule it was made under, so a reviewer can disagree with
        # the rule rather than with 38 separate opinions.
        assert member["rule"] in adjudication["rules"]
    # A KEEP rule may never be used to exclude, or the rule labels mean nothing.
    for member in members:
        keeps = member["rule"] in {"R4", "R5"}
        assert keeps == (member["verdict"] == "keep"), member

    # The review must not quietly have changed a verdict: it recorded what it found and
    # left the numbers alone, and that claim is checkable.
    review = adjudication.get("reviewed")
    if review:
        assert review["verdicts_changed"] == 0
        assert "NOT an independent human review" in review["reviewed_by"]
        # Anything marked disputed must carry the reasoning, not just the label.
        for member in members:
            if member["confidence"] == "disputed":
                assert member.get("review_note"), member["probe_id"]


def test_the_probe_set_still_hashes_to_what_the_split_recorded():
    """The whole point of the overlay. If a future change writes corrections back into the
    probe set, this fails and the held-out split is still provable."""
    split = REPO / "results/manifests/v4-probe-split.json"
    if not (PROBES.is_file() and split.is_file()):
        pytest.skip("probe artifacts are local research files")
    recorded = json.loads(split.read_text(encoding="utf-8"))["probe_set_sha256"]

    assert hashlib.sha256(PROBES.read_bytes()).hexdigest() == recorded, (
        "the probe set changed; corrections must be an overlay, never an edit"
    )


def small_overlay():
    probes = {
        "count_0": {
            "kind": "count",
            "question": "How many books read?",
            "namespace": "user",
            "answer": 2,
            "evidence_memory_ids": ["a", "b"],
        }
    }
    member = {
        "probe_id": "count_0",
        "memory_id": "a",
        "question": probes["count_0"]["question"],
        "memory": "Plans to read a book",
        "verdict": "exclude",
        "rule": "R1",
        "confidence": "high",
    }
    overlay = {
        "members": [member],
        "rules": {"R1": "plan", "R4": "completed"},
        "counts": {"adjudicated": 1, "keep": 0, "exclude": 1},
    }
    return probes, overlay, {"a": {"content": member["memory"]}, "b": {"content": "read a book"}}


@pytest.mark.parametrize(
    "change", ["hidden", "duplicate", "question", "memory", "nonmember", "rule", "counts"]
)
def test_overlay_cannot_silently_change_scope_or_evidence(change):
    probes, overlay, memories = small_overlay()
    m = overlay["members"][0]
    if change == "hidden":
        m["probe_id"] = "held_out"
    elif change == "duplicate":
        overlay["members"].append(dict(m))
    elif change in {"question", "memory"}:
        m[change] = "changed"
    elif change == "nonmember":
        m["memory_id"] = "unknown"
    elif change == "rule":
        m["rule"] = "R4"
    else:
        overlay["counts"]["exclude"] = 2
    with pytest.raises(ValueError):
        apply_corrections.validate_overlay(overlay, probes, memories)


def test_valid_overlay_only_removes_its_recorded_development_member():
    probes, overlay, memories = small_overlay()
    excluded = apply_corrections.validate_overlay(overlay, probes, memories)
    assert excluded == {"count_0": {"a"}}
    assert apply_corrections.corrected_gold(probes, excluded)["count_0"]["corrected"] == 1


@pytest.mark.parametrize("repeats", [[1, 2], [1, 1, 3], [1, 2, 3, 3]])
def test_partial_or_duplicate_repeats_cannot_be_reported_as_majority(repeats):
    probes, _, _ = small_overlay()
    rows = [
        {
            "kind": "count",
            "probe_id": "count_0",
            "question": probes["count_0"]["question"],
            "namespace": "user",
            "gold": 2,
            "repeat": rep,
        }
        for rep in repeats
    ]
    with pytest.raises(ValueError, match="incomplete or duplicate"):
        apply_corrections.validate_count_rows(rows, probes, 3)


def test_nonmember_exclusion_is_not_silently_counted_as_a_correction():
    probes, _, _ = small_overlay()
    with pytest.raises(ValueError, match="nonmember"):
        apply_corrections.corrected_gold(probes, {"count_0": {"not-a-member"}})


def test_missing_evidence_rows_cannot_change_original_answer_silently():
    probes, _, _ = small_overlay()
    probes["count_0"]["answer"] = 3
    with pytest.raises(ValueError, match="original gold"):
        apply_corrections.corrected_gold(probes, {})
