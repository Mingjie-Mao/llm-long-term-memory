"""The stage decomposition, against the two ways it could mislead the reader.

It must not call a canonical label "missing evidence" just because the label was composed
rather than quoted, and it must not let an assistant-authored mention stand in for the
user's own words. The rest of the file pins the boundary between those buckets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

stages = pytest.importorskip("count_stage_decomposition")


def item(entities, sources, memories, relation="owns"):
    return {
        "id": "q1",
        "kind": "entity",
        "relation": relation,
        "proposed_entities": entities,
        "sources": sources,
        "memories": memories,
        "flags": [],
    }


def user(text):
    return {"id": "s1", "role": "user", "text": text}


def assistant(text):
    return {"id": "s2", "role": "assistant", "text": text}


def memory(predicate, content, obj=None):
    return {"id": "m1", "predicate": predicate, "content": content, "object": obj}


APPROVED = {"vinyl_collection"}


def test_a_label_quoted_whole_by_the_user_under_an_approved_predicate_reaches_the_answer():
    stage, cover = stages.stage_of(
        "Space Oddity",
        item(
            ["Space Oddity"],
            [user("I own a rare single of Space Oddity.")],
            [memory("vinyl_collection", "The user owns Space Oddity.", "Space Oddity")],
        ),
        APPROVED,
    )
    assert stage == "quotable_under_an_approved_predicate"
    assert cover == 1.0


def test_a_composed_label_is_partly_supported_not_called_missing():
    # The generator's label joins artist and album; the turn says only the album.
    stage, cover = stages.stage_of(
        "Fleetwood Mac - Rumours",
        item(
            ["Fleetwood Mac - Rumours"],
            [user("I am displaying my limited pressing of Rumours on the wall.")],
            [memory("vinyl_collection", "The user owns Rumours.", "Rumours")],
        ),
        APPROVED,
    )
    assert stage == "label_partly_supported"
    assert 0 < cover < 1


def test_a_member_only_the_assistant_ever_named_is_separated_from_missing_evidence():
    stage, _ = stages.stage_of(
        "Akai MPC Live 2",
        item(
            ["Akai MPC Live 2"],
            [user("Can you help me plan the episode?"), assistant("Show the Akai MPC Live 2.")],
            [memory("vinyl_collection", "The user owns an Akai MPC Live 2.")],
        ),
        APPROVED,
    )
    assert stage == "only_in_assistant_text"


def test_a_member_nobody_named_is_unsupported():
    stage, cover = stages.stage_of(
        "grandmother",
        item(["grandmother"], [user("I had an argument with my sister.")], []),
        APPROVED,
    )
    assert stage == "unsupported_in_user_text"
    assert cover == 0.0


def test_a_quoted_member_carried_only_by_an_unapproved_predicate_is_a_unit_error():
    stage, _ = stages.stage_of(
        "tofu",
        item(
            ["tofu"],
            [user("I bought tofu at the market.")],
            [memory("groceries", "The user bought tofu.", "tofu")],
            relation="ate_at",
        ),
        {"restaurants_visited"},
    )
    assert stage == "carried_by_unapproved_predicate"


def test_the_report_states_what_it_cannot_measure_and_counts_every_proposed_member():
    packet = {
        "packet_sha256": "a" * 64,
        "policy": {"relations": {"owns": {"predicates": sorted(APPROVED)}}},
        "items": [
            item(
                ["Space Oddity", "grandmother"],
                [user("I own a rare single of Space Oddity.")],
                [memory("vinyl_collection", "The user owns Space Oddity.", "Space Oddity")],
            ),
            {"kind": "legacy", "id": "old1"},
        ],
    }
    report = stages.decompose(packet)
    assert report["provider_calls"] == 0
    assert "extraction" in report["cannot_measure"]
    assert report["totals"] == {
        "questions": 1,
        "proposed_members": 2,
        "questions_with_a_blocking_member": 1,
        "questions_carrying_a_packet_flag": 0,
    }
    assert sum(report["members_by_stage"].values()) == 2
