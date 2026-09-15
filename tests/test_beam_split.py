"""The BEAM split, against the ways a split stops meaning what it says without failing.

A split that moves with input order cannot be reproduced from its manifest; one that
separates two conversations sharing a seed can put related material on both sides; one
that misses a scale's target changes the final test after its size was chosen; and a
manifest that quotes a question publishes benchmark text into an MIT repository.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

beam_split = pytest.importorskip("beam_split")

MANIFESTS = REPO / "results/manifests"


def conversation(scale, conversation_id, seed_id):
    probing = {
        "abstention": [{"question": "q"}, {"question": "q"}],
        "event_ordering": [{"question": "q"}, {"question": "q"}],
    }
    return {
        "key": f"{scale}-{conversation_id}",
        "scale": scale,
        "seed_id": seed_id,
        "questions": beam_split.question_rows(scale, conversation_id, probing),
    }


def beam_shaped(pairs=20, singles=15):
    """The measured shape: every 100K seed recurs once in 500K, and 15 500K seeds do not."""
    conversations = []
    for i in range(pairs):
        conversations.append(conversation("100K", str(i + 1), seed_id=i))
        conversations.append(conversation("500K", str(i + 1), seed_id=i))
    for j in range(singles):
        conversations.append(conversation("500K", str(pairs + j + 1), seed_id=1000 + j))
    return conversations


def count(keys, scale):
    return sum(1 for key in keys if key.startswith(f"{scale}-"))


def test_each_scale_gets_exactly_its_share_of_the_final_half():
    dev, test = beam_split.split(beam_shaped(), 0.6, 7)
    assert (count(test, "100K"), count(test, "500K")) == (12, 21)
    assert (count(dev, "100K"), count(dev, "500K")) == (8, 14)


def test_conversations_sharing_a_seed_never_straddle_the_halves():
    conversations = beam_shaped()
    dev, test = beam_split.split(conversations, 0.6, 7)
    side = {key: "dev" for key in dev} | {key: "test" for key in test}
    sides_by_seed = {}
    for c in conversations:
        sides_by_seed.setdefault(c["seed_id"], set()).add(side[c["key"]])
    assert all(len(sides) == 1 for sides in sides_by_seed.values())


def test_every_conversation_lands_in_exactly_one_half():
    conversations = beam_shaped()
    dev, test = beam_split.split(conversations, 0.6, 7)
    assert not set(dev) & set(test)
    assert set(dev) | set(test) == {c["key"] for c in conversations}


def test_the_seed_fixes_the_split_and_input_order_does_not_move_it():
    conversations = beam_shaped()
    first = beam_split.split(conversations, 0.6, 7)
    assert beam_split.split(conversations[::-1], 0.6, 7) == first
    assert beam_split.split(conversations, 0.6, 8) != first


def test_a_target_that_would_break_a_seed_group_is_refused():
    together = [conversation("500K", "1", seed_id=1), conversation("500K", "2", seed_id=1)]
    with pytest.raises(ValueError, match="no assignment"):
        beam_split.split(together, 0.5, 7)


def test_duplicate_conversation_keys_are_refused():
    one = conversation("100K", "1", seed_id=1)
    with pytest.raises(ValueError, match="unique"):
        beam_split.split([one, dict(one)], 0.5, 7)


def test_question_ids_keep_the_scale_because_conversation_ids_restart_per_file():
    probing = {"abstention": [{"question": "x"}, {"question": "y"}]}
    rows = beam_split.question_rows("100K", "1", probing) + beam_split.question_rows(
        "500K", "1", probing
    )
    ids = [row["question_id"] for row in rows]
    assert ids[0] == "beam-100K-1-abstention-0"
    assert len(set(ids)) == 4
    assert {beam_split.conversation_of(qid) for qid in ids} == {"100K-1", "500K-1"}


def test_probing_questions_decode_as_json_or_as_a_python_literal():
    assert beam_split.parse_probing('{"abstention": []}') == {"abstention": []}
    assert beam_split.parse_probing("{'abstention': []}") == {"abstention": []}
    with pytest.raises(ValueError, match="mapping"):
        beam_split.parse_probing("[1, 2]")


def test_committed_manifests_hold_ids_and_counts_but_no_benchmark_text():
    split_path, dev_path = MANIFESTS / "beam-split.json", MANIFESTS / "beam-dev.json"
    if not split_path.is_file():
        pytest.skip("no BEAM split has been cut in this checkout")
    test_path = MANIFESTS / "beam-test.json"
    for path in (split_path, dev_path, test_path):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for field in ('"question"', '"rubric"', '"answer"', '"ideal_response"', '"narratives"'):
            assert field not in text, f"{path.name} quotes BEAM through {field}"
    record = json.loads(split_path.read_text(encoding="utf-8"))
    dev_keys = set(record["dev"]["conversations"])
    test_keys = set(record["test"]["conversations"])
    assert not dev_keys & test_keys
    dev_ids = json.loads(dev_path.read_text(encoding="utf-8"))["question_ids"]
    assert len(dev_ids) == record["dev"]["questions"]
    assert {beam_split.conversation_of(qid) for qid in dev_ids} == dev_keys
    if test_path.is_file():
        test_ids = json.loads(test_path.read_text(encoding="utf-8"))["question_ids"]
        assert len(test_ids) == record["test"]["questions"]
        assert {beam_split.conversation_of(qid) for qid in test_ids} == test_keys
