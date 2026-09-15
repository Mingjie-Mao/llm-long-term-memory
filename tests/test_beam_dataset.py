"""BEAM loading, against the ways it could measure a different system without failing.

Twenty questions keyed as twenty users would build one conversation's store twenty times
and answer each question from a store built for another; sessions left at BEAM's length
would run the extractor far outside the size it was measured at; and evidence that failed
to resolve through one of BEAM's three source-id shapes would read as a retrieval miss.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_long_term_memory.evaluation.datasets import beam
from llm_long_term_memory.evaluation.datasets.longmemeval import (
    HaystackSession,
    HaystackTurn,
    Instance,
)
from llm_long_term_memory.evaluation.manifest import Manifest
from llm_long_term_memory.ingest.extract import _parse_date
from llm_long_term_memory.ingest.pipeline import namespaced_sessions

REPO = Path(__file__).resolve().parent.parent


def message(mid, role, chars, anchor=None):
    return {"id": mid, "role": role, "content": "x" * chars, "time_anchor": anchor}


def conversation(sessions=2, exchanges=6, reply_chars=3_000):
    chat, mid = [], 0
    for s in range(sessions):
        session = []
        for e in range(exchanges):
            session.append(message(mid, "user", 200, f"March-{10 + s}-2024" if e == 0 else None))
            session.append(message(mid + 1, "assistant", reply_chars))
            mid += 2
        chat.append(session)
    last = mid - 1
    probing = {
        "abstention": [{"question": "q0", "ideal_response": "declines", "rubric": ["declines"]}],
        "event_ordering": [
            {
                "question": "q1",
                "answer": "a, then b",
                "rubric": ["a", "b"],
                "source_chat_ids": [[0, 1], [last]],
            }
        ],
        "temporal_reasoning": [
            {
                "question": "q2",
                "answer": "one day",
                "rubric": ["one day"],
                "source_chat_ids": {"first_event": [0], "second_event": last},
            }
        ],
    }
    return {
        "scale": "100K",
        "conversation_id": "7",
        "seed_id": 1,
        "chat": chat,
        "probing_questions": probing,
    }


def by_ability():
    return {i.question_type: i for i in beam.conversation_instances(conversation())}


def test_chunks_are_whole_exchanges_and_lose_nothing():
    messages = conversation()["chat"][0]
    chunks = beam.chunk_session(messages, target_chars=7_000)
    assert [m for chunk in chunks for m in chunk] == messages
    assert all(chunk[0]["role"] == "user" for chunk in chunks)
    # 3,200 characters an exchange: two (6,400) sit nearer 7,000 than three (9,600) would.
    assert [sum(len(m["content"]) for m in chunk) for chunk in chunks] == [6_400] * 3


def test_a_chunk_passes_the_target_when_that_lands_nearer_it():
    messages = []
    for n in range(3):
        messages += [message(2 * n, "user", 1_000), message(2 * n + 1, "assistant", 5_000)]
    sizes = [
        sum(len(m["content"]) for m in chunk)
        for chunk in beam.chunk_session(messages, target_chars=10_000)
    ]
    assert sizes == [12_000, 6_000]


def test_an_exchange_longer_than_the_target_stands_alone():
    messages = [
        message(0, "user", 10),
        message(1, "assistant", 30_000),
        message(2, "user", 10),
        message(3, "assistant", 10),
    ]
    assert [len(chunk) for chunk in beam.chunk_session(messages)] == [2, 2]


def test_a_conversations_questions_share_its_namespace_and_its_sessions():
    instances = beam.conversation_instances(conversation())
    assert {i.store_namespace for i in instances} == {"beam-100K-7"}
    assert [i.question_id for i in instances] == [
        "beam-100K-7-abstention-0",
        "beam-100K-7-event_ordering-0",
        "beam-100K-7-temporal_reasoning-0",
    ]
    assert all(i.sessions is instances[0].sessions for i in instances)


def test_ingestion_extracts_a_shared_conversation_once():
    instances = beam.conversation_instances(conversation())
    assert len(namespaced_sessions(instances)) == len(instances[0].sessions)


def test_questions_without_a_namespace_still_get_a_store_each():
    session = HaystackSession("s1", "2023/05/20 (Sat) 02:21", [HaystackTurn("user", "hi")])
    a = Instance("qa", "t", "?", "a", "", [session], [])
    b = Instance("qb", "t", "?", "a", "", [session], [])
    assert [namespace for namespace, _ in namespaced_sessions([a, b])] == ["qa", "qb"]


def test_chunk_dates_parse_the_way_ingestion_reads_them_and_keep_their_order():
    sessions = beam.conversation_instances(conversation())[0].sessions
    parsed = [_parse_date(session.date) for session in sessions]
    assert all(parsed)
    assert parsed == sorted(parsed)
    assert [session.session_id for session in sessions[:3]] == ["s00c000", "s00c001", "s01c000"]
    assert [(p.day, p.minute) for p in parsed[:3]] == [(10, 0), (10, 1), (11, 0)]


def test_evidence_resolves_through_every_source_id_shape():
    abilities = by_ability()
    sessions = abilities["event_ordering"].sessions
    ends = sorted({sessions[0].session_id, sessions[-1].session_id})
    assert abilities["event_ordering"].answer_session_ids == ends
    assert abilities["temporal_reasoning"].answer_session_ids == ends
    assert abilities["abstention"].answer_session_ids == []
    assert abilities["abstention"].is_abstention


def test_reference_and_rubric_come_from_each_abilitys_own_fields():
    abilities = by_ability()
    assert abilities["abstention"].answer == "declines"
    assert abilities["event_ordering"].answer == "a, then b"
    assert abilities["event_ordering"].rubric == ("a", "b")


def test_a_session_whose_first_message_has_no_date_is_refused():
    broken = conversation()
    broken["chat"][1][0]["time_anchor"] = None
    with pytest.raises(ValueError, match="no dated first message"):
        beam.conversation_instances(broken)


def test_manifests_name_their_half():
    def named(name):
        return Manifest(name=name, variant="beam", seed=0, question_ids=())

    assert beam.half_of(named("beam-dev")) == "dev"
    assert beam.half_of(named("beam-test")) == "test"
    assert beam.half_of(named("test100")) is None
    with pytest.raises(ValueError, match="names no BEAM half"):
        beam.half_of(named("beam-split"))


def test_loading_reads_its_own_half_and_refuses_another(tmp_path):
    (tmp_path / "beam").mkdir()
    path = tmp_path / "beam" / "beam-dev.json"
    payload = {"half": "dev", "conversations": [conversation()]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert len(beam.load("dev", tmp_path)) == 3
    path.rename(tmp_path / "beam" / "beam-test.json")
    with pytest.raises(ValueError, match="holds the 'dev' half"):
        beam.load("test", tmp_path)
    with pytest.raises(FileNotFoundError, match="beam_export"):
        beam.load("dev", tmp_path)


def test_the_exported_development_half_is_the_registered_one():
    if not (REPO / "data/beam/beam-dev.json").is_file():
        pytest.skip("the BEAM development half is not exported in this checkout")
    manifest = json.loads((REPO / "results/manifests/beam-dev.json").read_text(encoding="utf-8"))
    loaded = {instance.question_id for instance in beam.load("dev", REPO / "data")}
    assert loaded == set(manifest["question_ids"])
