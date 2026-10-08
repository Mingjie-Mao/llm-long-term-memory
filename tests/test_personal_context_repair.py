from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from llm_long_term_memory.config import ExperimentConfig
from llm_long_term_memory.conversation import ConversationSession, ConversationTurn
from llm_long_term_memory.ingest import fingerprint
from llm_long_term_memory.ingest.dedup import Deduplicator
from llm_long_term_memory.ingest.personal_context import PersonalContextRepair, configured_repair
from llm_long_term_memory.store import Memory, Session, SQLiteMemoryStore, Turn, scoped_session_id


def session(*lines):
    return ConversationSession(
        "s", "2026-10-03", [ConversationTurn(role, text) for role, text in lines]
    )


def test_preserve_user_preferences_experience_plans_negation_without_fabricating_state():
    raw = session(
        (
            "user",
            "I prefer winding down by 9:30pm. I attended a class last month. "
            "I'm considering a new guitar. I do not like jazz.",
        ),
        ("assistant", "You love jazz and should attend my class."),
    )
    result = PersonalContextRepair().repair(raw, [], "alice")
    assert not result.called and len(result.memories) == 4
    assert all(
        m.predicate == "source_quote" and m.scope == "shared_context" for m in result.memories
    )
    assert all(m.event_time is None and not m.replaces_previous for m in result.memories)
    for m in result.memories:
        assert (
            raw.turns[m.source_turn_index].content[m.source_char_start : m.source_char_end]
            == m.content
        )
    assert "I do not like jazz." in [m.content for m in result.memories]


def test_hypotheticals_and_questions_without_personal_assertion_are_not_claims():
    raw = session(("user", "If I have five birds, how many wings? What podcast do you recommend?"))
    assert PersonalContextRepair().repair(raw, [], "alice").memories == []


def test_numeric_echo_reanchor_requires_one_exact_content_supported_user_source():
    raw = session(
        ("user", "My commute is about 40 minutes each way."),
        ("assistant", "Your commute is about 40 minutes each way."),
    )
    memory = Memory(
        "m",
        "alice",
        "semantic",
        "The user has a 40 minutes commute each way.",
        10,
        source_session_id="s",
        source_turn_index=1,
    )
    PersonalContextRepair().repair(raw, [memory], "alice")
    assert memory.source_turn_index == 0 and memory.type == "semantic"
    ambiguous = replace(memory, source_turn_index=2)
    repeated = session(
        ("user", raw.turns[0].content),
        ("user", raw.turns[0].content),
        ("assistant", raw.turns[1].content),
    )
    PersonalContextRepair().repair(repeated, [ambiguous], "alice")
    assert ambiguous.source_turn_index == 2
    with pytest.raises(ValueError, match="foreign"):
        PersonalContextRepair().repair(raw, [replace(memory, user_id="bob")], "alice")


def test_quote_identity_is_idempotent_and_tenant_bound():
    raw = session(("user", "I enjoy history podcasts."))
    repair = PersonalContextRepair()
    a = repair.repair(raw, [], "alice").memories[0]
    b = repair.repair(raw, [], "bob").memories[0]
    assert a.id != b.id
    assert repair.repair(raw, [a], "alice").memories == []


def test_decimal_price_and_time_are_not_split_into_a_false_integer_quote():
    raw = session(("user", "I paid $9.50 for a book. I prefer stopping at 9:30pm."))
    result = PersonalContextRepair().repair(raw, [], "alice")
    assert [m.content for m in result.memories] == [
        "I paid $9.50 for a book.",
        "I prefer stopping at 9:30pm.",
    ]


@pytest.mark.parametrize("specificity", [False, True])
def test_runtime_and_config_repair_fingerprint_match_and_off_preserved(specificity):
    cfg = ExperimentConfig.from_yaml("configs/fallback.yaml")
    old = fingerprint.from_config(cfg, sessions_per_request=8).as_dict()
    cfg.ingest.personal_context_repair = True
    cfg.ingest.specificity_repair = specificity
    repair = configured_repair(cfg)
    assert (
        fingerprint._repair(repair)
        == fingerprint.from_config(cfg, sessions_per_request=8).specificity_repair
    )
    assert "specificity_repair" not in old


def test_real_dedup_keeps_quote_beside_summary_and_never_merges_other_tenants(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "s.db")
    store.initialize()
    raw = session(("user", "I enjoy history podcasts and my commute is 40 minutes each way."))
    a = PersonalContextRepair().repair(raw, [], "alice").memories[0]
    b = PersonalContextRepair().repair(raw, [], "bob").memories[0]
    from datetime import datetime

    for memory in (a, b):
        memory.source_session_id = scoped_session_id(memory.user_id, "s")
        store.add_session(
            Session(
                memory.source_session_id,
                memory.user_id,
                datetime(2026, 10, 3),
                "test",
                [
                    Turn(
                        memory.id + ":0",
                        memory.source_session_id,
                        0,
                        "user",
                        raw.turns[0].content,
                        datetime(2026, 10, 3),
                    )
                ],
            )
        )
    dedup = Deduplicator(
        SimpleNamespace(generate=lambda **k: pytest.fail("no quote adjudication")),
        "same",
        SimpleNamespace(),
        store=store,
    )
    outcome = dedup.process([a, b, a], np.ones((3, 4)))
    assert outcome.kept == [a, b] and outcome.duplicates == 1
    store.add_memories(outcome.kept)
    assert not dedup.process([a], np.ones((1, 4))).kept
    assert [m.id for m in store.iter_all("bob")] == [b.id]
    store.close()
