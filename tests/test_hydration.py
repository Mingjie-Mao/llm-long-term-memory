"""Evidence hydration keeps structured retrieval grounded in raw source text."""

from datetime import datetime

from llm_long_term_memory.retrieve import EvidenceHydrator
from llm_long_term_memory.store import Memory, Session, SQLiteMemoryStore, Turn


def test_hydration_expands_the_anchored_sentence_by_local_context(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "m.db")
    store.initialize()
    text = "I started collecting vintage cameras three months ago. I now own 17 cameras."
    store.add_session(
        Session(
            id="s1",
            user_id="u",
            started_at=datetime(2026, 1, 1),
            turns=[Turn("s1:0", "s1", 0, "user", text, datetime(2026, 1, 1))],
        )
    )
    start = text.index("I now own")
    store.add_memories(
        [
            Memory(
                id="m1",
                user_id="u",
                type="semantic",
                content="The user owns 17 vintage cameras.",
                token_count=8,
                source_session_id="s1",
                source_turn_index=0,
                source_char_start=start,
                source_char_end=len(text),
            )
        ]
    )

    hydrated = EvidenceHydrator(store, neighbouring_sentences=1).hydrate([store.get("m1")])

    assert hydrated.tokens > 0
    assert hydrated.evidence[0].text == text
    store.close()


def test_hydration_respects_its_own_token_budget(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "m.db")
    store.initialize()
    text = "The user started collecting cameras three months ago."
    store.add_session(
        Session(
            id="s1",
            user_id="u",
            started_at=datetime(2026, 1, 1),
            turns=[Turn("s1:0", "s1", 0, "user", text, datetime(2026, 1, 1))],
        )
    )
    store.add_memories(
        [
            Memory(
                id="m1",
                user_id="u",
                type="semantic",
                content="The user started collecting cameras.",
                token_count=6,
                source_session_id="s1",
                source_turn_index=0,
                source_char_start=0,
                source_char_end=len(text),
            )
        ]
    )

    hydrated = EvidenceHydrator(store).hydrate([store.get("m1")], max_tokens=1)

    assert hydrated.evidence == []
    assert hydrated.skipped_for_budget == 1
    store.close()


def test_session_fair_hydration_gives_each_session_a_first_chance(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "m.db")
    store.initialize()
    texts = {
        "s1": ["Alpha evidence has value 111.", "Alpha evidence has value 222."],
        "s2": ["Bravo evidence has value 333."],
    }
    for session_id, contents in texts.items():
        store.add_session(
            Session(
                id=session_id,
                user_id="u",
                started_at=datetime(2026, 1, 1),
                turns=[
                    Turn(
                        f"{session_id}:{index}",
                        session_id,
                        index,
                        "user",
                        content,
                        datetime(2026, 1, 1),
                    )
                    for index, content in enumerate(contents)
                ],
            )
        )
    specs = [
        ("m1", "s1", 0, texts["s1"][0]),
        ("m2", "s1", 1, texts["s1"][1]),
        ("m3", "s2", 0, texts["s2"][0]),
    ]
    store.add_memories(
        [
            Memory(
                id=memory_id,
                user_id="u",
                type="semantic",
                content=content,
                token_count=6,
                source_session_id=session_id,
                source_turn_index=turn_index,
                source_char_start=0,
                source_char_end=len(content),
            )
            for memory_id, session_id, turn_index, content in specs
        ]
    )
    memories = store.get_many(["m1", "m2", "m3"])
    unbounded = EvidenceHydrator(store, neighbouring_sentences=0).hydrate(memories)
    budget = unbounded.evidence[0].token_count + unbounded.evidence[2].token_count

    ranked = EvidenceHydrator(store, neighbouring_sentences=0).hydrate(memories, max_tokens=budget)
    fair = EvidenceHydrator(store, neighbouring_sentences=0, allocation="session_fair").hydrate(
        memories, max_tokens=budget
    )

    assert [item.memory_id for item in ranked.evidence] == ["m1", "m2"]
    assert [item.memory_id for item in fair.evidence] == ["m1", "m3"]
    assert fair.eligible_sessions == 2
    assert fair.hydrated_sessions == 2
    assert fair.tokens <= budget
    store.close()


def test_session_fair_hydration_deduplicates_the_same_source_sentence(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "m.db")
    store.initialize()
    text = "The launch date is June 3 and the confirmed count is 17."
    store.add_session(
        Session(
            id="s1",
            user_id="u",
            started_at=datetime(2026, 1, 1),
            turns=[Turn("s1:0", "s1", 0, "user", text, datetime(2026, 1, 1))],
        )
    )
    store.add_memories(
        [
            Memory(
                id="m-date",
                user_id="u",
                type="semantic",
                content="The launch date is June 3.",
                token_count=6,
                source_session_id="s1",
                source_turn_index=0,
                source_char_start=text.index("June 3"),
                source_char_end=text.index("June 3") + len("June 3"),
            ),
            Memory(
                id="m-count",
                user_id="u",
                type="semantic",
                content="The confirmed count is 17.",
                token_count=6,
                source_session_id="s1",
                source_turn_index=0,
                source_char_start=text.index("17"),
                source_char_end=text.index("17") + len("17"),
            ),
        ]
    )

    result = EvidenceHydrator(store, neighbouring_sentences=0, allocation="session_fair").hydrate(
        store.get_many(["m-date", "m-count"])
    )

    assert len(result.evidence) == 1
    assert result.evidence[0].text == text
    assert result.redundant_anchors == 1
    store.close()


def test_hydrator_rejects_unknown_allocation(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "m.db")
    store.initialize()

    try:
        EvidenceHydrator(store, allocation="unknown")  # type: ignore[arg-type]
    except ValueError as exc:
        assert "unknown hydration allocation" in str(exc)
    else:
        raise AssertionError("unknown allocation must be rejected")
    finally:
        store.close()
