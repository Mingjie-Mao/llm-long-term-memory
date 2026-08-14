"""Evidence hydration keeps structured retrieval grounded in raw source text."""

from datetime import datetime

from chronomem.retrieve import EvidenceHydrator
from chronomem.store import Memory, Session, SQLiteMemoryStore, Turn


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
