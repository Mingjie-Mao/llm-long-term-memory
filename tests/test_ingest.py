"""Extraction and dedup.

The dedup tests carry most of the weight here: they encode *why* the stage is two
stages rather than a similarity threshold, which is the design question this part of
the system exists to answer.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

from llm_long_term_memory.evaluation.datasets.longmemeval import HaystackSession, HaystackTurn
from llm_long_term_memory.ingest import Deduplicator, Extractor, memory_id, normalize_predicate
from llm_long_term_memory.ingest.extract import _parse_date, render_batch
from llm_long_term_memory.store import Memory, NumpyFlatIndex, SQLiteMemoryStore

# --------------------------------------------------------------------- helpers


class ScriptedClient:
    """Returns queued JSON payloads and records the prompts it was given."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.prompts: list[str] = []

    def generate(self, *, role, model, prompt, **kw):
        self.prompts.append(prompt)
        text = self.payloads.pop(0)

        class C:
            pass

        c = C()
        c.text = text if isinstance(text, str) else json.dumps(text)
        c.input_tokens = 100
        c.output_tokens = 20
        c.thinking_tokens = 0
        c.api_latency_ms = 5.0
        return c


class StubEncoder:
    """Maps text to a vector by keyword, so similarity is controllable in tests."""

    def __init__(self, mapping: dict[str, list[float]], dim: int = 3):
        self.mapping = mapping
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts, show_progress=False):
        return np.array([self.mapping[t] for t in texts], dtype=np.float32)

    def encode_one(self, text):
        return self.encode([text])[0]


def session(sid: str, date: str, *lines: str) -> HaystackSession:
    turns = [
        HaystackTurn(role="user" if i % 2 == 0 else "assistant", content=line)
        for i, line in enumerate(lines)
    ]
    return HaystackSession(session_id=sid, date=date, turns=turns)


def mem(mid: str, content: str, subject="user", predicate="prefers", obj="x") -> Memory:
    return Memory(
        id=mid,
        user_id="u1",
        type="preference",
        content=content,
        token_count=5,
        subject=subject,
        predicate=predicate,
        object=obj,
        ingested_at=datetime(2026, 1, 1),
    )


# ------------------------------------------------------------------ extraction


def test_predicate_normalization_makes_repeat_mentions_collide():
    """P4 matches supersede on (subject, predicate). Unnormalized relations never
    collide and the temporal layer becomes a no-op."""
    assert normalize_predicate("Prefers Framework") == "prefers_framework"
    assert normalize_predicate("prefers-framework") == "prefers_framework"
    assert normalize_predicate("  PREFERS   framework  ") == "prefers_framework"
    assert normalize_predicate("lives_in") == "lives_in"


def test_extraction_attaches_the_right_session_date_and_id():
    sessions = [
        session("s0", "2026/01/05 (Mon) 09:00", "I moved to Sydney"),
        session("s1", "2026/03/12 (Thu) 14:00", "I started using PyTorch"),
    ]
    client = ScriptedClient(
        [
            {
                "memories": [
                    {
                        "session_index": 1,
                        "type": "semantic",
                        "content": "The user started using PyTorch",
                        "subject": "user",
                        "predicate": "uses tool",
                        "object": "PyTorch",
                        "entities": ["PyTorch"],
                        "importance": 0.8,
                    }
                ]
            }
        ]
    )
    outcome = Extractor(client, "m", user_id="u1").extract(sessions)

    (m,) = outcome.memories
    assert m.source_session_id == "s1", "must map to the session it named, not the first"
    assert m.event_time == datetime(2026, 3, 12, 14, 0)
    assert m.valid_from == m.event_time
    assert m.valid_to is None, "a new fact is open-ended until something supersedes it"
    assert m.predicate == "uses_tool", "normalized on the way in"
    assert m.entities == ["PyTorch"]


def test_out_of_range_session_index_is_counted_not_silently_dropped():
    """A nonzero rate here is the signal that the batch is too big for the model to
    track, and the cue to lower sessions_per_request."""
    sessions = [session("s0", "2026/01/05", "hi")]
    client = ScriptedClient(
        [
            {
                "memories": [
                    {
                        "session_index": 7,
                        "type": "semantic",
                        "content": "orphan",
                        "subject": "user",
                        "predicate": "x",
                        "object": "y",
                        "entities": [],
                        "importance": 0.5,
                    }
                ]
            }
        ]
    )
    outcome = Extractor(client, "m").extract(sessions)
    assert outcome.memories == []
    assert outcome.dropped_bad_index == 1


def test_batch_rendering_labels_every_session_with_its_index():
    text = render_batch([session("a", "2026/01/01", "one"), session("b", "2026/02/02", "two")])
    assert "Session index 0" in text
    assert "Session index 1" in text
    assert "date: 2026/02/02" in text


def test_memory_ids_are_stable_so_reingest_replaces_rather_than_duplicates():
    a = memory_id("u1", "s1", "The user lives in Sydney")
    b = memory_id("u1", "s1", "The user lives in Sydney")
    c = memory_id("u1", "s2", "The user lives in Sydney")
    assert a == b
    assert a != c


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2026/01/05 (Mon) 09:00", datetime(2026, 1, 5, 9, 0)),
        ("2026/01/05", datetime(2026, 1, 5)),
        ("2026-01-05 09:00", datetime(2026, 1, 5, 9, 0)),
        ("garbage", None),
        ("", None),
    ],
)
def test_date_parsing_covers_the_corpus_formats(raw, expected):
    assert _parse_date(raw) == expected


def test_extraction_of_an_empty_batch_costs_no_request():
    client = ScriptedClient([])
    outcome = Extractor(client, "m").extract([])
    assert outcome.memories == []
    assert client.prompts == []


# ----------------------------------------------------------------------- dedup


def test_negation_is_an_update_not_a_duplicate(tmp_path):
    """The case that rules out a pure similarity threshold: these two embed at ~0.95
    but mean opposite things. Dropping the second as a duplicate would leave the
    store asserting a preference the user has since reversed."""
    likes = "The user likes Python"
    dislikes = "The user does not like Python"
    encoder = StubEncoder({likes: [1.0, 0.0, 0.0], dislikes: [0.98, 0.2, 0.0]})

    client = ScriptedClient([{"verdict": "UPDATE", "reason": "negated preference"}])
    dedup = Deduplicator(client, "m", encoder, threshold=0.92)

    outcome = dedup.process([mem("m1", likes), mem("m2", dislikes)])

    assert outcome.duplicates == 0
    assert [m.id for m in outcome.kept] == ["m1", "m2"]
    assert len(outcome.updates) == 1
    new, superseded = outcome.updates[0]
    assert (new.id, superseded.id) == ("m2", "m1")
    assert outcome.adjudications == 1


def test_genuine_restatement_is_dropped(tmp_path):
    a = "The user lives in Canberra"
    b = "The user's home is in Canberra"
    encoder = StubEncoder({a: [1.0, 0.0, 0.0], b: [0.99, 0.1, 0.0]})
    client = ScriptedClient([{"verdict": "DUPLICATE", "reason": "same fact"}])

    outcome = Deduplicator(client, "m", encoder, threshold=0.92).process(
        [mem("m1", a), mem("m2", b)]
    )
    assert outcome.duplicates == 1
    assert [m.id for m in outcome.kept] == ["m1"]


def test_dissimilar_candidates_never_reach_the_llm():
    """The embedding stage exists to keep the expensive call rare."""
    a = "The user lives in Canberra"
    b = "The user's cat is named Mochi"
    encoder = StubEncoder({a: [1.0, 0.0, 0.0], b: [0.0, 1.0, 0.0]})
    client = ScriptedClient([])  # any call would IndexError

    outcome = Deduplicator(client, "m", encoder, threshold=0.92).process(
        [mem("m1", a), mem("m2", b)]
    )
    assert len(outcome.kept) == 2
    assert outcome.adjudications == 0


def test_a_near_duplicate_in_another_namespace_is_not_adjudicated(tmp_path):
    """Retrieval filters by namespace after searching the shared index; dedup did not.

    The consequence was not a wasted call but a deletion: a DUPLICATE verdict against a
    neighbour belonging to someone else drops a fact from a history that never contained
    it. Sampling a built store found 42 cross-namespace neighbours above threshold
    against 3 same-namespace ones, so this was the ordinary case rather than the corner.

    `ScriptedClient([])` is the assertion — any adjudication would raise.
    """
    store = SQLiteMemoryStore(tmp_path / "s.db")
    store.initialize()
    index = NumpyFlatIndex(tmp_path / "index", dim=3)
    theirs = "The user lives in Canberra"
    stored = replace(mem("theirs", theirs), user_id="u2")
    store.add_memories([stored])
    index.add([stored.id], np.array([[1.0, 0.0, 0.0]], dtype=np.float32))

    mine = "The user's home is in Canberra"
    encoder = StubEncoder({mine: [0.99, 0.1, 0.0]})

    outcome = Deduplicator(
        ScriptedClient([]), "m", encoder, store=store, index=index, threshold=0.92
    ).process([mem("mine", mine, predicate="home_area")])

    assert outcome.adjudications == 0
    assert [m.id for m in outcome.kept] == ["mine"]
    store.close()


def test_a_near_duplicate_in_the_same_namespace_still_is(tmp_path):
    """The scoping must not turn the stage off: the same pair inside one namespace is
    still compared, so the fix removes the foreign comparisons and nothing else."""
    store = SQLiteMemoryStore(tmp_path / "s.db")
    store.initialize()
    index = NumpyFlatIndex(tmp_path / "index", dim=3)
    existing = mem("existing", "The user lives in Canberra")
    store.add_memories([existing])
    index.add([existing.id], np.array([[1.0, 0.0, 0.0]], dtype=np.float32))

    mine = "The user's home is in Canberra"
    encoder = StubEncoder({mine: [0.99, 0.1, 0.0]})
    client = ScriptedClient([{"verdict": "DUPLICATE", "reason": "same fact"}])

    outcome = Deduplicator(client, "m", encoder, store=store, index=index, threshold=0.92).process(
        [mem("mine", mine, predicate="home_area")]
    )

    assert outcome.adjudications == 1
    assert outcome.duplicates == 1
    assert outcome.kept == []
    store.close()


def test_shared_predicate_is_checked_even_when_wording_diverges(tmp_path):
    """'lives in Canberra' and 'relocated to Sydney' are far apart in embedding space,
    but they are exactly the collision UPDATE exists for. The (subject, predicate)
    lookup catches what similarity misses."""
    store = SQLiteMemoryStore(tmp_path / "s.db")
    store.initialize()
    store.add_memories(
        [mem("old", "The user lives in Canberra", predicate="lives_in", obj="Canberra")]
    )

    new_text = "The user relocated to Sydney"
    encoder = StubEncoder({new_text: [0.0, 1.0, 0.0]})
    client = ScriptedClient([{"verdict": "UPDATE", "reason": "location changed"}])

    outcome = Deduplicator(client, "m", encoder, store=store, threshold=0.92).process(
        [mem("new", new_text, predicate="lives_in", obj="Sydney")]
    )

    assert outcome.adjudications == 1, "similarity alone would have skipped this"
    assert len(outcome.updates) == 1
    assert outcome.updates[0][1].id == "old"
    store.close()


def test_multi_valued_predicate_collisions_do_not_trigger_adjudication(tmp_path):
    """Owning a fern does not stop you owning a Fitbit.

    Measured cost of getting this wrong: 72 LLM calls per 60 sessions producing
    zero verdicts — twelve times the extraction budget — because every
    `user/owns/...` fact collided with every other one on the same key.
    """
    store = SQLiteMemoryStore(tmp_path / "s.db")
    store.initialize()
    store.add_memories([mem("old", "The user owns a peace lily", predicate="owns", obj="lily")])

    new_text = "The user owns a Fitbit Charge 3"
    encoder = StubEncoder({new_text: [0.0, 1.0, 0.0]})
    client = ScriptedClient([])  # any adjudication would IndexError

    outcome = Deduplicator(client, "m", encoder, store=store, threshold=0.92).process(
        [mem("new", new_text, predicate="owns", obj="Fitbit")]
    )

    assert outcome.adjudications == 0
    assert [m.id for m in outcome.kept] == ["new"]
    store.close()


def test_single_valued_predicates_are_the_ones_that_can_supersede():
    from llm_long_term_memory.ingest import is_single_valued

    assert is_single_valued("lives_in"), "you live in one place"
    assert is_single_valued("works_as")
    assert is_single_valued("Lives In"), "normalized before lookup"
    assert not is_single_valued("owns"), "you can own many things"
    assert not is_single_valued("prefers")
    assert not is_single_valued("completed")


def test_within_batch_duplicates_are_caught():
    """Ten sessions of one user restate the same facts constantly; without this the
    store fills with copies before anything reaches disk."""
    a, b, c = "fact one", "fact one restated", "different fact"
    encoder = StubEncoder({a: [1.0, 0.0, 0.0], b: [0.999, 0.01, 0.0], c: [0.0, 0.0, 1.0]})
    client = ScriptedClient([{"verdict": "DUPLICATE", "reason": "same"}])

    outcome = Deduplicator(client, "m", encoder, threshold=0.92).process(
        [mem("m1", a), mem("m2", b), mem("m3", c)]
    )
    assert [m.id for m in outcome.kept] == ["m1", "m3"]
    assert outcome.duplicates == 1
