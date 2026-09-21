"""A live turn is extracted with what came before it, or it cannot be understood.

The REST surface receives one turn at a time, and a turn refers to the conversation
around it. Extracting

    user: I bought the Sony one you recommended yesterday.

on its own can only produce "the user bought the Sony one", which names nothing and
answers nothing later. Batch ingestion never had this problem — it reads a whole
conversation — so the loss lived only on the live surface and is invisible in every
benchmark figure this project reports.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from llm_long_term_memory.api.service import LiveTurnExtractor


class _Recorder:
    """Stands in for the batch extractor and keeps what it was asked to read."""

    def __init__(self) -> None:
        self.user_id = None
        self.sessions = []

    def extract(self, sessions):
        self.sessions.append(sessions[0])
        return SimpleNamespace(memories=[], dropped_bad_index=0)


def _extractor(recorder: _Recorder) -> LiveTurnExtractor:
    from llm_long_term_memory.llm import UsageTracker

    return LiveTurnExtractor(
        batch_extractor=recorder,
        deduplicator=SimpleNamespace(
            process=lambda memories, vectors: SimpleNamespace(kept=[], duplicates=0, updates=[])
        ),
        encoder=SimpleNamespace(encode=lambda contents: []),
        usage=UsageTracker(),
    )


def _turn(role: str, content: str):
    return SimpleNamespace(role=role, content=content)


def _run(recorder: _Recorder, context):
    _extractor(recorder).extract_turn(
        user_id="alice",
        session_id="s1",
        role="user",
        content="I bought the Sony one you recommended yesterday.",
        now=datetime(2026, 9, 21, 10, 0),
        context_turns=context,
    )
    return recorder.sessions[-1]


def test_the_preceding_turns_are_shown_to_the_extractor():
    recorder = _Recorder()

    session = _run(recorder, [_turn("assistant", "I'd recommend the Sony WH-1000XM6.")])

    assert [t.content for t in session.turns] == [
        "I'd recommend the Sony WH-1000XM6.",
        "I bought the Sony one you recommended yesterday.",
    ]


def test_the_turn_being_extracted_is_always_last():
    """Order is what makes the final turn the one the facts are about."""
    recorder = _Recorder()

    session = _run(recorder, [_turn("user", "a"), _turn("assistant", "b")])

    assert session.turns[-1].content.startswith("I bought the Sony one")


def test_the_window_is_bounded_so_a_long_conversation_is_not_re_extracted():
    recorder = _Recorder()
    history = [_turn("user", f"turn {i}") for i in range(20)]

    session = _run(recorder, history)

    assert len(session.turns) == LiveTurnExtractor.CONTEXT_TURNS + 1
    # The most recent ones, not the oldest: a pronoun refers backwards a little way.
    assert [t.content for t in session.turns[:-1]] == [f"turn {i}" for i in range(16, 20)]


def test_the_first_turn_of_a_session_has_no_context_and_still_works():
    recorder = _Recorder()

    session = _run(recorder, [])

    assert len(session.turns) == 1


def test_a_deployment_can_turn_the_window_off_and_get_the_previous_behaviour(monkeypatch):
    monkeypatch.setattr(LiveTurnExtractor, "CONTEXT_TURNS", 0)
    recorder = _Recorder()

    session = _run(recorder, [_turn("assistant", "I'd recommend the Sony WH-1000XM6.")])

    assert len(session.turns) == 1


def test_the_service_hands_over_the_turns_already_on_the_session():
    """The API had them all along — `_add_message` reads them to number the new one."""
    import inspect

    from llm_long_term_memory.api.service import MemoryService

    source = inspect.getsource(MemoryService._add_message)

    assert "context_turns=existing.turns if existing else ()" in source


@pytest.mark.parametrize("stub", ["tests", "tools"])
def test_every_extract_turn_stand_in_accepts_the_context(stub):
    """A double that cannot take it would hide a caller that stopped passing it."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / stub
    for path in root.glob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if re.search(r"def extract_turn\(self, \*,", line):
                assert "context_turns" in line, f"{path.name}: {line.strip()}"
