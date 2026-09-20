"""The conversation types this project owns.

Ingestion, extraction, provenance and the API all used to be typed against
`evaluation.datasets.longmemeval.HaystackSession`, which meant the product could not
be read — or built — without the benchmark it is measured on. Worse, those types carry
the benchmark's own labels: a turn knows whether it contains the gold answer. Product
code has no business receiving that, and the API had to fabricate a benchmark instance
with an empty gold answer to ask a live question.

These are the same shapes minus the labels. `HaystackTurn` and `HaystackSession`
subclass them and add what only a benchmark has, so the dependency points from the
benchmark to the product rather than the other way round, and every existing caller
keeps working on the objects it already has.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(slots=True)
class ConversationTurn:
    """One utterance. `role` is the speaker as the corpus names it."""

    role: str
    content: str


@dataclass(slots=True)
class ConversationSession:
    """One conversation, and the day it happened.

    `date` is kept as the corpus wrote it rather than parsed on construction: the
    temporal layer resolves relative expressions against it and records the raw form
    as provenance, so normalising it here would throw away the evidence of what was
    actually said.
    """

    session_id: str
    date: str
    turns: list[ConversationTurn] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(len(turn.content) for turn in self.turns)


@dataclass(slots=True, frozen=True)
class AnswerRequest:
    """One question put to a user's long-term memory.

    `user_id` is the tenant whose memories may answer it, and is a correctness
    boundary rather than a filter: nothing outside it may be retrieved. `asked_on` is
    the date the question is asked, because "last month" means nothing without it.

    There is deliberately no field for the expected answer. A request is what a caller
    sends; what a benchmark knows about it belongs to the benchmark.
    """

    question: str
    asked_on: str
    user_id: str

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("an answer request needs a question")
        if not self.user_id:
            raise ValueError("an answer request needs a user_id: retrieval is tenant-scoped")


@runtime_checkable
class ConversationSource(Protocol):
    """Conversations that belong in one tenant's store.

    Ingestion needs exactly two things from whatever a corpus calls its unit of work:
    which store the conversations go into, and what they are. A benchmark instance
    satisfies this and carries a great deal besides — the question, the gold answer,
    the evidence session ids — none of which ingestion may see.
    """

    @property
    def store_namespace(self) -> str: ...

    @property
    def sessions(self) -> list[ConversationSession]: ...
