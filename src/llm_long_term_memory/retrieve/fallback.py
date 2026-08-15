"""Raw-conversation fallback: memory first, source when needed.

Structured memory is a lossy compression of the conversation. The pilot showed the
compression is usually good enough — `two_stage` matched naive RAG on a fiftieth of
the context — but "usually" is not "always", and the failures have a shape:
extraction keeps the gist and drops the artifact. *"The assistant recommended a Mayo
Clinic resource"* is a true memory that cannot answer *"what was the URL?"*.

The naive fix is to always attach raw evidence. That was measured
(`two_stage_hydrated`) and it tripled context for no detectable accuracy gain, which
is a slow slide back into naive RAG. So this module makes recovery **conditional**:
the answerer says whether it can answer, and only when it cannot does a second pass
pay for raw text.

Two levels, in order of cost:

* **Source-local** — the turns the retrieved memories were extracted from. Cheap and
  precise; this is the Mayo Clinic case, where retrieval already found the right
  memory and only the detail is missing.
* **Archive-wide** — BM25 over every turn in the namespace. Used only when memory
  retrieval found nothing, because extraction can miss a fact entirely.

**Neither level licenses invention.** If the archive has nothing either, the correct
answer is still "I do not know" — abstention is a measured strength (100% against
`full_context`'s 50%) and a fallback that turns misses into confident guesses would
trade it away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from llm_long_term_memory.store import Memory, MemoryStore, Turn

FallbackLevel = Literal["none", "source_local", "archive_wide"]


@dataclass(slots=True)
class RawEvidence:
    """Raw turns recovered for one question, and how they were found."""

    turns: list[Turn] = field(default_factory=list)
    level: FallbackLevel = "none"
    reason: str = ""

    @property
    def used(self) -> bool:
        return bool(self.turns)

    def render(self, max_chars: int = 2400) -> str:
        blocks = []
        budget = max_chars
        for turn in self.turns:
            text = turn.content[:budget]
            if not text:
                break
            budget -= len(text)
            blocks.append(
                f"[session {turn.session_id} · turn {turn.turn_index} · {turn.role}]\n{text}"
            )
        return "\n\n".join(blocks)


class RawFallback:
    """Finds original conversation turns when structured memory is insufficient."""

    def __init__(self, store: MemoryStore, max_turns: int = 3) -> None:
        self.store = store
        self.max_turns = max_turns

    def recover(self, user_id: str, query: str, memories: list[Memory]) -> RawEvidence:
        """Level 1 when memories exist, level 2 when they do not.

        The branch is on whether retrieval found anything, not on how confident it
        was: a memory that was found but lacks a detail points *at* the turn holding
        it, which is strictly better evidence than a keyword search over everything.
        """
        if memories:
            turns = self.store.turns_for_memories(memories)[: self.max_turns]
            if turns:
                return RawEvidence(
                    turns=turns,
                    level="source_local",
                    reason=(
                        "Structured memory identified the source but did not "
                        "preserve the detail asked for."
                    ),
                )

        turns = self.store.search_turns(user_id, query, limit=self.max_turns)
        if turns:
            return RawEvidence(
                turns=turns,
                level="archive_wide",
                reason=(
                    "No structured memory covered the question; searched the raw "
                    "conversation archive directly."
                ),
            )
        return RawEvidence(
            level="none",
            reason="Neither structured memory nor the raw archive contains this.",
        )
