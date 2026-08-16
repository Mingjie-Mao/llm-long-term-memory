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

Two sources of candidate turns:

* **Source-local** — the turns the retrieved memories were extracted from. Precise
  when retrieval found the right memory and only the detail is missing.
* **Archive-wide** — BM25 over every turn in the namespace. Catches what extraction
  missed entirely.

**Candidates are ranked against the question before anything is truncated**, and
which source supplies them is decided by that same ranking rather than by which
query happened to run. This is not how it started, and the reason is worth keeping
because the obvious diagnosis was the wrong one.

The symptom: on the clean P10 store the Mayo question — recover a URL that
extraction dropped — stopped working, in both formal arms, after a rebuild that
*improved* retrieval.

The tempting explanation was the level branch. The first version took the source
turns whenever retrieval had returned anything, and reached the archive only when
it came back empty; nothing checked that the memories were about the question. That
is a real defect and it is fixed below. **It was not what broke Mayo.**

What broke Mayo was truncation. `turns_for_memories` returns turns ordered by
session id, and the code kept `[:max_turns]` of them. Retrieval had in fact found
the right conversation — the gold turn was among the candidates — but it sat tenth
of sixteen in an alphabetical ordering, and the three kept were all from a
conversation about live music. The level was correct; the slice threw the answer
away. On the smaller mixed store the same question retrieved nothing, fell through
to the archive, and never met the slice, which is why the defect had gone a year
without being seen.

Both defects are the same omission — nothing ranked the candidates against the
question — so ranking them fixes both. The extra cost is one FTS query; the
expensive part of this path is the second answerer call, which is unchanged.

**No level licenses invention.** If the archive has nothing either, the correct
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

    def __init__(self, store: MemoryStore, max_turns: int = 3, pool: int | None = None) -> None:
        self.store = store
        self.max_turns = max_turns
        # How deep to rank before truncating. Wider than `max_turns` because its job
        # is to place the source-local turns against the archive's, and a pool the
        # width of the output can only compare the winners. Derived rather than
        # configured: it is a property of the ranking, not a knob worth an entry in
        # every config file and a line in the arms-differ-only-in check.
        self.pool = pool or max(15, max_turns * 5)

    def recover(self, user_id: str, query: str, memories: list[Memory]) -> RawEvidence:
        """Rank the archive once, and let that ranking decide the level.

        One FTS query does both jobs: it supplies the archive-wide candidates, and
        its top hit says whether the memories found the right conversation. Neither
        source is preferred by construction any more.
        """
        ranked = self.store.search_turns(user_id, query, limit=self.pool)
        rank = {(t.session_id, t.turn_index): i for i, t in enumerate(ranked)}

        local = self.store.turns_for_memories(memories) if memories else []
        local_sessions = {t.session_id for t in local}

        # The question the level now turns on: does the best evidence for this query
        # live in a conversation the retrieved memories point at?
        #
        # If it does, they found the right conversation and merely lost a detail
        # inside it — the case source-local exists for, and their anchored turns are
        # the precise answer. If it does not, extraction missed that conversation
        # altogether and the memories are pointing somewhere else entirely, however
        # confident they look.
        #
        # Session, not turn: within one conversation BM25 routinely prefers the
        # user's question to the assistant's answer, because a question repeats the
        # query's own words. Deciding at turn granularity would read that as "the
        # archive beat the memories" and abandon a memory that had in fact found the
        # right place.
        found_the_conversation = bool(local) and (
            not ranked or ranked[0].session_id in local_sessions
        )

        if found_the_conversation:
            turns = sorted(local, key=lambda t: rank.get((t.session_id, t.turn_index), len(ranked)))
            return RawEvidence(
                turns=turns[: self.max_turns],
                level="source_local",
                reason=(
                    "Structured memory identified the source but did not "
                    "preserve the detail asked for."
                ),
            )

        if ranked:
            return RawEvidence(
                turns=ranked[: self.max_turns],
                level="archive_wide",
                reason=(
                    "Structured memory did not cover the conversation this was in; "
                    "searched the raw conversation archive directly."
                ),
            )

        return RawEvidence(
            level="none",
            reason="Neither structured memory nor the raw archive contains this.",
        )
