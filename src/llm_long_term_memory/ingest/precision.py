"""Of what the extractor wrote, how much is actually in the conversation?

`fidelity` measures recall: a specific the user stated, did it survive into a memory.
It is deliberately blind to the opposite question, and `batch-size-result.md` says so
in as many words: the ruler cannot see what the extractor invented, so a rising score
is a necessary condition and not a sufficient one. That blindness has now cost a
decision. `gemini-3.5-flash-lite` scored 71.6% against the shipped model's 64.9% while
writing forty per cent more memories, and on a recall-only ruler those two facts are
the same fact: writing more retains more, mechanically. The batch-size curve shows the
same shape — 4.33 memories per session for 64.2%, 13.10 for 78.4%.

This is the mirror, built from the same parts so the two numbers are commensurable:

    fidelity   specifics the user stated  ->  present in some memory?
    precision  specifics a memory asserts ->  present in the conversation?

A memory that states a number, duration, amount, date, relative time or proper noun
that appears nowhere in its session is asserting something the source does not support.

**What it is not.** This is not a hallucination detector. It checks the *specifics* a
memory carries, not its claim: "the user dislikes running" is unsupported in a way no
string match will find, and a memory carrying no detectable specific cannot be checked
at all and is counted apart rather than scored as clean. It also reads the whole
session, user and assistant turns alike, because a memory about what the assistant
recommended is properly grounded in an assistant turn — scoring it against the user's
words only would invent unsupported memories out of correct ones.

So a rise here is evidence of a real problem; a zero is not evidence of none.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_long_term_memory.conversation import ConversationSession
from llm_long_term_memory.store import Memory

from .fidelity import _extract_facets


def session_text(session: ConversationSession) -> str:
    """Everything said in the conversation, both speakers.

    Wider than `user_assertions` on purpose. Recall asks what the *user* stated and
    should not be credited for the assistant's words; support is the other direction,
    and an assistant turn is a legitimate source for a memory about a recommendation.
    """
    return "\n".join(turn.content for turn in session.turns)


@dataclass(slots=True)
class FacetSupport:
    facet: str
    asserted: int
    supported: int

    @property
    def rate(self) -> float:
        return self.supported / self.asserted if self.asserted else 1.0


@dataclass(slots=True)
class PrecisionReport:
    memories: int
    memories_with_a_checkable_specific: int
    memories_with_an_unsupported_specific: int
    asserted: int
    supported: int
    per_facet: dict[str, FacetSupport] = field(default_factory=dict)
    examples: list[tuple[str, str, str]] = field(default_factory=list)
    """(session id, the unsupported value, the memory that asserted it)."""

    @property
    def unsupported_memory_rate(self) -> float:
        """Of the memories that can be checked, how many assert something absent."""
        if not self.memories_with_a_checkable_specific:
            return 0.0
        return self.memories_with_an_unsupported_specific / self.memories_with_a_checkable_specific

    @property
    def specific_support_rate(self) -> float:
        return self.supported / self.asserted if self.asserted else 1.0

    @property
    def uncheckable_memories(self) -> int:
        """Memories carrying no detectable specific. Neither clean nor dirty."""
        return self.memories - self.memories_with_a_checkable_specific


def score_support(
    pairs: list[tuple[ConversationSession, list[Memory]]], keep_examples: int = 6
) -> PrecisionReport:
    """How much of what was written is in the conversation it came from."""
    report = PrecisionReport(0, 0, 0, 0, 0)
    for session, memories in pairs:
        # `.lower()`, exactly as `score_sessions` normalises its haystack, and
        # `_extract_facets` already lowercases what it returns. The two rulers have
        # to share a normalisation or their numbers are not comparable.
        haystack = session_text(session).lower()
        for memory in memories:
            report.memories += 1
            written = f"{memory.content} {memory.object or ''}"
            values = {v for group in _extract_facets(written).values() for v in group}
            if not values:
                continue
            report.memories_with_a_checkable_specific += 1
            unsupported_here = False
            for facet, group in _extract_facets(written).items():
                score = report.per_facet.setdefault(facet, FacetSupport(facet, 0, 0))
                for value in group:
                    score.asserted += 1
                    report.asserted += 1
                    if value in haystack:
                        score.supported += 1
                        report.supported += 1
                        continue
                    unsupported_here = True
                    if len(report.examples) < keep_examples:
                        report.examples.append((session.session_id, value, memory.content[:100]))
            if unsupported_here:
                report.memories_with_an_unsupported_specific += 1
    return report
