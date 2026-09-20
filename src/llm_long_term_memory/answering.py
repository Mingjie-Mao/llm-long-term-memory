"""How this system answers, and what an answer is.

This is product behaviour, not measurement: the same prompt, the same structured
verdict and the same answer record are what the live API returns and what the
benchmark scores. They lived in `evaluation/runners/base.py`, which meant the running
product's answering contract could only be read inside the harness that measures it —
and the API had to import from `evaluation` to report which prompt produced a reply.

`evaluation.runners.base` re-exports all four, so the harness and every runner keep
their existing imports. Only the direction of the dependency changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

# The disposition this prompt encodes is the difference between a memory system that
# can recall and one that can *use* what it recalls.
#
# The previous version said only "if the history does not contain the information,
# say you do not know". That produced 100% abstention — the right behaviour when a
# fact was never discussed — but it also made the model refuse on advice-shaped
# questions where the memories were retrieved correctly and simply were not a
# verbatim answer. Asked for battery tips, having retrieved "the user owns a portable
# power bank", it replied "I do not know what phone you use" (results/preference-audit.md).
#
# So the distinction drawn here is not "answer vs decline". It is *missing fact* vs
# *available context*: decline when the thing asked for was never discussed, and use
# what is known when it can shape a reasonable reply.
# Bump whenever the wording changes in a way that could move a score. It is stamped
# on every result row, because "51.6%" is meaningless six months later if nobody can
# tell which prompt produced it.
#
#   memory-aware-v1  the original: recall-focused, declined whenever no memory held
#                    the answer verbatim
#   memory-aware-v2  2026-08-14: memories are context to apply, not a lookup table,
#                    while still declining on a genuinely missing fact
ANSWER_PROMPT_VERSION = "memory-aware-v2"

ANSWER_SYSTEM = (
    "You answer a user's questions using long-term memories drawn from their earlier "
    "conversations.\n\n"
    "Those memories are context about the user, not necessarily a direct answer to "
    "the question. Use them actively: when a memory can personalize, constrain, or "
    "improve your reply, build on it explicitly rather than repeating it. Ignore "
    "memories that are irrelevant to what was asked.\n\n"
    "If the question asks for a specific fact and no memory contains it, say you do "
    "not know rather than guessing. But do not refuse merely because no memory "
    "states the answer word for word — if the memories give you enough to give a "
    "useful, personalized reply, give it.\n\n"
    "Answer directly and concisely."
)


class AnswerVerdict(BaseModel):
    """The answerer's structured reply, so a second pass is paid for only on demand.

    Returning prose alone forces a choice between always attaching raw evidence
    (measured: 3x the context for no detectable gain) and never recovering it. A
    verdict lets the model say *which* case it is in, and the second LLM call
    happens only for `need_source`.
    """

    status: Literal["answer", "need_source", "no_evidence"] = Field(
        description=(
            "answer: the memories are sufficient. "
            "need_source: a memory is on topic but the specific detail asked for "
            "(a URL, an exact figure, a product name) was not preserved in it. "
            "no_evidence: nothing provided relates to the question."
        )
    )
    answer: str = Field(default="", description="The reply, when status is 'answer'")
    reason: str = Field(
        default="",
        description="For need_source/no_evidence: what is missing, in one sentence",
    )
    source_query: str = Field(
        default="",
        description="For need_source: keywords to search the raw conversation with",
    )


@dataclass(slots=True)
class Answer:
    text: str
    context_tokens: int
    """Tokens of retrieved/assembled context placed in the prompt. Excludes the
    question and system prompt so that variants are compared on the part they
    actually control."""

    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    retrieved_ids: list[str] = field(default_factory=list)
    notes: dict = field(default_factory=dict)
