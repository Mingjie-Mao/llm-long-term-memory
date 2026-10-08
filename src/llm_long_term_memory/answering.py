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

import re
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


NOTED_ANSWER_PROMPT_VERSION = "memory-aware-v2-notes"
NOTED_ANSWER_SYSTEM = (
    f"{ANSWER_SYSTEM}\n\n"
    "Before deciding, write `notes`: copy every memory or excerpt line that bears on the "
    "question, each with the date of the conversation it came from. For a total or a "
    "count, list every item as its own note before adding them up. For a duration or a "
    "gap between dates, use the day counts written next to each conversation date rather "
    "than working them out. Use the user's own figures over general knowledge. Then "
    "answer from the notes."
)


# memory-aware-v2-cl  2026-10-07 (results/prereg-raw-primary-count-latest-v1.md): v2
#                     plus a latest-value rule for every question, and item-by-item
#                     notes for questions asking for a count or total. From the dev100
#                     error taxonomy: 4 knowledge-update losses answered with an older
#                     value, 5 multi-session counts or sums wrong with every gold turn
#                     in context.
COUNT_LATEST_PROMPT_VERSION = "memory-aware-v2-cl"
LATEST_RULE = (
    "When the user's conversations give different values for the same thing — a "
    "schedule, an amount, a count so far, a status, a preference — the value from the "
    "most recent conversation is the current one and the earlier values are history. "
    "Answer with the current value unless the question asks about an earlier time."
)
COUNT_RULE = (
    "This question asks for a count or a total. Before deciding, write `notes`: one note "
    "per candidate item, from every conversation provided (items are usually spread "
    "across several), each with its conversation date and the user's own words. End each "
    "note with one verdict: COUNTS; REPEAT (the same item mentioned again — a running "
    "total the user restates is the current figure, not another item); or OUTSIDE (not "
    "what was asked: another time period, someone else's, planned but not done, only "
    "suggested). Then count or add only the items marked COUNTS, and answer with that "
    "result."
)
COUNT_LATEST_SYSTEM = f"{ANSWER_SYSTEM}\n\n{LATEST_RULE}"
COUNT_NOTES_SYSTEM = f"{COUNT_LATEST_SYSTEM}\n\n{COUNT_RULE}"
_AGGREGATE = re.compile(
    r"\b(?:how\s+many|how\s+much|in\s+total|total|combined|altogether|sum\s+of|number\s+of)\b",
    re.I,
)
_AGO_QUESTION = re.compile(r"\bhow\s+(?:many|much)\b.*\bago\b", re.I)


def asks_for_aggregate(question: str) -> bool:
    """A count or total the answer must assemble; "how many days ago" is a date, not one."""
    return bool(_AGGREGATE.search(question)) and not _AGO_QUESTION.search(question)


# memory-aware-v2-clt  2026-10-07 (results/prereg-t-probe-v2.md): v2-cl, and questions
#                      about dates, durations or order also answered from notes. The
#                      t-probe-v1 diagnosis: t2's temporal fixes (5 of 8) all came
#                      through the notes path, which only "how many …" questions took.
COUNT_TIME_PROMPT_VERSION = "memory-aware-v2-clt"
TIME_RULE = (
    "This question depends on dates, durations or order. Before deciding, write `notes`: "
    "one note per relevant event, from every conversation provided, each with the date "
    "it happened and the user's own words. Use a date in square brackets when one is "
    "given; when the user says something happened today or is happening now, it is the "
    "conversation's date. Then work the answer out from those dates — days between two "
    "dates, or the order of the dates — and answer with that result."
)
_DAYS = r"(?:mon|tues|wednes|thurs|fri|satur|sun)day"
_TIME_QUESTION = re.compile(
    rf"\b(?:when|how\s+long|how\s+(?:many|much)\s+(?:days?|weeks?|months?|years?|hours?|"
    rf"minutes?|time)|ago|first|earlier|later|before|after|since|until|order|"
    rf"most\s+recent(?:ly)?|last\s+(?:week|weekend|month|year|{_DAYS})|"
    rf"this\s+(?:week|weekend|month|year)|past\s+(?:week|weekend|month|year|few)|yesterday|"
    rf"date|what\s+day)\b",
    re.I,
)


def asks_about_time(question: str) -> bool:
    """Whether the answer turns on dates, durations or order. Deliberately broad: on
    train150, dev100 and heldout100 it selects 90 of 93 temporal-reasoning questions,
    and a wrong selection costs output tokens, not evidence."""
    return bool(_TIME_QUESTION.search(question))


def count_time_system(counting: bool, timing: bool) -> str:
    """v2-clt's system prompt for one question."""
    return "\n\n".join(
        [COUNT_LATEST_SYSTEM] + ([COUNT_RULE] if counting else []) + ([TIME_RULE] if timing else [])
    )


class NotedAnswerVerdict(BaseModel):
    """`AnswerVerdict` with the evidence written first.

    A separate class rather than a subclass on purpose: structured output is generated
    in field order, and a subclass puts its new field after `answer`, so the model would
    decide before it had listed anything. The other fields and their meaning are
    `AnswerVerdict`'s, and the fallback path reads them the same way.
    """

    notes: list[str] = Field(
        default_factory=list,
        description="Every relevant memory or excerpt line, each with its conversation date",
    )
    status: Literal["answer", "need_source", "no_evidence"] = Field(
        description=AnswerVerdict.model_fields["status"].description
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
