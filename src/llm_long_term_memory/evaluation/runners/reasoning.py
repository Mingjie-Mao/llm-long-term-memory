"""Opt-in v3 answer policy for questions that require deriving an answer.

The classifier uses only the user's question. It never consumes benchmark labels,
gold answers or answer-session ids, so the same path is available in a real product.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field

from .base import ANSWER_SYSTEM, AnswerVerdict

REASONED_ANSWER_PROMPT_VERSION = "memory-reasoned-v3.1"

ReasoningKind = Literal[
    "direct",
    "temporal",
    "multi_session_aggregation",
    "current_state",
    "preference_application",
]

_TEMPORAL = re.compile(
    r"\b(?:how long|when did|elapsed|before|after|since|ago|"
    r"(?:first|earlier|later)\b|older\b.*\bthan)\b",
    re.IGNORECASE,
)
_AGGREGATION = re.compile(
    r"\b(?:how many|total|count|average|mean|sum|combined|altogether|"
    r"all (?:the|my)|different|each)\b",
    re.IGNORECASE,
)
_CURRENT_STATE = re.compile(
    r"\b(?:current(?:ly)?|right now|latest|most recent|still|switched|changed|"
    r"prefer now|use now)\b",
    re.IGNORECASE,
)
_PREFERENCE = re.compile(
    r"\b(?:recommend|suggest|advice|tips?|should i|what should|best for me)\b",
    re.IGNORECASE,
)

_KIND_GUIDES: dict[ReasoningKind, str] = {
    "direct": (
        "Locate the smallest set of directly relevant facts. Do not add a calculation "
        "or assumption the question does not require."
    ),
    "temporal": (
        "Identify the two date anchors before doing arithmetic: use the stated event date, "
        "not merely the chat session date, and use the supplied question/today date only "
        "when the question asks how long ago. For 'between' or a duration, subtract the "
        "start from the end without adding an inclusive day unless requested. If only event "
        "order is asked, compare the dates without inventing an interval. Check the requested "
        "unit before answering."
    ),
    "multi_session_aggregation": (
        "Define the counted unit first, collect matching facts across sessions, and merge "
        "duplicates that describe the same item or event. For days of participation, count "
        "the distinct stated dates rather than the elapsed calendar span. For a sum or "
        "average, list the inputs and check the operation before answering."
    ),
    "current_state": (
        "Order conflicting statements by date. For a question about the present, use the "
        "newest explicit update; keep older facts only when the question asks about the past."
    ),
    "preference_application": (
        "Treat preferences and possessions as constraints for a useful recommendation. "
        "Do not refuse merely because the history does not contain a ready-made answer."
    ),
}

REASONED_ANSWER_SYSTEM = (
    ANSWER_SYSTEM
    + "\n\nBefore answering, separate evidence from the conclusion. When the answer must be "
    "derived, build the relevant timeline or item list, resolve newer updates over older "
    "ones, perform the calculation, and verify units and duplicates. Never turn an "
    "unstated assumption into a remembered fact. Keep private reasoning private; the "
    "structured evidence summary and calculation must be short and auditable."
)


class ReasonedAnswerVerdict(AnswerVerdict):
    """A compact audit trail plus the existing fallback decision."""

    evidence_summary: list[str] = Field(
        default_factory=list,
        max_length=6,
        description="Up to six short evidence claims actually used; no hidden reasoning.",
    )
    calculation: str = Field(
        default="",
        description="A short date/count/update operation, or empty when none is needed.",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description=(
            "High when evidence directly supports the checked answer; medium when a safe "
            "derivation is needed; low when evidence is conflicting or incomplete."
        ),
    )


def reasoning_kind(question: str) -> ReasoningKind:
    """Choose one primary operation from question wording alone."""
    if _TEMPORAL.search(question):
        return "temporal"
    if _AGGREGATION.search(question):
        return "multi_session_aggregation"
    if _CURRENT_STATE.search(question):
        return "current_state"
    if _PREFERENCE.search(question):
        return "preference_application"
    return "direct"


def render_reasoned_prompt(context: str, date: str, question: str) -> str:
    kind = reasoning_kind(question)
    return f"""\
Here is what is known about the user, drawn from their chat history.

{context}

Today's date is {date}.

Required answer operation: {kind}
{_KIND_GUIDES[kind]}

Use only supplied evidence. If a required exact detail is absent but an on-topic memory
identifies where it came from, request source recovery. If nothing relevant is supplied,
report no evidence. Otherwise return a direct, concise answer after checking the operation.

Question: {question}
"""
