"""Structured output contract for extraction.

The single most consequential field here is `predicate`. P4 detects that one fact
superseded another by matching on `(subject, predicate)` — so if the model emits
`prefers_framework` for one session and `favorite_framework` for the next, the two
never collide and temporal resolution silently does nothing. The prompt supplies a
controlled vocabulary and `normalize_predicate` enforces the shape, because a free-
form relation string would make the whole temporal layer a no-op.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

MemoryTypeLiteral = Literal["semantic", "episodic", "preference", "procedural", "profile"]

# Suggested to the model in the prompt. Not enforced — an unseen relation is better
# than a wrong one forced into this list — but it anchors the common cases so that
# repeated mentions of the same attribute collide instead of scattering.
COMMON_PREDICATES = [
    "lives_in",
    "works_as",
    "works_at",
    "studies",
    "owns",
    "uses_tool",
    "prefers",
    "dislikes",
    "plans_to",
    "completed",
    "has_goal",
    "has_constraint",
    "has_relationship",
    "has_health_condition",
    "scheduled",
    "spent",
]

# Predicates where the user can hold exactly one value at a time. A new value for
# one of these *replaces* the old one — you live in one place, you hold one job.
#
# Everything else is multi-valued: owning a fern does not stop you owning a Fitbit,
# and liking Python does not stop you liking Rust. The distinction decides two
# things:
#
#   * whether a `(subject, predicate)` collision is worth an LLM adjudication.
#     Treating them alike made dedup fire on every pair of `user/owns/...` facts,
#     which cost 72 LLM calls per 60 sessions and produced zero verdicts — 12x the
#     extraction budget spent to change nothing.
#   * whether P4 may supersede. Closing `valid_to` on "the user owns a peace lily"
#     because they later mention a snake plant would be simply wrong.
# Kept deliberately short. The cost of the two errors is not symmetric:
#
#   missing a supersede  leaves a stale fact competing with the current one — which
#                        is exactly the status quo the baselines already have, so
#                        the loss is bounded.
#   a wrong supersede    removes a still-true fact from retrieval entirely. It
#                        cannot be recovered downstream and directly produces wrong
#                        answers.
#
# So the list only holds predicates that are single-valued by the nature of the
# thing, and everything doubtful stays out. Two entries were removed after a real
# ingest showed them chaining unrelated facts together — see D23.
SINGLE_VALUED_PREDICATES = frozenset(
    {
        "lives_in",
        "works_as",
        "works_at",
        # "which framework do I currently use" reads as one primary answer, unlike
        # the deliberately multi-valued `uses_tool`.
        "uses_framework",
    }
)

_NON_WORD = re.compile(r"[^a-z0-9]+")


def is_single_valued(predicate: str) -> bool:
    return normalize_predicate(predicate) in SINGLE_VALUED_PREDICATES


def normalize_predicate(raw: str) -> str:
    """snake_case, no leading/trailing separators.

    Applied to every extracted predicate so that `Prefers Framework`,
    `prefers-framework`, and `prefers_framework` land on the same key.
    """
    return _NON_WORD.sub("_", raw.strip().lower()).strip("_")


class ExtractedMemory(BaseModel):
    session_index: int = Field(
        description="0-based index of the session in the provided batch that this fact came from"
    )
    type: MemoryTypeLiteral
    content: str = Field(
        description="A self-contained statement. Must make sense with no other context: "
        "no pronouns referring outside itself, no 'this' or 'that'."
    )
    subject: str = Field(description="Who or what the fact is about. Usually 'user'.")
    predicate: str = Field(description="The relation, in snake_case")
    object: str = Field(description="The value of the relation")
    entities: list[str] = Field(
        default_factory=list, description="Named entities mentioned: people, places, products, orgs"
    )
    importance: float = Field(
        ge=0.0, le=1.0, description="How likely this is to matter in a later conversation"
    )
    replaces_previous: bool = Field(
        default=False,
        description=(
            "True only if the user explicitly signalled that this replaces something "
            "they said before — 'I switched to X', 'I no longer do Y', 'I moved from "
            "A to B'. False for a plain new statement."
        ),
    )

    @field_validator("predicate")
    @classmethod
    def _normalize(cls, v: str) -> str:
        return normalize_predicate(v)

    @field_validator("subject")
    @classmethod
    def _lower_subject(cls, v: str) -> str:
        return v.strip().lower() or "user"


class ExtractionResult(BaseModel):
    memories: list[ExtractedMemory] = Field(default_factory=list)


DedupVerdict = Literal["DUPLICATE", "UPDATE", "DISTINCT"]


class DedupDecision(BaseModel):
    verdict: DedupVerdict = Field(
        description=(
            "DUPLICATE: the same fact, already stored. "
            "UPDATE: the same attribute with a different value — the new one replaces the old. "
            "DISTINCT: different facts that happen to be worded similarly."
        )
    )
    reason: str = Field(description="One short sentence")
