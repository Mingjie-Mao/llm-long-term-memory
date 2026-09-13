"""v4.0: the model identifies the operands, the code performs the operation.

The failure taxonomy and the probe reading agree on where accuracy is lost, and it is
not retrieval. On the synthesis probes v3.3 scored 42.5% on `duration` and 4.0% on
`count` while `current_state` — the operation that needs no arithmetic — scored 86.7%.
Separately, a third of all abstentions are on answers that must be computed rather than
looked up: asked how many days a book took to read, with both dates in context, the
system said it did not know.

So the split here is by aptitude. A language model is good at deciding *which two dates
the question is about* and *which memories are members of the set*; it is unreliable at
subtracting the dates and counting the members. Those two steps are arithmetic over
values it has already identified, and Python does them exactly.

This is opt-in (`answer_policy="synthesis_v4"`). The v2 and v3 arms keep their own
verdict schemas untouched, because the registered v4 protocol compares
`v3.3 retrieval + v3.3 answerer` against `v3.3 retrieval + v4.0 answerer` and a shared
schema would make that comparison two variables wide.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Literal

from pydantic import Field

from .base import AnswerVerdict

SYNTHESIS_ANSWER_PROMPT_VERSION = "memory-synthesis-v4.0"

SYNTHESIS_ANSWER_SYSTEM = (
    "You answer a user's questions using long-term memories drawn from their earlier "
    "conversations.\n\n"
    "Work in this order, and do not skip the first step.\n\n"
    "1. Decide what the question asks you to DO, and put it in `operation`:\n"
    "   count            how many of something\n"
    "   duration         how long between two points in time\n"
    "   comparison       which of two things came first\n"
    "   current_state    what holds now, for something that changed\n"
    "   lookup           everything else\n\n"
    "2. Fill the operands for that operation. Do not compute the result yourself.\n"
    "   For count: list every distinct member you can see in `items`, one short "
    "phrase each. List them even if you are unsure of the total; the total is not "
    "your job. Do not put a number in `items`.\n"
    "   For duration and comparison: put the two dates in `start_date` and "
    "`end_date` as YYYY-MM-DD, copied from the memories. Do not subtract them.\n"
    "   For current_state: several values may be recorded for the same thing over "
    "time. Use the one that holds now — the most recent, or the one not marked as "
    "having ended — and not an earlier one.\n"
    "   For lookup: answer in `answer` as usual.\n\n"
    "3. **Always write your reply in prose in `answer`, whatever the operation is.** "
    "The operand fields are working notes; `answer` is what the person reads. Leaving "
    "it empty because you filled `operation` is the single most common way this format "
    "goes wrong: the raw structure reaches the reader instead of a reply. For count and "
    "duration a total will be recomputed from your operands and may replace your "
    "wording, but `answer` must still be written.\n\n"
    "4. Only then decide whether you can answer.\n\n"
    "The order matters. Deciding 'I cannot answer' before identifying the operation "
    "is how a question you could have answered by counting or subtracting becomes "
    "'I do not know'.\n\n"
    "Abstain only when a named operand is genuinely absent — when there is no second "
    "date to subtract, or nothing to count. Then put just the NAME of what is missing "
    "in `missing_field`, a few words at most: 'end date', not a sentence and not your "
    "reasoning about whether to fill it. If nothing is missing, leave it empty. Do not "
    "abstain because no memory states the answer word for word: an answer you can "
    "derive from the memories is an answer you have.\n\n"
    "Never invent an operand to avoid abstaining. A date you did not read in a "
    "memory is worse than saying the date is missing.\n\n"
    "Answer directly and concisely."
)


SYNTHESIS_ENUMERATE_PROMPT_VERSION = "memory-synthesis-v4.2-enumerate"

# v4.2. The one change, and the reason for it.
#
# v4.0 handed arithmetic to Python and it worked: the control's stated number disagreed
# with its own item list on 27 of 30 count probes, and under v4 that is 0. The count
# score barely moved, because the number was never the problem — the list was. In the
# best run **44 gold facts sat in the retrieved context and were never named**, and
# roughly four out of five wrong counts were too low
# (`results/analysis/v4-probe-diagnosis.md`).
#
# Free-text `items` gives the model nothing to be complete *against*. It writes a list
# that reads plausibly and stops, and neither the model nor the code can tell that a
# memory was skipped, because a skipped memory leaves no trace.
#
# So members are selected by the label of the memory they come from. Every memory in the
# context carries one, the model names labels rather than phrases, and the code can then
# check what free text made uncheckable: which labels were in context, which were named,
# which were named twice, and which do not exist at all. Under-enumeration stops being
# something only a post-hoc analysis can see.
#
# Labelling the context and selecting by label are one mechanism, not two bundled
# changes: labels nobody cites are decoration, and citations with nothing to cite cannot
# be written. That distinction is the v4.0 design fault, so it is stated rather than
# assumed.
# The instruction is *appended* to the original count sentence rather than replacing it.
#
# The first version replaced the whole block with a longer one about working through labels
# in order. Measured on eight live probes, the candidate then omitted `answer` on three of
# four — including `current_state` and `comparison`, where no arithmetic rescues it. On one
# `current_state` probe the control answered substantively and the candidate returned
# "I do not know." purely from the missing field. Rewriting the shared prompt to emphasise
# one operation cost the others their prose, which is a second difference between the arms
# and exactly the confound the single-variable rule exists to prevent.
#
# So the original wording is left byte-identical and two sentences are added after it. The
# iteration was on that format property alone — whether `answer` is populated — and no
# verdict or accuracy figure was read while deciding it.
SYNTHESIS_ENUMERATE_ANSWER_SYSTEM = SYNTHESIS_ANSWER_SYSTEM.replace(
    "phrase each. List them even if you are unsure of the total; the total is not "
    "your job. Do not put a number in `items`.\n",
    "phrase each. List them even if you are unsure of the total; the total is not "
    "your job. Do not put a number in `items`. Every memory in the context is "
    "labelled [M1], [M2] and so on; put the label of each member in `member_labels`, "
    "reading through the context in label order so that none is skipped. Cite only "
    "labels you can see, and name both when two labels record the same thing.\n",
)


Operation = Literal["lookup", "count", "duration", "comparison", "current_state"]

# Accepts what the answerer actually emits: a bare date, or a full timestamp it copied
# out of a rendered memory.
_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


class SynthesisVerdict(AnswerVerdict):
    """Operands, not conclusions, for anything the code can compute.

    `answer` stays on the schema and stays authoritative for `lookup` and
    `current_state`. For `count`, `duration` and `comparison` the code recomputes from
    the operand fields and overwrites it, so a model that enumerates correctly and then
    miscounts still produces the right number.
    """

    # Required here, where the base leaves it optional. The prompt has asked for it on
    # every operation since v4.0 and the code already says why that is not enough — "a
    # prompt is not a contract". Two prompt revisions were measured against it on live
    # probes and both failed: the model omitted `answer` and the reader received
    # "I do not know." on questions the other arm answered substantively.
    #
    # Overridden on the v4 base rather than on one arm, so both v4 arms carry the same
    # requirement and the comparison still differs only in count-member citation. The v2
    # and v3 verdicts are untouched, and the archived v4 runs keep their own frozen
    # source, so nothing already measured changes meaning.
    answer: str = Field(
        description="The reply the person reads. Required on every operation, including "
        "those whose total the code recomputes."
    )

    operation: Operation = Field(
        default="lookup",
        description=(
            "count: the question asks how many. duration: how long between two dates. "
            "comparison: which of two things came first. current_state: what holds now "
            "for something that changed. lookup: everything else."
        ),
    )
    items: list[str] = Field(
        default_factory=list,
        max_length=40,
        description=(
            "For count: one short phrase per distinct member, listed before counting. "
            "List every member you can see; do not state a total here."
        ),
    )
    start_date: str = Field(
        default="", description="For duration/comparison: the earlier date, YYYY-MM-DD."
    )
    end_date: str = Field(
        default="", description="For duration/comparison: the later date, YYYY-MM-DD."
    )
    missing_field: str = Field(
        default="",
        description=(
            "The NAME of an absent operand, and nothing else — at most a few words, "
            "such as 'end date' or 'start and end date'. Leave it empty when nothing "
            "is absent. Do not write reasoning, deliberation, or an explanation of "
            "whether to fill this field."
        ),
    )


class EnumeratingSynthesisVerdict(SynthesisVerdict):
    """v4.2: count members are cited by label, so completeness becomes checkable.

    A subclass rather than a new field on `SynthesisVerdict`, because the registered
    comparison is v4.0's answerer against this one and a shared schema would put the
    field in both arms — which is how `answer_confidence` came to be read as a
    measurement on v3 rows that never carried it.
    """

    member_labels: list[str] = Field(
        default_factory=list,
        max_length=60,
        description=(
            "For count: the label of every memory that is a member of the set, such as "
            "'M3'. Work through the context in label order and cite every member you "
            "find. Cite only labels present in the context."
        ),
    )


_LABEL = re.compile(r"M\s*(\d+)", re.IGNORECASE)


def normalise_label(value: str) -> str:
    """`[M3]`, `m3`, `M 3` and `M03` are the same citation.

    The model is copying a token out of rendered text, so the failure to expect is
    transcription noise, not invention. Normalising it here keeps a correctly chosen
    member from being discarded over a bracket.
    """
    match = _LABEL.search(value or "")
    return f"M{int(match.group(1))}" if match else ""


class Enumeration:
    """What the citation check found. Every field is a count of labels, not a score."""

    __slots__ = ("duplicated", "in_context", "named", "uncited", "unknown")

    def __init__(
        self,
        named: list[str],
        unknown: list[str],
        duplicated: list[str],
        in_context: int,
        uncited: int,
    ) -> None:
        self.named = named
        self.unknown = unknown
        self.duplicated = duplicated
        self.in_context = in_context
        self.uncited = uncited

    def as_detail(self) -> dict:
        return {
            "members_cited": len(self.named),
            "labels_in_context": self.in_context,
            "labels_uncited": self.uncited,
            "labels_cited_twice": len(self.duplicated),
            "labels_not_in_context": self.unknown,
        }


def check_citations(labels: list[str], context_labels: set[str]) -> Enumeration:
    """Resolve cited labels against the ones the context actually carried.

    A label that is not in context is dropped rather than counted. It cannot be a member
    of a set drawn from memories the answerer was shown, and counting it would let an
    invented citation inflate a total that is meant to be auditable.
    """
    named: list[str] = []
    unknown: list[str] = []
    duplicated: list[str] = []
    seen: set[str] = set()
    for raw in labels:
        label = normalise_label(raw)
        if not label or label not in context_labels:
            unknown.append(raw)
            continue
        if label in seen:
            duplicated.append(label)
            continue
        seen.add(label)
        named.append(label)
    return Enumeration(
        named, unknown, duplicated, len(context_labels), len(context_labels) - len(named)
    )


# Deliberation words. The field asks for the name of an absent operand; on the v4.0-flat
# run three of five populated values were the model thinking out loud inside it —
# "none needed for current_state unless absent, which it is not here, so leave empty..."
# — which then blocked a computation that should have run. Detected rather than parsed
# away, so the two cases can be counted separately next time instead of being one
# ambiguous signal.
_NARRATION = re.compile(
    r"\b(?:wait|unless|actually|let's|let us|however|but|since|because|"
    r"schema|prompt|rule says|i should|we should|handled by)\b",
    re.IGNORECASE,
)
_MAX_MISSING_FIELD_WORDS = 8


def missing_field_is_narration(value: str) -> bool:
    """Is this the name of a missing operand, or the model reasoning at us?"""
    text = (value or "").strip()
    if not text:
        return False
    return len(text.split()) > _MAX_MISSING_FIELD_WORDS or bool(_NARRATION.search(text))


def _parse_date(value: str) -> date | None:
    match = _DATE.search(value or "")
    if not match:
        return None
    try:
        return datetime(*(int(part) for part in match.groups())).date()
    except ValueError:
        return None


def _normalise(item: str) -> str:
    """Fold the spelling differences that make one member look like two.

    Deliberately conservative: case, surrounding punctuation and article words only.
    Anything cleverer — stemming, or similarity — would start merging genuinely
    different members, and over-counting is the failure this exists to fix.
    """
    text = re.sub(r"[^\w\s]", " ", item.lower())
    words = [w for w in text.split() if w not in {"a", "an", "the"}]
    return " ".join(words)


def deduplicate(items: list[str]) -> list[str]:
    """Distinct members, first spelling kept, order preserved."""
    seen: set[str] = set()
    kept: list[str] = []
    for item in items:
        key = _normalise(item)
        if key and key not in seen:
            seen.add(key)
            kept.append(item.strip())
    return kept


class ComputationResult:
    """What the code derived, and whether it could derive anything at all."""

    __slots__ = ("answer", "computed", "detail")

    def __init__(self, answer: str, detail: dict, computed: bool) -> None:
        self.answer = answer
        self.detail = detail
        self.computed = computed


def compute(verdict: SynthesisVerdict, context_labels: set[str] | None = None) -> ComputationResult:
    """Perform the operation the verdict describes.

    Returns `computed=False` when the operands are absent or malformed. The caller then
    keeps the model's own prose rather than substituting a number derived from nothing —
    a wrong answer stated confidently by the code would be worse than the same wrong
    answer stated by the model, because it would carry a derivation that looks checked.
    """
    # A verdict that names a missing operand and then supplies operands anyway is
    # contradicting itself, and computing from the half it filled in is how an
    # abstention becomes a confidently wrong number with a derivation attached. The
    # model's own words win: it said something was missing.
    stated_missing = verdict.missing_field.strip()
    if stated_missing and not missing_field_is_narration(stated_missing):
        return ComputationResult(
            "",
            {
                "operation": verdict.operation,
                "reason": "the verdict named a missing operand",
                "missing_field": stated_missing,
            },
            False,
        )

    if verdict.operation == "count":
        # v4.2: when the verdict cites labels and the caller supplied the labels the
        # context carried, membership is resolved against them. The phrases in `items`
        # are still what the reader sees, but the *count* comes from citations that can
        # be checked against the context — which is what free text could not offer.
        cited = getattr(verdict, "member_labels", None)
        enumeration = None
        if cited and context_labels is not None:
            enumeration = check_citations(cited, context_labels)
            if not enumeration.named:
                return ComputationResult(
                    "",
                    {
                        "operation": "count",
                        "reason": "no cited label was present in the context",
                        **enumeration.as_detail(),
                    },
                    False,
                )

        if not verdict.items and enumeration is None:
            return ComputationResult("", {"reason": "no items enumerated"}, False)

        if enumeration is not None:
            # The label set is the membership decision; phrases only name it. Pairing
            # them positionally recovers a readable answer when the model supplied both,
            # and falls back to the labels themselves when it did not — a count whose
            # members are auditable is worth more than one whose members read nicely.
            phrases = [item.strip() for item in verdict.items if item.strip()]
            listed = (
                ", ".join(phrases)
                if len(phrases) == len(enumeration.named)
                else ", ".join(enumeration.named)
            )
            return ComputationResult(
                f"{len(enumeration.named)} distinct: {listed}",
                {
                    "operation": "count",
                    "items_stated": len(verdict.items),
                    "items_distinct": len(enumeration.named),
                    "items": phrases,
                    "count": len(enumeration.named),
                    "counted_from": "cited_labels",
                    **enumeration.as_detail(),
                },
                True,
            )

        distinct = deduplicate(verdict.items)
        listed = ", ".join(distinct)
        # "N distinct: ..." rather than "N (...)". Any reader of this answer — the
        # probe grader included — has to find the asserted total among the numbers in
        # the text, and a member can itself contain one ("2 concert tickets"). Binding
        # the total to a counting word makes it unambiguous; the bare-parenthesis form
        # was measured against the grader and misread as 2.
        return ComputationResult(
            f"{len(distinct)} distinct: {listed}",
            {
                "operation": "count",
                "items_stated": len(verdict.items),
                "items_distinct": len(distinct),
                "items": distinct,
                "count": len(distinct),
                "counted_from": "free_text",
            },
            True,
        )

    if verdict.operation in {"duration", "comparison"}:
        start, end = _parse_date(verdict.start_date), _parse_date(verdict.end_date)
        if start is None or end is None:
            return ComputationResult("", {"reason": "a date operand is missing"}, False)
        earlier, later = sorted((start, end))
        if verdict.operation == "duration":
            days = (later - earlier).days
            return ComputationResult(
                f"{days} days",
                {
                    "operation": "duration",
                    "start": earlier.isoformat(),
                    "end": later.isoformat(),
                    "days": days,
                },
                True,
            )
        # `comparison` asks which came first, so the code returns the earlier operand
        # and lets the caller keep the model's wording for what that date refers to.
        return ComputationResult(
            "",
            {
                "operation": "comparison",
                "earlier": earlier.isoformat(),
                "later": later.isoformat(),
                "start_is_earlier": start <= end,
            },
            True,
        )

    return ComputationResult("", {"operation": verdict.operation}, False)
