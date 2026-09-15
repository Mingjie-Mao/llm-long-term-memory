"""Where a required fact stops, layer by layer, with no provider call.

`source_session_recalled` answers "did any labelled source session survive into the
context". It does not answer "did the fact the question actually needs survive", and the
2026-09-13 review recorded that the 98.3% / 55% gap cannot be read as a reasoning-error
rate for exactly that reason. BEAM makes the stronger question answerable: its rubric
items name the required fact in words — `58%`, `7 pages`, `Five days` — so the fact can be
looked for directly, in the source, in the memories, in the context and in the answer.

Four layers, fixed in `results/prereg-beam-v1.md` as mandatory secondary metrics:

    1  any source session      at least one labelled source session reached the context
    2  all source sessions     every labelled source session reached it
    3  required-fact coverage  the rubric's own facts reached the final context
    4  answer utilisation      of the facts that did reach it, how many the answer used

Layers 1 and 2 are already on every row. Layers 3 and 4 are what this adds.

## Why the matcher refuses most items

A rubric item earns a verdict only when it carries a **distinctive** token: money, a
percentage, a written date, a number with a unit from a closed list, or a multi-word
proper noun. `LLM response should contain: mention of cultural differences` carries none,
and a bare number does not count either — "26" occurs somewhere in any 500K-token
conversation, so matching on it would manufacture evidence.

That refusal is the point. Every automated classification this project has attempted was
confidently wrong (`results/failure-stages.md` assigns its stages by hand for that
reason), so this one reports **present / absent / undecidable** and publishes the
undecidable share beside every figure it produces. On the development half it decides
about a fifth of all rubric items, concentrated in the abilities whose answers are
numbers and dates, and it decides almost nothing in instruction following, preference
following and contradiction resolution — whose rubrics name behaviours, not facts.

## The self-check that makes it usable

BEAM labels which sessions hold each question's evidence, so the matcher can be graded
against a ground truth that already exists: a required fact should be findable in its own
labelled source sessions. On the development half it is, for 84% of evidence-bearing
items (`tools/beam_fact_coverage.py --validate`). That number is the instrument's own
ceiling and is reported with any result read through it.

A fact absent from the whole conversation is not a miss — it is **derived**, the answer
to a temporal or aggregation question rather than something stated. Those are excluded
from layers 1 to 3 and kept only for layer 4, or a duration question would read as total
extraction loss.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from statistics import fmean

MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"

UNITS = [
    "day", "week", "month", "year", "hour", "minute", "second", "page", "mile",
    "kilometre", "kilometer", "km", "kg", "gram", "pound", "lb", "litre", "liter", "ml",
    "calorie", "degree", "time", "session", "class", "lesson", "recipe", "book",
    "episode", "chapter", "player", "people", "person", "ticket", "seat", "set", "rep",
    "lap", "step", "point", "item", "night", "meal", "dish", "serving", "bottle", "box",
    "piece", "room", "word",
]  # fmt: skip
"""A closed list, so that "20 till" and "2024 deadline" are not read as quantities. The
cost of the list being short is an undecidable item; the cost of it being open is a
verdict that looks measured and is not."""

_UNIT = "|".join(sorted(set(UNITS), key=len, reverse=True))

PAYLOAD = re.compile(r"^\s*(?:LLM\s+)?[Rr]esponse should [a-z ]{0,30}?:\s*")
NEGATIVE = re.compile(r"should (?:avoid|not )", re.I)

MONEY = re.compile(r"\$\s?\d[\d,]*(?:\.\d{1,2})?")
# No trailing \b after the sign: `%` is not a word character, so `%\b` needs a letter
# after it and "58%" at the end of a rubric item matched nothing.
PERCENT = re.compile(r"\b\d[\d,]*(?:\.\d+)?\s*(?:%|percent\b)", re.I)
DATE = re.compile(
    rf"\b(?:{MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b|\b(?:{MONTHS})\s+\d{{4}}\b",
    re.I,
)
QUANTITY = re.compile(rf"\b(\d[\d,]*(?:\.\d+)?)[\s-]*({_UNIT})s?\b", re.I)
# Two or more capitalised words. "and" is not a connector: it joined "OnePlus 10 Pro and
# Samsung Galaxy" into one token that matched nothing.
NAME = re.compile(
    r"\b[A-Z][a-zA-Z]{2,}(?:\s+(?:of|the|de|la)\s+|\s+)[A-Z][a-zA-Z]{2,}"
    r"(?:\s+[A-Z][a-zA-Z]{2,}){0,2}\b"
)
WORD_NUMBERS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
}  # fmt: skip

UNDECIDABLE = "undecidable"
NEGATIVE_ITEM = "negative"
DERIVED = "derived"
EVIDENCE = "evidence"


def payload(item: str) -> str:
    """The rubric item without its `LLM response should mention:` frame."""
    match = PAYLOAD.match(item)
    return item[match.end() :] if match else item


def normalise(text: str) -> str:
    """One spelling for both sides of every comparison."""
    text = text.lower().replace(",", "")
    text = re.sub(r"(\d)\s*(?:%|percent)\b", r"\1%", text)
    text = re.sub(r"\$\s+", "$", text)
    text = re.sub("[-\u2013\u2014]", " ", text)  # hyphen, en dash, em dash
    text = re.sub(rf"\b({_UNIT})s\b", r"\1", text)
    text = re.sub(r"(\d)(?:st|nd|rd|th)\b", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def distinctive_tokens(text: str) -> set[str]:
    """Every token distinctive enough that finding it means something."""
    found: set[str] = set()
    for pattern in (MONEY, PERCENT, DATE):
        found |= {match.group(0) for match in pattern.finditer(text)}
    spelled = text
    for word, digit in WORD_NUMBERS.items():
        spelled = re.sub(rf"\b{word}\b", digit, spelled, flags=re.I)
    found |= {f"{m.group(1)} {m.group(2)}" for m in QUANTITY.finditer(spelled)}
    # The opening word of a rubric item is capitalised because it opens a sentence, not
    # because it names anything: "Discussing recipe scaling", "Partial decoration".
    head = re.match(r"^\W*\w+", text)
    head_end = head.end() if head else 0
    for match in NAME.finditer(text):
        if match.start() < head_end or re.match(r"^\w+ing\b", match.group(0)):
            continue
        found.add(match.group(0))
    return {normalise(token) for token in found if normalise(token)}


@dataclass(slots=True)
class Item:
    """One rubric item, with the facts it requires and where they were found."""

    number: int
    kind: str
    tokens: tuple[str, ...] = ()
    reached: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def counts(self) -> bool:
        return self.kind == EVIDENCE


def haystack(*parts: str) -> str:
    return normalise("\n".join(part for part in parts if part))


def classify(item: str, conversation: str) -> Item:
    """Whether this item can be judged at all, and by which tokens.

    `conversation` is every turn of the conversation, normalised. A token absent from all
    of it was never stated, so the item is derived rather than lost.
    """
    if NEGATIVE.search(item):
        return Item(0, NEGATIVE_ITEM)
    tokens = distinctive_tokens(payload(item))
    if not tokens:
        return Item(0, UNDECIDABLE)
    if not any(token in conversation for token in tokens):
        return Item(0, DERIVED, tuple(sorted(tokens)))
    return Item(0, EVIDENCE, tuple(sorted(tokens)))


STAGES = ("source", "extracted", "retrieved", "context", "answer")
"""In pipeline order. A fact's first absence is where it was lost."""


def ladder(
    rubric: Sequence[str],
    conversation: str,
    stage_text: dict[str, str],
) -> list[Item]:
    """Every rubric item, with the stages its required tokens survived to.

    `stage_text` maps each name in `STAGES` to the text that stage supplied — the labelled
    source turns, every memory of the namespace, the ranked memories, the assembled
    context, the answer. Each is searched independently rather than assumed nested, so a
    fact that reappears at a later stage is visible rather than silently dropped.
    """
    items = []
    for number, raw in enumerate(rubric, start=1):
        item = classify(raw, conversation)
        item.number = number
        if item.tokens:
            item.reached = {
                stage: tuple(token for token in item.tokens if token in stage_text.get(stage, ""))
                for stage in STAGES
            }
        items.append(item)
    return items


def question_report(items: Sequence[Item], rubric_score: float | None = None) -> dict:
    """The per-question ladder: how many required facts each stage still had.

    Two of the buckets are not the system's fault, and both are recorded because this
    project has mistaken each for a system failure before — three times a low score turned
    out to be the metric rather than the pipeline (D30). MemTrace's taxonomy
    (arXiv:2605.28732) names them Annotation Error and LLM-as-a-Judge Error; here they
    surface as a fact missing from its own labelled source, and a rubric scored zero on an
    answer that contains every fact it asked for.
    """
    counting = [item for item in items if item.counts]
    required = sum(len(item.tokens) for item in counting)
    reached = {
        stage: sum(len(item.reached.get(stage, ())) for item in counting) for stage in STAGES
    }
    in_context = reached["context"]
    return {
        "required_facts": required,
        "reached": reached,
        "decidable_items": len(counting),
        "derived_items": sum(1 for item in items if item.kind == DERIVED),
        "undecidable_items": sum(1 for item in items if item.kind == UNDECIDABLE),
        "negative_items": sum(1 for item in items if item.kind == NEGATIVE_ITEM),
        # Layer 3: did the rubric's own facts reach the final context?
        "required_fact_coverage": (in_context / required) if required else None,
        # Layer 4: of those that did, how many did the answer actually use?
        "answer_utilisation": (reached["answer"] / in_context) if in_context else None,
        "first_loss": _first_loss(reached, required),
        # The fact is not in the sessions BEAM labels as its source. Either the label is
        # incomplete or the matcher missed it; it is not extraction loss.
        "suspected_annotation_gap": bool(required) and reached["source"] < required,
        # Every fact the rubric asked for is in the answer, and the judge scored nothing.
        "suspected_judge_error": (
            bool(required) and reached["answer"] == required and rubric_score == 0.0
        ),
    }


def _first_loss(reached: dict[str, int], required: int) -> str | None:
    """The earliest stage that lost a fact, which is where a fix belongs."""
    if not required:
        return None
    previous = required
    for stage in STAGES:
        if reached[stage] < previous:
            return stage
        previous = reached[stage]
    return None


def layers(rows: Iterable[dict], per_question: dict[str, dict]) -> dict:
    """The four registered layers over a run, averaged by conversation.

    Layers 1 and 2 come off the row, which has recorded them since the staged-recall work;
    3 and 4 come from `per_question`, keyed by question id.
    """
    from llm_long_term_memory.evaluation.beam_report import conversation_of

    buckets: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        notes = row.get("notes") or {}
        name = conversation_of(row["question_id"], row["question_type"])
        bucket = buckets.setdefault(name, {key: [] for key in _LAYERS})
        stages = notes.get("recall_stages") or {}
        coverage = notes.get("recall_coverage") or {}
        if stages.get("selected") is not None:
            bucket["any_source_session"].append(float(bool(stages["selected"])))
        if coverage.get("selected") is not None:
            bucket["all_source_sessions"].append(float(coverage["selected"] == 1.0))
        report = per_question.get(row["question_id"]) or {}
        for key in ("required_fact_coverage", "answer_utilisation"):
            if report.get(key) is not None:
                bucket[key].append(float(report[key]))

    out = {}
    for key in _LAYERS:
        per_conversation = [fmean(b[key]) for b in buckets.values() if b[key]]
        out[key] = {
            "conversations": len(per_conversation),
            "value": fmean(per_conversation) if per_conversation else None,
        }
    return out


_LAYERS = (
    "any_source_session",
    "all_source_sessions",
    "required_fact_coverage",
    "answer_utilisation",
)
