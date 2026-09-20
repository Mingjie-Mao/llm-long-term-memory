"""v2d: auditable operands selected by the model, arithmetic performed by code.

The answerer is still responsible for the semantic decision (which facts satisfy the
question).  It is not trusted to count or calculate.  Every selected member and number
must cite a memory label that was actually present in the context; Python validates the
labels, de-duplicates count members, parses dates/decimals, and performs the operation.
"""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import Field

from .base import AnswerVerdict
from .v2c import V2C_ANSWER_SYSTEM

V2D_ANSWER_PROMPT_VERSION = "memory-aware-v2d.4-compact-schema"
V2D_ANSWER_SYSTEM = (
    f"{V2C_ANSWER_SYSTEM}\n\n"
    "Before writing the reply, classify the requested operation. Use `lookup` when no "
    "calculation is requested. For count, duration, sum, average, difference, or "
    "percentage_change, "
    "do not do the arithmetic yourself: identify the operands and let code calculate. "
    "Return operands as parallel positional arrays of strings. Put members, numbers, or "
    "dates in `operand_values` and align every item with `operand_labels`. For a count, "
    "use `operand_qualifiers` for optional groups; for numeric arithmetic use it for each "
    "operand's unit. Empty qualifiers may be omitted. Every member/value must cite the [M1], [M2], "
    "... memory label that supports it. One memory may support several distinct count "
    "members, and one member repeated in several memories is still one member. When one "
    "question asks for counts at two times or for two categories, put a short category "
    "in its qualifier. "
    "Include every in-scope "
    "member, not merely examples. Copy numeric values without converting units. For a "
    "duration, put the start and end dates in that order, cite each date's memory label "
    "(or QUESTION_DATE for today's date), and put days or weeks in `calculation_mode`. "
    "For mixed time units in "
    "sum/average/difference, keep each original unit and choose `result_unit`; code will "
    "convert. For a "
    "difference, put first_minus_second, second_minus_first, or absolute in "
    "`calculation_mode`; use absolute only when the question "
    "asks for a gap or 'how much' without a direction. For percentage_change, put the "
    "baseline first and the later/comparison value second. If a required operand is "
    "absent, name it in `missing_field`; never invent it. First check whether the "
    "question contains a factual premise about the user's history. Mark it supported, "
    "contradicted, or unknown (use not_applicable when there is no such premise). If it "
    "is contradicted, cite the memory labels that prove the conflict and state the "
    "correction; do not answer as if the premise were true. If it is unknown, say so "
    "rather than turning nearby but different facts into support. Always provide concise prose "
    "in `answer`, even though code may replace its arithmetic."
)

Operation = Literal[
    "lookup",
    "count",
    "duration",
    "sum",
    "average",
    "difference",
    "percentage_change",
    "comparison",
    "current_state",
]
PremiseStatus = Literal["not_applicable", "supported", "contradicted", "unknown"]


class V2DVerdict(AnswerVerdict):
    premise_status: PremiseStatus = Field(
        description="Whether a factual premise embedded in the question is supported."
    )
    premise_claim: str = Field(
        default="", description="The factual premise being checked, in a short phrase."
    )
    premise_evidence_labels: list[str] = Field(default_factory=list)
    premise_correction: str = Field(
        default="", description="The source-grounded correction when the premise is false."
    )
    operation: Operation = "lookup"
    operand_values: list[str] = Field(default_factory=list)
    operand_labels: list[str] = Field(default_factory=list)
    operand_qualifiers: list[str] = Field(default_factory=list)
    result_unit: str = Field(
        default="", description="Requested output unit for numeric arithmetic, if applicable."
    )
    calculation_mode: str = Field(
        default="",
        description="Duration unit or difference direction; empty for other operations.",
    )
    missing_field: str = Field(
        default="", description="Short name of a required operand that is absent."
    )


class ComputationResult:
    __slots__ = ("answer", "computed", "detail")

    def __init__(self, answer: str, detail: dict, computed: bool) -> None:
        self.answer = answer
        self.detail = detail
        self.computed = computed


# Every refusal has to be a diagnosis, not a "no". The gate16 run refused nine of its
# sixteen questions and the rows recorded one sentence each; `an item has no valid
# source label` collapses two failures that need opposite fixes — the model returned no
# member text at all, and the model cited a label that was never in its context. The
# rows cannot be classified after the fact, so the only way to tell those apart was to
# spend the questions again.
#
# A refusal now carries a stable machine-readable `cause`, the operands exactly as the
# model supplied them, and the labels that were actually available to cite. Nothing
# about what is refused changes: same conditions, same order, same `reason` text, same
# `computed` flag. Only the record gets wider.
REFUSAL_CAUSES = frozenset(
    {
        "missing_field_named",
        "operand_arrays_misaligned",
        "member_text_empty",
        "label_not_in_context",
        "no_members",
        "duration_needs_two_operands",
        "date_unparseable",
        "numeric_unparseable",
        "too_few_operands",
        "units_incompatible",
        "percentage_baseline_zero",
        "comparison_operands_invalid",
        "operation_not_computable",
    }
)


def _refused(operation: str, cause: str, reason: str, **evidence) -> ComputationResult:
    """A refusal recorded well enough to classify offline."""
    assert cause in REFUSAL_CAUSES, cause
    detail = {"operation": operation, "cause": cause}
    if reason:
        detail["reason"] = reason
    detail.update(evidence)
    return ComputationResult("", detail, False)


def _supplied_operands(verdict: V2DVerdict) -> list[dict]:
    """The operand arrays as the model returned them, aligned as far as they align."""
    qualifiers = verdict.operand_qualifiers
    return [
        {
            "index": index,
            "value": value,
            "label_supplied": (
                verdict.operand_labels[index - 1] if index <= len(verdict.operand_labels) else None
            ),
            "qualifier": qualifiers[index - 1] if index <= len(qualifiers) else None,
        }
        for index, value in enumerate(verdict.operand_values, start=1)
    ]


def _array_lengths(verdict: V2DVerdict) -> dict:
    return {
        "values": len(verdict.operand_values),
        "labels": len(verdict.operand_labels),
        "qualifiers": len(verdict.operand_qualifiers),
    }


_LABEL = re.compile(r"M\s*0*(\d+)", re.I)
_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _label(value: str) -> str:
    match = _LABEL.search(value or "")
    return f"M{int(match.group(1))}" if match else ""


def _member_key(value: str) -> str:
    text = re.sub(r"[^\w\s]", " ", value.casefold())
    return " ".join(word for word in text.split() if word not in {"a", "an", "the"})


def _decimal(value: str) -> Decimal | None:
    text = (value or "").strip().replace(",", "")
    # Currency symbols are presentation, not part of the number. Reject all other text.
    text = re.sub(r"^[\$£€¥]", "", text).strip()
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def _date(value: str):
    match = _DATE.search(value or "")
    if not match:
        return None
    try:
        return datetime(*(int(part) for part in match.groups())).date()
    except ValueError:
        return None


def _format(number: Decimal) -> str:
    value = format(number.normalize(), "f")
    return "0" if value in {"-0", ""} else value


def _unit_suffix(unit: str) -> str:
    return f" {unit.strip()}" if unit.strip() else ""


_TIME_FACTORS = {
    "minute": Decimal(1),
    "minutes": Decimal(1),
    "min": Decimal(1),
    "mins": Decimal(1),
    "hour": Decimal(60),
    "hours": Decimal(60),
    "hr": Decimal(60),
    "hrs": Decimal(60),
    "day": Decimal(1440),
    "days": Decimal(1440),
    "week": Decimal(10080),
    "weeks": Decimal(10080),
}


def _normalise_numeric_units(values, result_unit: str):
    """Convert compatible time operands through exact Decimal minute factors."""
    raw_units = [unit.strip() for _, _, unit, _ in values]
    units = [unit.casefold() for unit in raw_units]
    target = result_unit.strip().casefold()
    nonempty = {unit for unit in units if unit}
    if len(nonempty) <= 1:
        display_unit = result_unit.strip() or next((unit for unit in raw_units if unit), "")
        return [number for _, number, _, _ in values], display_unit
    if (
        not target
        or target not in _TIME_FACTORS
        or any(unit not in _TIME_FACTORS for unit in units)
    ):
        return None
    converted = [
        number * _TIME_FACTORS[unit] / _TIME_FACTORS[target]
        for (_, number, _, _), unit in zip(values, units, strict=True)
    ]
    return converted, result_unit.strip()


def _validated_operands(verdict: V2DVerdict, labels: set[str]):
    """Parse the numeric operands, or say which one failed and how.

    Returns `(values, None)` on success and `(None, rejection)` otherwise. The rejection
    is the evidence a later replay classifies the failure from: an unparseable number
    and a hallucinated citation both used to surface as `numeric operands are invalid`.
    """
    count = len(verdict.operand_values)
    if len(verdict.operand_labels) != count or (
        verdict.operand_qualifiers and len(verdict.operand_qualifiers) != count
    ):
        return None, {"cause": "operand_arrays_misaligned", "lengths": _array_lengths(verdict)}
    units = verdict.operand_qualifiers or [""] * count
    values: list[tuple[str, Decimal, str, str]] = []
    for index, (value, unit, source_label) in enumerate(
        zip(verdict.operand_values, units, verdict.operand_labels, strict=True), start=1
    ):
        label = _label(source_label)
        number = _decimal(value)
        if number is None:
            return None, {
                "cause": "numeric_unparseable",
                "index": index,
                "value": value,
                "label_supplied": source_label,
            }
        if label not in labels:
            return None, {
                "cause": "label_not_in_context",
                "index": index,
                "value": value,
                "label_supplied": source_label,
                "label_resolved": label,
            }
        values.append((f"operand_{index}", number, unit, label))
    return values, None


def compute(verdict: V2DVerdict, context_labels: set[str]) -> ComputationResult:
    """Validate cited operands and deterministically perform the requested operation."""
    premise_labels = [_label(value) for value in verdict.premise_evidence_labels]
    valid_premise_labels = [label for label in premise_labels if label in context_labels]
    if verdict.premise_status == "contradicted":
        correction = verdict.premise_correction.strip()
        if not correction or not valid_premise_labels:
            return ComputationResult(
                "I do not know.",
                {
                    "operation": "false_premise",
                    "premise_status": "contradicted",
                    "reason": "the claimed contradiction lacks a correction or valid citation",
                    "labels_supplied": premise_labels,
                    "labels_available": sorted(context_labels),
                    "correction_supplied": bool(correction),
                },
                True,
            )
        return ComputationResult(
            correction,
            {
                "operation": "false_premise",
                "premise_status": "contradicted",
                "claim": verdict.premise_claim.strip(),
                "correction": correction,
                "evidence_labels": valid_premise_labels,
            },
            True,
        )
    if verdict.premise_status == "unknown":
        claim = verdict.premise_claim.strip()
        answer = f"I do not know whether {claim}." if claim else "I do not know."
        return ComputationResult(
            answer,
            {
                "operation": "false_premise",
                "premise_status": "unknown",
                "claim": claim,
            },
            True,
        )

    if verdict.missing_field.strip():
        return _refused(
            verdict.operation,
            "missing_field_named",
            "the verdict named a missing operand",
            missing_field=verdict.missing_field.strip(),
            operands_supplied=_supplied_operands(verdict),
            labels_available=sorted(context_labels),
        )

    if verdict.operation == "count":
        count = len(verdict.operand_values)
        if len(verdict.operand_labels) != count or (
            verdict.operand_qualifiers and len(verdict.operand_qualifiers) != count
        ):
            return _refused(
                "count",
                "operand_arrays_misaligned",
                "count operand arrays are misaligned",
                lengths=_array_lengths(verdict),
                operands_supplied=_supplied_operands(verdict),
            )
        groups_in = verdict.operand_qualifiers or [""] * count
        distinct: dict[tuple[str, str], tuple[str, str]] = {}
        citations: list[dict] = []
        for member, source_label, group_value in zip(
            verdict.operand_values, verdict.operand_labels, groups_in, strict=True
        ):
            label = _label(source_label)
            key = _member_key(member)
            if not key or label not in context_labels:
                return _refused(
                    "count",
                    # Two failures wore this one sentence. An empty member is the model
                    # returning nothing to count; a label outside the context is the
                    # model citing a memory it was never shown. The first is a prompt
                    # problem, the second a grounding problem.
                    "member_text_empty" if not key else "label_not_in_context",
                    "an item has no valid source label",
                    index=len(citations) + 1,
                    member=member,
                    label_supplied=source_label,
                    label_resolved=label,
                    operands_supplied=_supplied_operands(verdict),
                    labels_available=sorted(context_labels),
                )
            group = group_value.strip()
            distinct.setdefault((group.casefold(), key), (member.strip(), group))
            citations.append({"member": member.strip(), "group": group, "label": label})
        if not distinct:
            return _refused(
                "count",
                "no_members",
                "no members were supplied",
                operands_supplied=_supplied_operands(verdict),
            )
        groups: dict[str, list[str]] = {}
        for member, group in distinct.values():
            groups.setdefault(group, []).append(member)
        names = [member for member, _ in distinct.values()]
        if set(groups) == {""}:
            answer = f"{len(names)} distinct: {', '.join(names)}"
        else:
            answer = "; ".join(
                f"{group or 'all'}: {len(members)} distinct ({', '.join(members)})"
                for group, members in groups.items()
            )
        return ComputationResult(
            answer,
            {
                "operation": "count",
                "count": len(names),
                "members": names,
                "groups": {group: len(members) for group, members in groups.items()},
                "citations": citations,
                "counted_from": "source-labelled members",
            },
            True,
        )

    if verdict.operation == "duration":
        if len(verdict.operand_values) != 2 or len(verdict.operand_labels) != 2:
            return _refused(
                "duration",
                "duration_needs_two_operands",
                "two dated operands are required",
                lengths=_array_lengths(verdict),
                operands_supplied=_supplied_operands(verdict),
            )
        start, end = (_date(value) for value in verdict.operand_values)
        start_source, end_source = verdict.operand_labels
        start_label = _label(start_source)
        end_label = _label(end_source)
        valid_start = start_label in context_labels or start_source == "QUESTION_DATE"
        valid_end = end_label in context_labels or end_source == "QUESTION_DATE"
        if start is None or end is None or not valid_start or not valid_end:
            return _refused(
                "duration",
                # An unreadable date and a citation that names nothing are different
                # repairs: one is a format the parser does not accept, the other is a
                # memory the answerer never saw.
                "date_unparseable" if start is None or end is None else "label_not_in_context",
                "a date operand or citation is invalid",
                endpoints=[
                    {
                        "name": name,
                        "value": value,
                        "parsed": parsed.isoformat() if parsed else None,
                        "label_supplied": source,
                        "label_resolved": label,
                        "cited_in_context": cited,
                    }
                    for name, value, parsed, source, label, cited in (
                        (
                            "start",
                            verdict.operand_values[0],
                            start,
                            start_source,
                            start_label,
                            valid_start,
                        ),
                        ("end", verdict.operand_values[1], end, end_source, end_label, valid_end),
                    )
                ],
                labels_available=sorted(context_labels),
            )
        earlier, later = sorted((start, end))
        days = (later - earlier).days
        duration_unit = (
            verdict.calculation_mode if verdict.calculation_mode in {"days", "weeks"} else "days"
        )
        if duration_unit == "weeks":
            weeks, remainder = divmod(days, 7)
            answer = f"{weeks} weeks" + (f" and {remainder} days" if remainder else "")
        else:
            answer = f"{days} days"
        return ComputationResult(
            answer,
            {
                "operation": "duration",
                "start": earlier.isoformat(),
                "end": later.isoformat(),
                "days": days,
                "requested_unit": duration_unit,
                "start_source": start_source,
                "end_source": end_source,
            },
            True,
        )

    if verdict.operation in {"sum", "average", "difference", "percentage_change"}:
        values, rejection = _validated_operands(verdict, context_labels)
        required = 1 if verdict.operation in {"sum", "average"} else 2
        if values is None or len(values) < required:
            if rejection is None:
                # The operands all parsed; there were simply too few of them for this
                # operation, which is the model under-selecting rather than mis-citing.
                rejection = {
                    "cause": "too_few_operands",
                    "supplied": len(values),
                    "required": required,
                }
            cause = rejection.pop("cause")
            return _refused(
                verdict.operation,
                cause,
                "numeric operands are invalid",
                **rejection,
                operands_supplied=_supplied_operands(verdict),
                labels_available=sorted(context_labels),
            )
        normalised = _normalise_numeric_units(values, verdict.result_unit)
        if normalised is None:
            return _refused(
                verdict.operation,
                "units_incompatible",
                "operand units do not match",
                units_supplied=[unit for _, _, unit, _ in values],
                result_unit=verdict.result_unit,
            )
        numbers, unit = normalised
        if verdict.operation == "sum":
            result = sum(numbers, Decimal(0))
            answer = f"{_format(result)}{_unit_suffix(unit)} total"
        elif verdict.operation == "average":
            result = sum(numbers, Decimal(0)) / Decimal(len(numbers))
            answer = f"{_format(result)}{_unit_suffix(unit)} average"
        elif verdict.operation == "difference":
            first, second = numbers[:2]
            if verdict.calculation_mode == "first_minus_second":
                result = first - second
            elif verdict.calculation_mode == "second_minus_first":
                result = second - first
            else:
                result = abs(second - first)
            answer = f"{_format(result)}{_unit_suffix(unit)} difference"
        else:
            baseline, comparison = numbers[:2]
            if baseline == 0:
                return _refused(
                    verdict.operation,
                    "percentage_baseline_zero",
                    "percentage baseline is zero",
                    operands_supplied=_supplied_operands(verdict),
                )
            result = (comparison - baseline) / abs(baseline) * Decimal(100)
            direction = "increase" if result > 0 else "decrease" if result < 0 else "change"
            answer = f"{_format(abs(result))}% {direction}"
        return ComputationResult(
            answer,
            {
                "operation": verdict.operation,
                "operands": [
                    {
                        "name": name,
                        "value": _format(number),
                        "unit": operand_unit,
                        "label": label,
                    }
                    for name, number, operand_unit, label in values
                ],
                "result": _format(result),
                "unit": "%" if verdict.operation == "percentage_change" else unit,
                "difference_direction": (
                    verdict.calculation_mode or "absolute"
                    if verdict.operation == "difference"
                    else None
                ),
            },
            True,
        )

    if verdict.operation == "comparison":
        if len(verdict.operand_values) != 2 or len(verdict.operand_labels) != 2:
            return _refused(
                "comparison",
                "comparison_operands_invalid",
                "",
                lengths=_array_lengths(verdict),
                operands_supplied=_supplied_operands(verdict),
            )
        start, end = (_date(value) for value in verdict.operand_values)
        start_source, end_source = verdict.operand_labels
        start_label = _label(start_source)
        end_label = _label(end_source)
        valid_start = start_label in context_labels or start_source == "QUESTION_DATE"
        valid_end = end_label in context_labels or end_source == "QUESTION_DATE"
        if start is not None and end is not None and valid_start and valid_end:
            return ComputationResult(
                "",
                {
                    "operation": "comparison",
                    "earlier": min(start, end).isoformat(),
                    "later": max(start, end).isoformat(),
                    "start_is_earlier": start <= end,
                },
                True,
            )
    return _refused(
        verdict.operation,
        # `comparison` reaches here when its two operands were present but unreadable or
        # uncited; every other operation reaches it because code performs no arithmetic
        # for it at all.
        "comparison_operands_invalid"
        if verdict.operation == "comparison"
        else "operation_not_computable",
        "",
        operands_supplied=_supplied_operands(verdict),
        labels_available=sorted(context_labels),
    )
