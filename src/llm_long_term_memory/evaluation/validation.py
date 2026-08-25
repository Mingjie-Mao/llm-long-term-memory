"""Aggregate-only reporting for validation and final-test runs.

``dev100`` is validation data, not a second development set.  Its raw JSONL files
must exist so interrupted runs can resume, but the normal report must not reveal a
question id, answer, hypothesis, or judge reason.  This module deliberately accepts
those rows and returns only counts and rates.

The failure buckets are observations about *where the gold source session was
lost*, not claims that the exact answer-bearing fact survived extraction.  A row in
``wrong_despite_gold_session_context`` can still be an extraction loss; the label
only says that at least one memory from the right conversation reached the model.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median, stdev
from typing import Any

from .compare import _binom_two_sided
from .harness import QuestionResult, RunReport
from .reproducibility import sha256_file

RECALL_STAGES = ("candidates", "ranked", "selected")


class ValidationArtifactError(ValueError):
    """A repeat is incomplete, duplicated, or does not match the frozen manifest."""


@dataclass(slots=True)
class UsageAggregate:
    requests: int = 0
    failed_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    by_role: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ArmAggregate:
    arm: str
    questions: int
    runs: int
    run_accuracies: list[float]
    mean_accuracy: float
    stdev_accuracy: float
    spread_accuracy: float
    majority_accuracy: float
    unanimous_agreement: float
    median_context_tokens: float
    recall_by_stage: dict[str, float | None]
    majority_accuracy_by_type: dict[str, float]
    majority_failure_types: dict[str, int]
    fallback_trigger_rate: float
    correct_with_raw_fallback_rate: float
    usage: UsageAggregate

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PairedMajority:
    baseline: str
    candidate: str
    questions: int
    both_right: int
    both_wrong: int
    candidate_wins: int
    candidate_losses: int
    accuracy_delta: float
    p_value: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DevCandidateDecision:
    selected_arm: str
    selected_variant: str
    diagnostic: str
    baseline_majority_accuracy: float
    coherent_majority_accuracy: float
    oracle_majority_accuracy: float
    coherent_context_ratio: float
    accuracy_gate_passed: bool
    context_gate_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_dev_candidate(arms: list[ArmAggregate]) -> DevCandidateDecision:
    """Apply the dev100 rule recorded before any validation result was viewed."""
    by_name = {arm.arm: arm for arm in arms}
    expected = {"flat20", "coherent-auto", "coherent-oracle"}
    if set(by_name) != expected or len(arms) != len(expected):
        raise ValidationArtifactError(
            "dev decision requires exactly flat20, coherent-auto and coherent-oracle"
        )
    if any(arm.questions != 100 or arm.runs != 3 for arm in arms):
        raise ValidationArtifactError(
            "dev decision requires three complete runs over exactly 100 questions per arm"
        )

    baseline = by_name["flat20"]
    coherent = by_name["coherent-auto"]
    oracle = by_name["coherent-oracle"]
    if baseline.median_context_tokens == 0:
        context_ratio = 1.0 if coherent.median_context_tokens == 0 else float("inf")
    else:
        context_ratio = coherent.median_context_tokens / baseline.median_context_tokens
    accuracy_ok = coherent.majority_accuracy >= baseline.majority_accuracy
    context_ok = context_ratio <= 1.50
    if accuracy_ok and context_ok:
        selected_arm = "coherent-auto"
        selected_variant = "two_stage_coherent"
        diagnostic = "adopted_registered_coherent_candidate"
    else:
        selected_arm = "flat20"
        selected_variant = "two_stage_fallback"
        if not accuracy_ok and not context_ok:
            diagnostic = "accuracy_and_context_gates_failed"
        elif not context_ok:
            diagnostic = "context_budget_gate_failed"
        elif oracle.majority_accuracy >= baseline.majority_accuracy:
            diagnostic = "session_selection_gap"
        else:
            diagnostic = "coherent_shape_not_supported"
    return DevCandidateDecision(
        selected_arm=selected_arm,
        selected_variant=selected_variant,
        diagnostic=diagnostic,
        baseline_majority_accuracy=baseline.majority_accuracy,
        coherent_majority_accuracy=coherent.majority_accuracy,
        oracle_majority_accuracy=oracle.majority_accuracy,
        coherent_context_ratio=context_ratio,
        accuracy_gate_passed=accuracy_ok,
        context_gate_passed=context_ok,
    )


def bind_artifacts(paths: list[Path]) -> dict[str, dict[str, int | str]]:
    """Hash sealed result files by basename without exposing benchmark rows."""
    names = [path.name for path in paths]
    if len(names) != len(set(names)):
        raise ValidationArtifactError("sealed artifact basenames are not unique")
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise ValidationArtifactError(f"{len(missing)} sealed artifact(s) are missing")
    return {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in paths
    }


def validate_bound_artifacts(
    aggregate: dict[str, Any],
    artifact_dir: Path,
    expected_names: set[str],
) -> None:
    """Prove the aggregate still corresponds to every sealed source artifact."""
    bound = aggregate.get("sealed_artifacts")
    if not isinstance(bound, dict) or set(bound) != expected_names:
        raise ValidationArtifactError("aggregate has an incomplete sealed-artifact inventory")
    for name, record in bound.items():
        if Path(name).name != name or not isinstance(record, dict):
            raise ValidationArtifactError("aggregate contains an invalid sealed-artifact entry")
        path = artifact_dir / name
        if (
            not path.exists()
            or record.get("sha256") != sha256_file(path)
            or record.get("bytes") != path.stat().st_size
        ):
            raise ValidationArtifactError(f"sealed artifact changed after aggregation: {name}")


def load_dev_decision(decision_path: str | Path, aggregate_path: str | Path) -> dict[str, Any]:
    """Validate the dev decision and bind it to the aggregate that produced it."""
    decision_file = Path(decision_path)
    aggregate_file = Path(aggregate_path)
    try:
        decision = json.loads(decision_file.read_text(encoding="utf-8"))
        aggregate = json.loads(aggregate_file.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise ValidationArtifactError(f"invalid dev100 decision artifact: {exc}") from exc
    if not isinstance(decision, dict) or decision.get("schema_version") != 2:
        raise ValidationArtifactError("dev100 decision has an unsupported schema")
    if decision.get("aggregate_sha256") != sha256_file(aggregate_file):
        raise ValidationArtifactError("dev100 decision does not match the aggregate report hash")
    selected_variant = decision.get("selected_variant")
    selected_arm = decision.get("selected_arm")
    expected_pair = {
        "coherent-auto": "two_stage_coherent",
        "flat20": "two_stage_fallback",
    }
    if expected_pair.get(str(selected_arm)) != selected_variant:
        raise ValidationArtifactError("dev100 decision contains an invalid arm/variant selection")
    embedded = aggregate.get("registered_decision")
    try:
        arm_rows = aggregate["arms"]
        if not isinstance(arm_rows, list):
            raise TypeError("arms is not a list")
        recomputed = select_dev_candidate(
            [ArmAggregate(**row) for row in arm_rows if isinstance(row, dict)]
        ).to_dict()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationArtifactError(f"cannot recompute the dev100 decision: {exc}") from exc
    fields = DevCandidateDecision.__dataclass_fields__
    if embedded != recomputed or any(decision.get(key) != recomputed[key] for key in fields):
        raise ValidationArtifactError("dev100 decision disagrees with its aggregate report")
    expected_artifacts = {
        filename
        for name in ("flat20", "coherent-auto", "coherent-oracle")
        for number in range(1, 4)
        for filename in (f"{name}.rep{number}.jsonl", f"{name}.rep{number}.usage.json")
    }
    validate_bound_artifacts(
        aggregate,
        aggregate_file.parent.parent / "sealed" / "dev100",
        expected_artifacts,
    )
    return decision


def _rows_by_question(
    reports: list[RunReport], expected_ids: set[str]
) -> dict[str, list[QuestionResult]]:
    if not reports:
        raise ValidationArtifactError("at least one complete repeat is required")
    if len(reports) % 2 == 0:
        raise ValidationArtifactError("an odd number of repeats is required for majority voting")

    grouped = {qid: [] for qid in expected_ids}
    for number, report in enumerate(reports, 1):
        ids = [row.question_id for row in report.results]
        duplicates = len(ids) - len(set(ids))
        missing = expected_ids - set(ids)
        extra = set(ids) - expected_ids
        if duplicates or missing or extra:
            # Counts are safe to show.  Identifiers are intentionally omitted: this
            # exception is part of the aggregate-only validation interface.
            raise ValidationArtifactError(
                f"repeat {number} does not match the frozen manifest: "
                f"{duplicates} duplicate, {len(missing)} missing, {len(extra)} extra"
            )
        for row in report.results:
            grouped[row.question_id].append(row)
    return grouped


def _majority(values: list[bool]) -> bool:
    return sum(values) > len(values) // 2


def _stage(row: QuestionResult, name: str) -> bool | None:
    stages = row.notes.get("recall_stages")
    if not isinstance(stages, dict):
        return None
    value = stages.get(name)
    return value if isinstance(value, bool) else None


def _used_fallback(row: QuestionResult) -> bool:
    return str(row.notes.get("fallback_level", "none")) != "none"


def _failure_bucket(rows: list[QuestionResult]) -> str:
    """One non-overlapping, evidence-backed stage label for a majority-wrong item."""
    stages: dict[str, bool | None] = {}
    for name in RECALL_STAGES:
        values = [value for row in rows if (value := _stage(row, name)) is not None]
        stages[name] = _majority(values) if values else None

    if stages["candidates"] is False:
        return "no_gold_session_in_candidates"
    if stages["candidates"] is True and stages["ranked"] is False:
        return "gold_session_lost_before_top_k"
    if stages["ranked"] is True and stages["selected"] is False:
        return "gold_session_lost_in_context"
    if _majority([_used_fallback(row) for row in rows]):
        return "wrong_after_raw_fallback"
    if stages["selected"] is True:
        return "wrong_despite_gold_session_context"
    return "stage_trace_unavailable"


def load_usage(path: str | Path) -> UsageAggregate:
    """Read call records rather than trusting a separately cached summary."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    calls = payload.get("calls")
    if not isinstance(calls, list):
        raise ValidationArtifactError("usage artifact has no call-record list")

    aggregate = UsageAggregate()
    roles: dict[str, Counter] = {}
    for call in calls:
        if not isinstance(call, dict):
            raise ValidationArtifactError("usage artifact contains a non-object call")
        role = str(call.get("role", "unknown"))
        input_tokens = int(call.get("input_tokens", 0))
        output_tokens = int(call.get("output_tokens", 0))
        failed = not bool(call.get("ok", True))
        aggregate.requests += 1
        aggregate.failed_requests += int(failed)
        aggregate.input_tokens += input_tokens
        aggregate.output_tokens += output_tokens
        counter = roles.setdefault(role, Counter())
        counter["requests"] += 1
        counter["failed_requests"] += int(failed)
        counter["input_tokens"] += input_tokens
        counter["output_tokens"] += output_tokens
    aggregate.by_role = {role: dict(counts) for role, counts in sorted(roles.items())}
    return aggregate


def merge_usage(items: list[UsageAggregate]) -> UsageAggregate:
    merged = UsageAggregate()
    roles: dict[str, Counter] = {}
    for item in items:
        merged.requests += item.requests
        merged.failed_requests += item.failed_requests
        merged.input_tokens += item.input_tokens
        merged.output_tokens += item.output_tokens
        for role, counts in item.by_role.items():
            roles.setdefault(role, Counter()).update(counts)
    merged.by_role = {role: dict(counts) for role, counts in sorted(roles.items())}
    return merged


def aggregate_arm(
    arm: str,
    reports: list[RunReport],
    expected_ids: set[str],
    usage: list[UsageAggregate] | None = None,
) -> ArmAggregate:
    """Summarize complete repeats without returning any per-question material."""
    grouped = _rows_by_question(reports, expected_ids)
    run_accuracies = [report.accuracy for report in reports]
    majority_correct = {
        qid: _majority([row.correct for row in rows]) for qid, rows in grouped.items()
    }
    unanimous = sum(len({row.correct for row in rows}) == 1 for rows in grouped.values())

    by_type: dict[str, list[bool]] = {}
    failures: Counter[str] = Counter()
    for qid, rows in grouped.items():
        question_types = {row.question_type for row in rows}
        if len(question_types) != 1:
            raise ValidationArtifactError("a question changed type between repeats")
        qtype = next(iter(question_types))
        by_type.setdefault(qtype, []).append(majority_correct[qid])
        if not majority_correct[qid]:
            failures[_failure_bucket(rows)] += 1

    stage_rates: dict[str, float | None] = {}
    all_rows = [row for report in reports for row in report.results]
    for name in RECALL_STAGES:
        values = [value for row in all_rows if (value := _stage(row, name)) is not None]
        stage_rates[name] = sum(values) / len(values) if values else None

    context = [row.context_tokens for row in all_rows]
    fallback = [_used_fallback(row) for row in all_rows]
    rescued = [row.correct and used for row, used in zip(all_rows, fallback, strict=True)]
    mean = sum(run_accuracies) / len(run_accuracies)
    return ArmAggregate(
        arm=arm,
        questions=len(expected_ids),
        runs=len(reports),
        run_accuracies=run_accuracies,
        mean_accuracy=mean,
        stdev_accuracy=stdev(run_accuracies) if len(run_accuracies) > 1 else 0.0,
        spread_accuracy=max(run_accuracies) - min(run_accuracies),
        majority_accuracy=sum(majority_correct.values()) / len(expected_ids),
        unanimous_agreement=unanimous / len(expected_ids),
        median_context_tokens=median(context),
        recall_by_stage=stage_rates,
        majority_accuracy_by_type={
            qtype: sum(values) / len(values) for qtype, values in sorted(by_type.items())
        },
        majority_failure_types=dict(sorted(failures.items())),
        fallback_trigger_rate=sum(fallback) / len(fallback),
        correct_with_raw_fallback_rate=sum(rescued) / len(rescued),
        usage=merge_usage(usage or []),
    )


def paired_majority(
    baseline_name: str,
    baseline_reports: list[RunReport],
    candidate_name: str,
    candidate_reports: list[RunReport],
    expected_ids: set[str],
) -> PairedMajority:
    """Exact McNemar comparison over majority-of-repeats outcomes, without ids."""
    baseline = _rows_by_question(baseline_reports, expected_ids)
    candidate = _rows_by_question(candidate_reports, expected_ids)
    both_right = both_wrong = wins = losses = 0
    for qid in expected_ids:
        left = _majority([row.correct for row in baseline[qid]])
        right = _majority([row.correct for row in candidate[qid]])
        if left and right:
            both_right += 1
        elif not left and not right:
            both_wrong += 1
        elif right:
            wins += 1
        else:
            losses += 1
    return PairedMajority(
        baseline=baseline_name,
        candidate=candidate_name,
        questions=len(expected_ids),
        both_right=both_right,
        both_wrong=both_wrong,
        candidate_wins=wins,
        candidate_losses=losses,
        accuracy_delta=(wins - losses) / len(expected_ids),
        p_value=_binom_two_sided(wins, wins + losses),
    )


def single_run_arm_dict(arm: ArmAggregate) -> dict[str, Any]:
    """Remove repeat-only fields from a formal one-shot result."""
    if arm.runs != 1 or len(arm.run_accuracies) != 1:
        raise ValidationArtifactError("a final-test arm must contain exactly one run")
    return {
        "arm": arm.arm,
        "questions": arm.questions,
        "accuracy": arm.run_accuracies[0],
        "median_context_tokens": arm.median_context_tokens,
        "recall_by_stage": arm.recall_by_stage,
        "accuracy_by_type": arm.majority_accuracy_by_type,
        "failure_types": arm.majority_failure_types,
        "fallback_trigger_rate": arm.fallback_trigger_rate,
        "correct_with_raw_fallback_rate": arm.correct_with_raw_fallback_rate,
        "usage": arm.usage.to_dict(),
    }


def render_final_markdown(
    arms: list[ArmAggregate],
    comparisons: list[PairedMajority],
    shared_ingestion: UsageAggregate,
) -> str:
    """Render one-shot test results without fake repeat/agreement statistics."""
    rows = [single_run_arm_dict(arm) for arm in arms]
    lines = [
        "# Final one-shot test report",
        "",
        "Each frozen arm ran exactly once. No repeat variance or agreement statistic exists.",
        "No question ids, answers, model answers, or judge reasons are included.",
        "",
        "| arm | accuracy | candidate/ranked/selected recall | median context | "
        "requests | tokens |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        stage_values = row["recall_by_stage"]
        recalls = "/".join(
            "—" if stage_values.get(name) is None else f"{stage_values[name]:.1%}"
            for name in RECALL_STAGES
        )
        usage = row["usage"]
        lines.append(
            f"| `{row['arm']}` | {row['accuracy']:.1%} | {recalls} | "
            f"{row['median_context_tokens']:,.0f} | {usage['requests']:,} | "
            f"{usage['input_tokens'] + usage['output_tokens']:,} |"
        )

    evaluation_usage = merge_usage([arm.usage for arm in arms])
    total_usage = merge_usage([shared_ingestion, evaluation_usage])
    lines.extend(
        [
            "",
            "## API usage (cost evidence)",
            "",
            "| phase | requests | failed requests | input tokens | output tokens | total tokens |",
            "|---|---:|---:|---:|---:|---:|",
            f"| shared ingestion | {shared_ingestion.requests:,} | "
            f"{shared_ingestion.failed_requests:,} | {shared_ingestion.input_tokens:,} | "
            f"{shared_ingestion.output_tokens:,} | {shared_ingestion.total_tokens:,} |",
            f"| answer + judge arms | {evaluation_usage.requests:,} | "
            f"{evaluation_usage.failed_requests:,} | {evaluation_usage.input_tokens:,} | "
            f"{evaluation_usage.output_tokens:,} | {evaluation_usage.total_tokens:,} |",
            f"| **total** | **{total_usage.requests:,}** | "
            f"**{total_usage.failed_requests:,}** | **{total_usage.input_tokens:,}** | "
            f"**{total_usage.output_tokens:,}** | **{total_usage.total_tokens:,}** |",
            "",
            "> Requests and tokens are auditable cost evidence. Provider billing is reported "
            "separately when available; no dollar amount is guessed.",
            "",
            "## Paired single-run comparisons",
            "",
            "| baseline → v2 | wins-losses | accuracy delta | exact p |",
            "|---|---:|---:|---:|",
        ]
    )
    lines.extend(
        f"| `{item.baseline}` → `{item.candidate}` | "
        f"{item.candidate_wins}-{item.candidate_losses} | "
        f"{item.accuracy_delta:+.1%} | {item.p_value:.4f} |"
        for item in comparisons
    )
    for row in rows:
        lines.extend(
            [
                "",
                f"## `{row['arm']}` failure stages",
                "",
                f"Raw fallback fired on {row['fallback_trigger_rate']:.1%}; "
                f"{row['correct_with_raw_fallback_rate']:.1%} of outcomes were correct after it.",
                "",
                "| question type | accuracy |",
                "|---|---:|",
                *(
                    f"| `{question_type}` | {accuracy:.1%} |"
                    for question_type, accuracy in row["accuracy_by_type"].items()
                ),
                "",
            ]
        )
        failures = row["failure_types"]
        if failures:
            lines.extend(["| observed stage | questions |", "|---|---:|"])
            lines.extend(f"| `{name}` | {count} |" for name, count in failures.items())
        else:
            lines.append("No failures.")
    lines.extend(
        [
            "",
            "> Failure stages track whether the source session survived. They do not prove "
            "that the exact answer fact survived extraction.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_markdown(
    arms: list[ArmAggregate],
    comparisons: list[PairedMajority],
    shared_ingestion: UsageAggregate | None = None,
) -> str:
    """A compact aggregate report suitable for dev100 without opening raw rows."""
    lines = [
        "# Aggregate validation report",
        "",
        "No question ids, answers, model answers, or judge reasons are included.",
        "",
        "| arm | repeats | accuracy (runs) | mean ± sd | majority | agreement | "
        "candidate/ranked/selected recall | median context | requests | tokens |",
        "|---|---:|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for arm in arms:
        runs = ", ".join(f"{value:.1%}" for value in arm.run_accuracies)
        recalls = "/".join(
            "—" if arm.recall_by_stage[name] is None else f"{arm.recall_by_stage[name]:.1%}"
            for name in RECALL_STAGES
        )
        lines.append(
            f"| `{arm.arm}` | {arm.runs} | {runs} | {arm.mean_accuracy:.1%} ± "
            f"{arm.stdev_accuracy:.1%} | {arm.majority_accuracy:.1%} | "
            f"{arm.unanimous_agreement:.1%} | {recalls} | "
            f"{arm.median_context_tokens:,.0f} | {arm.usage.requests:,} | "
            f"{arm.usage.total_tokens:,} |"
        )

    if shared_ingestion is not None:
        evaluation_usage = merge_usage([arm.usage for arm in arms])
        total_usage = merge_usage([shared_ingestion, evaluation_usage])
        lines.extend(
            [
                "",
                "## API usage (cost evidence)",
                "",
                "| phase | requests | failed requests | input tokens | output tokens | "
                "total tokens |",
                "|---|---:|---:|---:|---:|---:|",
                f"| shared ingestion | {shared_ingestion.requests:,} | "
                f"{shared_ingestion.failed_requests:,} | {shared_ingestion.input_tokens:,} | "
                f"{shared_ingestion.output_tokens:,} | {shared_ingestion.total_tokens:,} |",
                f"| answer + judge arms | {evaluation_usage.requests:,} | "
                f"{evaluation_usage.failed_requests:,} | {evaluation_usage.input_tokens:,} | "
                f"{evaluation_usage.output_tokens:,} | {evaluation_usage.total_tokens:,} |",
                f"| **total** | **{total_usage.requests:,}** | "
                f"**{total_usage.failed_requests:,}** | **{total_usage.input_tokens:,}** | "
                f"**{total_usage.output_tokens:,}** | **{total_usage.total_tokens:,}** |",
                "",
                (
                    "> Usage artifacts prove requests and tokens. They do not contain provider "
                    "billing; a dollar amount must come from the provider's billing export rather "
                    "than be guessed."
                ),
            ]
        )

    lines.extend(["", "## Majority-vote paired comparisons", ""])
    if not comparisons:
        lines.append("No paired comparison requested.")
    else:
        lines.extend(
            [
                "| baseline → candidate | wins-losses | accuracy delta | exact p |",
                "|---|---:|---:|---:|",
            ]
        )
        for item in comparisons:
            lines.append(
                f"| `{item.baseline}` → `{item.candidate}` | "
                f"{item.candidate_wins}-{item.candidate_losses} | "
                f"{item.accuracy_delta:+.1%} | {item.p_value:.4f} |"
            )

    for arm in arms:
        lines.extend(["", f"## `{arm.arm}` failure stages", ""])
        lines.append(
            f"Raw fallback fired on {arm.fallback_trigger_rate:.1%} of runs; "
            f"{arm.correct_with_raw_fallback_rate:.1%} of all run outcomes were correct "
            "after raw fallback."
        )
        lines.extend(
            [
                "",
                "| question type | majority accuracy |",
                "|---|---:|",
                *(
                    f"| `{question_type}` | {accuracy:.1%} |"
                    for question_type, accuracy in arm.majority_accuracy_by_type.items()
                ),
                "",
            ]
        )
        if not arm.majority_failure_types:
            lines.append("No majority-vote failures.")
        else:
            lines.extend(["| observed stage | questions |", "|---|---:|"])
            lines.extend(
                f"| `{name}` | {count} |" for name, count in arm.majority_failure_types.items()
            )
    lines.extend(
        [
            "",
            "> These are source-session stage labels, not proof that the exact answer fact "
            "survived extraction. `wrong_despite_gold_session_context` therefore combines "
            "possible extraction, reasoning, and judge errors.",
        ]
    )
    return "\n".join(lines) + "\n"
