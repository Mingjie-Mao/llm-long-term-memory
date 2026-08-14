"""Running the ablation that produces utility labels.

The measurement `measure.py` describes, actually executed: for each question, answer
once with everything retrieved, then once per memory with it removed, then once per
memory with it alone. Each answer is judged, and the three verdicts give the label.

Checkpointed per question, because the cost makes this a multi-day job on a
request-capped quota — `requests_needed(50, 20)` is 2,050 answers and as many judge
calls. Resuming is the normal path, not the error path.

Features are captured at the same moment as the labels and written alongside them.
They have to come from the same retrieval that produced the ablation, or the
predictor would be fitted against features describing a different context than the
one that was measured.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chronomem.evaluation.datasets.longmemeval import Instance
from chronomem.evaluation.judge import Judge
from chronomem.llm.client import DailyQuotaExhausted
from chronomem.store import Memory

from .features import FEATURE_NAMES, build
from .measure import InfluenceDataset, MemoryInfluence


@dataclass(slots=True)
class MeasurementOutcome:
    dataset: InfluenceDataset
    features: np.ndarray
    completed: bool
    stopped_reason: str | None = None
    questions_done: int = 0


def _answer_and_judge(answer_fn, judge: Judge, instance: Instance, memories: list[Memory]) -> bool:
    text = answer_fn(instance, memories)
    verdict = judge.grade(
        question=instance.question,
        gold=instance.answer,
        hypothesis=text,
        is_abstention=instance.is_abstention,
    )
    return verdict.correct


def measure_influence(
    instances: list[Instance],
    retrieve_fn,
    answer_fn,
    judge: Judge,
    out_path: str | Path,
    *,
    leave_one_in: bool = True,
    resume: bool = True,
    on_question=None,
) -> MeasurementOutcome:
    """`retrieve_fn(instance) -> [(memory, score)]`, `answer_fn(instance, memories) -> str`.

    Injected rather than constructed here so the same measurement can be run against
    any retrieval configuration without this module knowing about the store.
    """
    path = Path(out_path)
    features_path = path.with_suffix(".features.npy")
    header_path = features_path.with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[MemoryInfluence] = []
    feature_rows: list[list[float]] = []
    done_questions: set[str] = set()

    if resume and path.exists() and features_path.exists() and header_path.exists():
        existing = InfluenceDataset.load(path)
        try:
            loaded = np.load(features_path)
            header = json.loads(header_path.read_text())
            completed = set(header.get("completed_questions", []))
            if (
                tuple(header.get("features", ())) == FEATURE_NAMES
                and len(loaded) == len(existing.rows)
                and all(isinstance(question_id, str) for question_id in completed)
            ):
                committed = [
                    (row, features)
                    for row, features in zip(existing.rows, loaded.tolist(), strict=True)
                    if row.question_id in completed
                ]
                rows = [row for row, _ in committed]
                feature_rows = [features for _, features in committed]
                done_questions = completed
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    def flush() -> None:
        InfluenceDataset(rows=rows).save(path)
        np.save(features_path, np.array(feature_rows, dtype=float).reshape(-1, len(FEATURE_NAMES)))
        write_feature_header(header_path, done_questions)

    for instance in instances:
        if instance.question_id in done_questions:
            continue
        try:
            retrieved = retrieve_fn(instance)
            if not retrieved:
                done_questions.add(instance.question_id)
                flush()
                if on_question:
                    on_question(instance, InfluenceDataset(rows=rows))
                continue
            memories = [m for m, _ in retrieved]

            full_correct = _answer_and_judge(answer_fn, judge, instance, memories)
            question_rows: list[MemoryInfluence] = []
            question_features: list[list[float]] = []

            for rank, (memory, score) in enumerate(retrieved):
                without = [m for m in memories if m.id != memory.id]
                without_correct = _answer_and_judge(answer_fn, judge, instance, without)

                alone_correct = None
                if leave_one_in:
                    alone_correct = _answer_and_judge(answer_fn, judge, instance, [memory])

                question_rows.append(
                    MemoryInfluence(
                        question_id=instance.question_id,
                        memory_id=memory.id,
                        full_correct=full_correct,
                        without_correct=without_correct,
                        alone_correct=alone_correct,
                    )
                )
                question_features.append(
                    build(
                        memory,
                        query=instance.question,
                        retrieval_score=score,
                        rank=rank,
                        neighbours=memories,
                    ).as_list()
                )
            rows.extend(question_rows)
            feature_rows.extend(question_features)
        except DailyQuotaExhausted as exc:
            flush()
            return MeasurementOutcome(
                InfluenceDataset(rows=rows),
                np.array(feature_rows, dtype=float).reshape(-1, len(FEATURE_NAMES)),
                completed=False,
                stopped_reason=str(exc),
                questions_done=len(done_questions),
            )
        except Exception as exc:  # any failure is a stop, not a loss
            flush()
            return MeasurementOutcome(
                InfluenceDataset(rows=rows),
                np.array(feature_rows, dtype=float).reshape(-1, len(FEATURE_NAMES)),
                completed=False,
                stopped_reason=f"{type(exc).__name__}: {exc}",
                questions_done=len(done_questions),
            )

        done_questions.add(instance.question_id)
        flush()
        if on_question:
            on_question(instance, InfluenceDataset(rows=rows))

    flush()
    return MeasurementOutcome(
        InfluenceDataset(rows=rows),
        np.array(feature_rows, dtype=float).reshape(-1, len(FEATURE_NAMES)),
        completed=True,
        questions_done=len(done_questions),
    )


def write_feature_header(path: str | Path, completed_questions: set[str]) -> None:
    """Record the column order beside the array.

    `fit_grouped` reads the retrieval score from column 0 by convention. Persisting
    the names makes that checkable rather than assumed.
    """
    Path(path).write_text(
        json.dumps(
            {
                "features": list(FEATURE_NAMES),
                "completed_questions": sorted(completed_questions),
            }
        )
    )
