"""Judge/human agreement.

D1 rejects LoCoMo partly because its judge accepts a large share of wrong answers.
Using an LLM judge here without measuring it would reproduce that flaw — and the
free tier offers nothing stronger than the answerer to grade with (D12), so the
judge is weaker than one would choose. This makes the check load-bearing: it is
what lets a number in the README carry a stated reliability bound.

Workflow:

    chronomem eval label naive_rag --n 50     # writes a worksheet
    # fill in the `human` column by hand
    chronomem eval agreement naive_rag        # compares

The worksheet deliberately hides the judge's verdict. Seeing it first would anchor
the labeler, and an agreement number produced that way measures nothing.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

from .harness import QuestionResult

FIELDS = [
    "question_id",
    "question_type",
    "is_abstention",
    "question",
    "gold",
    "hypothesis",
    "human",
]


def write_worksheet(
    results: list[QuestionResult],
    questions: dict[str, str],
    path: str | Path,
    n: int = 50,
    seed: int = 0,
) -> Path:
    sample = list(results)
    if n < len(sample):
        random.Random(seed).shuffle(sample)
        sample = sample[:n]
    sample.sort(key=lambda r: r.question_id)

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for r in sample:
            writer.writerow(
                {
                    "question_id": r.question_id,
                    "question_type": r.question_type,
                    "is_abstention": int(r.is_abstention),
                    "question": questions.get(r.question_id, ""),
                    "gold": r.gold,
                    "hypothesis": r.hypothesis,
                    # Left blank on purpose — the judge's verdict is not shown.
                    "human": "",
                }
            )
    return dest


@dataclass(slots=True)
class Agreement:
    n: int
    agree: int
    judge_lenient: int
    """Judge said correct, human said wrong — the failure mode that inflates scores."""
    judge_strict: int
    """Judge said wrong, human said correct."""
    disagreements: list[tuple[str, bool, bool]]

    @property
    def rate(self) -> float:
        return self.agree / self.n if self.n else 0.0

    def summary(self) -> str:
        lines = [
            f"Judge/human agreement: **{self.rate:.1%}** (n={self.n})",
            f"- judge too lenient: {self.judge_lenient}",
            f"- judge too strict: {self.judge_strict}",
        ]
        if self.rate < 0.90:
            lines.append(
                "\n⚠️ Below 90%. Rework the judge prompt before publishing accuracy "
                "numbers graded by it."
            )
        return "\n".join(lines)


def _to_bool(raw: str) -> bool | None:
    v = raw.strip().lower()
    if v in {"1", "true", "t", "y", "yes", "correct"}:
        return True
    if v in {"0", "false", "f", "n", "no", "wrong", "incorrect"}:
        return False
    return None


def score_worksheet(path: str | Path, results: list[QuestionResult]) -> Agreement:
    verdicts = {r.question_id: r.correct for r in results}
    agree = lenient = strict = 0
    disagreements: list[tuple[str, bool, bool]] = []

    with Path(path).open() as fh:
        for row in csv.DictReader(fh):
            human = _to_bool(row.get("human", ""))
            if human is None:
                continue  # not labeled yet
            qid = row["question_id"]
            if qid not in verdicts:
                continue
            judge = verdicts[qid]
            if judge == human:
                agree += 1
            else:
                disagreements.append((qid, judge, human))
                if judge and not human:
                    lenient += 1
                else:
                    strict += 1

    n = agree + len(disagreements)
    return Agreement(n, agree, lenient, strict, disagreements)


def save_agreement(agreement: Agreement, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "n": agreement.n,
                "agreement_rate": agreement.rate,
                "judge_lenient": agreement.judge_lenient,
                "judge_strict": agreement.judge_strict,
                "disagreements": [
                    {"question_id": q, "judge": j, "human": h}
                    for q, j, h in agreement.disagreements
                ],
            },
            indent=2,
        )
    )
