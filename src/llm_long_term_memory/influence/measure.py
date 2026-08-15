"""Which retrieved memories actually change the answer.

Retrieval reports relevance. What a packer needs is *utility*: whether putting this
memory in the prompt changes what the model says. The two diverge, and the gap is
measurable by ablation rather than argued about.

Two probes per memory, because one is not enough:

    leave-one-out   answer with the memory removed. A drop means it was carrying
                    something. No drop means either it was useless *or* another
                    memory carries the same evidence — LOO alone cannot tell those
                    apart, and calling the second case "inert" would be wrong.
    leave-one-in    answer with only that memory. High alone plus flat under LOO
                    identifies redundancy; low in both identifies genuine dead
                    weight.

Cost is the reason this is offline. Scoring k memories costs 2k+1 answers plus
judging, so a packer that ran it per query would spend forty times the inference it
saves. The output here is a *training set*: (query, memory) → utility, which a cheap
predictor learns to approximate at serving time.

Both directions matter. Ablating only the questions the system already answers
correctly measures necessity and misses the failure this project actually has —
a stale or wrong memory that *causes* the error, and whose removal fixes it. Those
only appear among the questions currently answered wrong.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path


class Influence(StrEnum):
    """Four outcomes, not two.

    `INERT` and `REDUNDANT` are separated deliberately: both look identical to
    leave-one-out, and treating a redundant memory as useless would drop evidence
    the moment its twin is evicted.
    """

    CRITICAL = "critical"
    """Removing it breaks a correct answer."""
    DROWNED = "drowned"
    """Answers correctly on its own, but the full context answers wrongly.

    Not a healthy label despite the positive utility: it says the evidence was
    present and something else in the prompt overrode it. These rows are the
    strongest available signal that the *other* memories are interfering, which is
    why they are named for the symptom rather than filed under "helpful"."""
    REDUNDANT = "redundant"
    """Sufficient alone but not missed when removed. Its evidence exists elsewhere."""
    INERT = "inert"
    """Neither missed when removed nor sufficient alone."""
    HARMFUL = "harmful"
    """Removing it *fixes* a wrong answer. The stale-memory case."""


@dataclass(slots=True)
class MemoryInfluence:
    question_id: str
    memory_id: str
    full_correct: bool
    without_correct: bool
    alone_correct: bool | None = None

    @property
    def label(self) -> Influence:
        if self.full_correct and not self.without_correct:
            return Influence.CRITICAL
        if not self.full_correct and self.without_correct:
            return Influence.HARMFUL
        if self.alone_correct:
            # Survives removal but answers on its own: the evidence is duplicated.
            return Influence.REDUNDANT if self.full_correct else Influence.DROWNED
        return Influence.INERT

    @property
    def utility(self) -> float:
        """A scalar for the predictor to regress on.

        Signed, so a packer trained on it learns to exclude harmful memories rather
        than merely to rank them last.
        """
        return {
            Influence.CRITICAL: 1.0,
            Influence.DROWNED: 0.6,
            Influence.REDUNDANT: 0.3,
            Influence.INERT: 0.0,
            Influence.HARMFUL: -1.0,
        }[self.label]


@dataclass
class InfluenceDataset:
    rows: list[MemoryInfluence] = field(default_factory=list)

    def distribution(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.rows:
            out[r.label.value] = out.get(r.label.value, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    @property
    def load_bearing_rate(self) -> float:
        """Share that changed the answer either way. The headline number: how much
        of what retrieval returned was doing anything at all."""
        if not self.rows:
            return 0.0
        moved = sum(1 for r in self.rows if r.label in (Influence.CRITICAL, Influence.HARMFUL))
        return moved / len(self.rows)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for r in self.rows:
                fh.write(
                    json.dumps({**asdict(r), "label": r.label.value, "utility": r.utility}) + "\n"
                )

    @classmethod
    def load(cls, path: str | Path) -> InfluenceDataset:
        rows = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            rows.append(
                MemoryInfluence(
                    question_id=d["question_id"],
                    memory_id=d["memory_id"],
                    full_correct=d["full_correct"],
                    without_correct=d["without_correct"],
                    alone_correct=d.get("alone_correct"),
                )
            )
        return cls(rows)


def requests_needed(n_questions: int, top_k: int, leave_one_in: bool = True) -> int:
    """Answer calls for a full measurement — judging doubles it.

    Exposed so the cost is visible before it is spent: 50 questions at k=20 is
    2,050 answers with both probes, four days of a 500/day budget.
    """
    per_question = 1 + top_k * (2 if leave_one_in else 1)
    return n_questions * per_question
