"""Frozen question sets.

A manifest pins the exact question ids an experiment runs on, so that every row of
the results table is comparable by construction rather than by convention.

This exists because of a real defect. `lltm eval run --limit 31` does not
evaluate the first 31 questions of the 50-question dev set. Stratified samples do
nest — `stratify(pool, 31)` is a subset of `stratify(pool, 50)` — but the sample is
drawn per category and re-sorted by id, so it is scattered across all 50 positions
rather than being a prefix. Ingestion, meanwhile, processes namespaces in dataset
order, so a partial ingest leaves data for a *prefix*. On 2026-08-14 those two
orderings were assumed to agree: 11 of the requested 31 questions had no ingested
data, retrieved nothing, and were scored wrong. The headline number looked plausible
(35.5%) and was meaningless.

The lesson is not "be careful with --limit". It is that **a sample size is not an
experiment identity**. A manifest is the identity: a list of ids, plus enough
provenance to regenerate it.

Once a manifest is written it must not be regenerated with different parameters
under the same name. Freezing the question set is what makes a held-out run (A4)
mean anything — a test set that gets resampled until the number improves is a second
development set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Exposure = Literal["unseen", "development", "regression"]


@dataclass(frozen=True, slots=True)
class Manifest:
    name: str
    variant: str
    """LongMemEval split the ids belong to ('s', 'm', 'oracle')."""
    seed: int
    question_ids: tuple[str, ...]
    note: str = ""
    exposure: Exposure = "regression"
    """How this question set may be described now, not when it was first created.

    Legacy manifests default to regression because every LongMemEval question in this
    repository has been inspected. A new unseen claim therefore requires an explicit
    manifest field rather than inheriting a historical name such as `test100`.
    """

    def __len__(self) -> int:
        return len(self.question_ids)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "variant": self.variant,
            "seed": self.seed,
            "n": len(self.question_ids),
            "note": self.note,
            "exposure": self.exposure,
            "question_ids": list(self.question_ids),
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return p


def load_manifest(path: str | Path) -> Manifest:
    """Read a manifest.

    Accepts a plain newline-delimited id list as well, so an ad-hoc set can be
    promoted into a manifest without being rewritten first. `#` starts a comment.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")

    if p.suffix == ".json":
        data = json.loads(text)
        ids = tuple(data["question_ids"])
        if len(set(ids)) != len(ids):
            raise ValueError(f"{p} lists a question id more than once")
        return Manifest(
            name=data.get("name", p.stem),
            variant=data.get("variant", "s"),
            seed=int(data.get("seed", 0)),
            question_ids=ids,
            note=data.get("note", ""),
            exposure=data.get("exposure", "regression"),
        )

    ids = tuple(
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(set(ids)) != len(ids):
        raise ValueError(f"{p} lists a question id more than once")
    return Manifest(name=p.stem, variant="s", seed=0, question_ids=ids)


def require_claim(manifest: Manifest, claim: Exposure) -> None:
    """Refuse to promote an exposed set through a command-line label."""
    order = {"regression": 0, "development": 1, "unseen": 2}
    if order[claim] > order[manifest.exposure]:
        raise ValueError(
            f"manifest {manifest.name!r} is {manifest.exposure} evidence and cannot be "
            f"reported as {claim}; create and register a genuinely new manifest"
        )
