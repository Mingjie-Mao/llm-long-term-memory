"""Create or verify the fixed train-only v3 reasoning pilot manifest."""

from __future__ import annotations

import json
import random
from pathlib import Path

from llm_long_term_memory.evaluation.datasets import longmemeval as lme
from llm_long_term_memory.evaluation.manifest import Manifest, load_manifest

REPO = Path(__file__).resolve().parent.parent
TRAIN = REPO / "results/manifests/train150.json"
DESTINATION = REPO / "results/manifests/v3-reasoning48.json"
SEED = 20260903
QUOTAS = {
    "temporal-reasoning": 12,
    "multi-session": 12,
    "knowledge-update": 8,
    "single-session-user": 6,
    "single-session-assistant": 6,
    "single-session-preference": 4,
}


def select_ids(instances, train_ids: tuple[str, ...], seed: int = SEED) -> tuple[str, ...]:
    by_id = {instance.question_id: instance for instance in instances}
    missing = set(train_ids) - set(by_id)
    if missing:
        raise ValueError(f"train manifest has {len(missing)} unknown question(s)")
    by_type: dict[str, list[str]] = {}
    for question_id in train_ids:
        question_type = by_id[question_id].question_type
        by_type.setdefault(question_type, []).append(question_id)

    selected: list[str] = []
    for question_type, count in QUOTAS.items():
        candidates = sorted(by_type.get(question_type, []))
        if len(candidates) < count:
            raise ValueError(
                f"train150 has only {len(candidates)} {question_type!r} questions; need {count}"
            )
        random.Random(f"{seed}:{question_type}").shuffle(candidates)
        selected.extend(candidates[:count])
    return tuple(sorted(selected))


def expected_manifest() -> Manifest:
    train = load_manifest(TRAIN)
    instances = lme.load(train.variant, REPO / "data")
    return Manifest(
        name="v3-reasoning48",
        variant=train.variant,
        seed=SEED,
        question_ids=select_ids(instances, train.question_ids),
        note=(
            "Train150-only stratified v3 answer pilot: 12 temporal, 12 multi-session, "
            "8 knowledge-update, 6 user, 6 assistant, 4 preference. Selected without "
            "using answers, correctness or test100."
        ),
    )


def main() -> int:
    expected = expected_manifest()
    if DESTINATION.exists():
        actual = json.loads(DESTINATION.read_text(encoding="utf-8"))
        if actual != expected.to_dict():
            print(f"STOP: existing {DESTINATION.relative_to(REPO)} differs from fixed selection")
            return 2
        print(f"PASS: unchanged {expected.name} · {len(expected)} train-only questions")
        return 0
    expected.save(DESTINATION)
    print(f"CREATED: {DESTINATION.relative_to(REPO)} · {len(expected)} train-only questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
