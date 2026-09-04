"""Create the disjoint tune42 and sealed dev60 manifests for v3 reasoning work."""

from __future__ import annotations

import json
import random
from pathlib import Path

from llm_long_term_memory.evaluation.datasets import longmemeval as lme
from llm_long_term_memory.evaluation.manifest import Manifest, load_manifest

REPO = Path(__file__).resolve().parent.parent
TRAIN = REPO / "results/manifests/train150.json"
PILOT = REPO / "results/manifests/v3-reasoning48.json"
TUNE_DESTINATION = REPO / "results/manifests/v3-reasoning-tune42.json"
DEV_DESTINATION = REPO / "results/manifests/v3-reasoning-dev60.json"
SEED = 20260904
DEV_QUOTAS = {
    "temporal-reasoning": 18,
    "multi-session": 18,
    "knowledge-update": 10,
    "single-session-user": 5,
    "single-session-assistant": 5,
    "single-session-preference": 4,
}


def split_ids(
    instances,
    train_ids: tuple[str, ...],
    pilot_ids: tuple[str, ...],
    seed: int = SEED,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    by_id = {instance.question_id: instance for instance in instances}
    train = set(train_ids)
    pilot = set(pilot_ids)
    if missing := train - set(by_id):
        raise ValueError(f"train manifest has {len(missing)} unknown question(s)")
    if not pilot <= train:
        raise ValueError("pilot manifest is not a subset of train150")

    remaining_by_type: dict[str, list[str]] = {}
    for question_id in train_ids:
        if question_id in pilot:
            continue
        question_type = by_id[question_id].question_type
        remaining_by_type.setdefault(question_type, []).append(question_id)

    dev: list[str] = []
    tune: list[str] = []
    for question_type, count in DEV_QUOTAS.items():
        candidates = sorted(remaining_by_type.get(question_type, []))
        if len(candidates) < count:
            raise ValueError(
                f"only {len(candidates)} remaining {question_type!r} questions; need {count}"
            )
        random.Random(f"{seed}:{question_type}").shuffle(candidates)
        dev.extend(candidates[:count])
        tune.extend(candidates[count:])

    if set(dev) & set(tune) or set(dev) & pilot or set(tune) & pilot:
        raise ValueError("v3 phase-3 manifests overlap")
    if set(dev) | set(tune) | pilot != train:
        raise ValueError("pilot, tune and dev manifests do not partition train150")
    return tuple(sorted(tune)), tuple(sorted(dev))


def expected_manifests() -> tuple[Manifest, Manifest]:
    train = load_manifest(TRAIN)
    pilot = load_manifest(PILOT)
    instances = lme.load(train.variant, REPO / "data")
    tune_ids, dev_ids = split_ids(instances, train.question_ids, pilot.question_ids)
    tune = Manifest(
        name="v3-reasoning-tune42",
        variant=train.variant,
        seed=SEED,
        question_ids=tune_ids,
        note=(
            "The 42 train150 questions left after the v3 pilot and sealed dev60 split. "
            "Individual content may be inspected for v3 mechanism development; no test100."
        ),
    )
    dev = Manifest(
        name="v3-reasoning-dev60",
        variant=train.variant,
        seed=SEED,
        question_ids=dev_ids,
        note=(
            "Sealed phase-3 development check: 18 temporal, 18 multi-session, 10 update, "
            "5 user, 5 assistant and 4 preference questions. Excludes the pilot48 and "
            "must not be inspected or used for tuning before the registered run."
        ),
    )
    return tune, dev


def _write_or_verify(manifest: Manifest, destination: Path) -> str:
    if destination.exists():
        actual = json.loads(destination.read_text(encoding="utf-8"))
        if actual != manifest.to_dict():
            raise ValueError(f"existing {destination.relative_to(REPO)} differs")
        return "unchanged"
    manifest.save(destination)
    return "created"


def main() -> int:
    try:
        tune, dev = expected_manifests()
        tune_state = _write_or_verify(tune, TUNE_DESTINATION)
        dev_state = _write_or_verify(dev, DEV_DESTINATION)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"STOP: {exc}")
        return 2
    print(
        f"PASS: {tune_state} tune42 + {dev_state} sealed dev60 · "
        "disjoint from pilot48 · complete train150 partition"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
