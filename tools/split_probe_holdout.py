"""Split the probe set into a development half and a held-out half, before v4 is measured.

LongMemEval-S is exhausted, so no unseen questions remain and `v4-hidden` cannot be cut
from it. The probes are a different instrument and still have one clean split left: 188
of the 229 have never been answered by any run. Reserving half of those now is the only
held-out check v4 can have without acquiring new data, and it is available exactly once —
the first v4 measurement over the whole set would spend it.

**This is not a substitute for a benchmark.** Both halves are derived from the same
store by the same generator, so a held-out probe controls for fitting the *probe set*
and not for fitting the *store*. It answers "did the operation improve on questions the
development loop never saw", which is narrower than "did the system improve", and it must
never be reported as accuracy.

The split is deterministic from `(seed, store fingerprint, probe id)` and is recorded as
two id lists rather than two files, so it cannot be re-drawn after a result is seen
without the change being visible in the manifest.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROBES = REPO / "results/analysis/synthesis-probes.json"
MANIFEST = REPO / "results/manifests/v4-probe-split.json"
ANSWERED = ("results/raw/probes*.jsonl", "results/archive/synthesis-probes-v1/probes*.jsonl")
# Below this a half cannot carry a result worth reporting, so the kind stays whole and
# goes to development rather than producing a hidden stratum of four questions.
MIN_PER_HALF = 20


def _answered_questions() -> set[str]:
    seen: set[str] = set()
    for pattern in ANSWERED:
        for path in glob.glob(str(REPO / pattern)):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        question = json.loads(line).get("question")
                        if question:
                            seen.add(question)
    return seen


def split(spec: dict, seen: set[str], seed: int) -> dict:
    fingerprint = spec["store_fingerprint"]
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for probe in spec["probes"]:
        by_kind[probe["kind"]].append(probe)

    dev: list[str] = []
    hidden: list[str] = []
    notes: dict[str, dict] = {}
    for kind, probes in sorted(by_kind.items()):
        untouched = [p for p in probes if p["question"] not in seen]
        touched = [p for p in probes if p["question"] in seen]
        # Anything already answered is development by definition; it cannot be held out
        # from a loop that has seen it.
        dev.extend(p["probe_id"] for p in touched)

        if len(untouched) < 2 * MIN_PER_HALF:
            dev.extend(p["probe_id"] for p in untouched)
            notes[kind] = {
                "untouched": len(untouched),
                "held_out": 0,
                "reason": (
                    f"fewer than {2 * MIN_PER_HALF} untouched probes; a half this small "
                    "cannot carry a reportable result, so the kind stays in development"
                ),
            }
            continue

        # Deterministic per probe: hashing the id with the seed and the store means the
        # assignment cannot shift when another kind's count changes.
        def rank(probe: dict) -> str:
            return hashlib.sha256(f"{seed}:{fingerprint}:{probe['probe_id']}".encode()).hexdigest()

        ordered = sorted(untouched, key=rank)
        cut = len(ordered) // 2
        hidden.extend(p["probe_id"] for p in ordered[:cut])
        dev.extend(p["probe_id"] for p in ordered[cut:])
        notes[kind] = {
            "untouched": len(untouched),
            "already_answered": len(touched),
            "held_out": cut,
        }

    return {
        "schema_version": 1,
        "name": "v4-probe-split",
        "seed": seed,
        "store_fingerprint": fingerprint,
        "probe_set_sha256": hashlib.sha256(PROBES.read_bytes()).hexdigest(),
        "rule": (
            "Development may be run and read at any time. The held-out half is answered "
            "once, after every v4 choice is frozen, and only aggregate per-operation "
            "results are reported from it."
        ),
        "not_a_benchmark": (
            "Both halves come from the same store and generator. A held-out probe "
            "controls for fitting the probe set, not for fitting the store, and is not "
            "accuracy."
        ),
        "per_kind": notes,
        "development": sorted(dev),
        "held_out": sorted(hidden),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    spec = json.loads(PROBES.read_text(encoding="utf-8"))
    manifest = split(spec, _answered_questions(), args.seed)

    kinds = {p["probe_id"]: p["kind"] for p in spec["probes"]}
    held = Counter(kinds[i] for i in manifest["held_out"])
    dev = Counter(kinds[i] for i in manifest["development"])
    print(f"{'kind':<16}{'development':>13}{'held out':>10}")
    for kind in ("count", "duration", "comparison", "current_state"):
        print(f"  {kind:<14}{dev[kind]:>13}{held[kind]:>10}")
    print(f"  {'TOTAL':<14}{len(manifest['development']):>13}{len(manifest['held_out']):>10}")

    if args.write:
        if MANIFEST.exists():
            print(
                f"\nSTOP: {MANIFEST.name} already exists; re-drawing a split after a "
                "result is the thing it prevents"
            )
            return 2
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {MANIFEST}")
    else:
        print("\n(dry run — pass --write to record it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
