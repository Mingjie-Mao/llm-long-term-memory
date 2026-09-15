"""Split BEAM into a development half and a final-test half, before any question is read.

BEAM (CC BY-SA 4.0) ships each scale as its own file, and conversation ids restart at "1"
in every file, so a conversation is `<scale>-<id>` here. The unit of the split is the
conversation, never the question: a conversation's twenty questions share one memory
store, so splitting its questions would put that store on both sides.

Twenty seed ids recur between the 100K and 500K files. On the pinned revision those pairs
share no theme, subtopics, title, narrative, persona or 200-character message, and only
five share a category, so the recurrence looks like numbering rather than shared content.
They are kept on one side anyway: at these counts it costs nothing, and it closes the only
route by which a development conversation could be related to a final-test one.

Only conversation ids, scales and seed ids decide the split. Question text is read for one
purpose, the candidate file `register_hidden_set.py` checks against every question this
project has read, and that file quotes the benchmark, so it is written under the ignored
`data/`. The committed manifests carry ids and counts only.

Reading the files needs pyarrow, which the project environment does not install, so run
this with a Python that has it. The split itself is plain Python, and that is what the
tests exercise.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import platform
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE = "https://huggingface.co/datasets/Mohammadta/BEAM"
REVISION = "3205395e897e7318c7b094ef4e6047b9b82dbb03"
LICENCE = "CC BY-SA 4.0"
# Pinned by content rather than by name: a different download would cut a different split,
# and nothing downstream would notice.
FILES = {
    "100K": (
        "data/beam/100K-00000-of-00001.parquet",
        "c0519be25907005ba873c927c50877471d550873039d96c041554d0075a78ace",
    ),
    "500K": (
        "data/beam/500K-00000-of-00001.parquet",
        "af05921c979355038e1761b7cde3d2dd713200dd3071b278de0200f6c7f30122",
    ),
}
SEED = 20260914
TEST_FRACTION = 0.6
MANIFESTS = REPO / "results/manifests"
SPLIT = MANIFESTS / "beam-split.json"
DEV = MANIFESTS / "beam-dev.json"
TEST = MANIFESTS / "beam-test.json"
CANDIDATE = REPO / "data/beam/beam-test.candidate.json"


def parse_probing(raw: str) -> dict:
    """`probing_questions` is stored as a string; the dataset card decodes it as a literal."""
    try:
        value = json.loads(raw)
    except ValueError:
        value = ast.literal_eval(raw)
    if not isinstance(value, dict):
        raise ValueError("probing_questions must decode to a mapping of ability to questions")
    return value


def question_rows(scale: str, conversation_id: str, probing: dict) -> list[dict]:
    """One row per question, with an id naming its scale, conversation and ability."""
    return [
        {
            "question_id": f"beam-{scale}-{conversation_id}-{ability}-{index}",
            "question": item.get("question"),
        }
        for ability in sorted(probing)
        for index, item in enumerate(probing[ability])
    ]


def conversation_of(question_id: str) -> str:
    """`beam-500K-7-temporal_reasoning-1` -> `500K-7`. Ability names contain no hyphen."""
    parts = question_id.split("-")
    return f"{parts[1]}-{parts[2]}"


def load(repo: Path = REPO) -> list[dict]:
    """Read the pinned files, keeping conversation ids, scales, seed ids and question rows."""
    import pyarrow.parquet as pq  # the only code path that needs pyarrow

    conversations = []
    for scale, (relative, expected) in FILES.items():
        path = repo / relative
        if not path.is_file():
            raise ValueError(f"missing {relative}; download revision {REVISION[:12]} first")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"{relative} is not the pinned revision")
        columns = ["conversation_id", "conversation_seed", "probing_questions"]
        for row in pq.read_table(path, columns=columns).to_pylist():
            conversation_id = row["conversation_id"]
            conversations.append(
                {
                    "key": f"{scale}-{conversation_id}",
                    "scale": scale,
                    "seed_id": row["conversation_seed"]["id"],
                    "questions": question_rows(
                        scale, conversation_id, parse_probing(row["probing_questions"])
                    ),
                }
            )
    return conversations


def split(
    conversations: list[dict], test_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Put whole seed groups on one side, hitting each scale's final-test count exactly.

    Groups are bucketed by how many conversations of each scale they hold, and each bucket
    is shuffled with the seed. How many groups each bucket gives the final half is solved
    exactly, preferring the solution closest to the fraction in every bucket. A greedy fill
    can miss a target that an exact count reaches.
    """
    keys = [c["key"] for c in conversations]
    if len(keys) != len(set(keys)):
        raise ValueError("conversation keys must be unique")
    scales = sorted({c["scale"] for c in conversations})
    members = defaultdict(list)
    for conversation in conversations:
        members[conversation["seed_id"]].append(conversation)
    buckets = defaultdict(list)
    for seed_id in sorted(members, key=str):
        group = members[seed_id]
        signature = tuple(sum(1 for c in group if c["scale"] == s) for s in scales)
        buckets[signature].append(sorted(c["key"] for c in group))
    signatures = sorted(buckets)
    rng = random.Random(seed)
    for signature in signatures:
        rng.shuffle(buckets[signature])
    target = tuple(
        round(sum(1 for c in conversations if c["scale"] == s) * test_fraction) for s in scales
    )
    best = None
    for takes in itertools.product(*(range(len(buckets[s]) + 1) for s in signatures)):
        reached = tuple(
            sum(take * signature[i] for take, signature in zip(takes, signatures, strict=True))
            for i in range(len(scales))
        )
        if reached != target:
            continue
        distance = sum(
            abs(take - test_fraction * len(buckets[signature]))
            for take, signature in zip(takes, signatures, strict=True)
        )
        if best is None or distance < best[0]:
            best = (distance, takes)
    if best is None:
        raise ValueError(f"no assignment of whole seed groups reaches target {target}")
    test = sorted(
        key
        for take, signature in zip(best[1], signatures, strict=True)
        for group in buckets[signature][:take]
        for key in group
    )
    return sorted(set(keys) - set(test)), test


def manifests(
    conversations: list[dict], dev: list[str], test: list[str], test_fraction: float, seed: int
) -> tuple[dict, dict, list[dict]]:
    """The committed records, which carry ids and counts only, and the ignored candidate."""
    by_key = {c["key"]: c for c in conversations}
    scales = sorted({c["scale"] for c in conversations})

    def question_ids(half: list[str]) -> list[str]:
        return sorted(q["question_id"] for key in half for q in by_key[key]["questions"])

    def describe(half: list[str]) -> dict:
        return {
            "conversations": half,
            "per_scale": {s: sum(1 for k in half if by_key[k]["scale"] == s) for s in scales},
            "questions": len(question_ids(half)),
        }

    groups = defaultdict(list)
    for conversation in conversations:
        groups[conversation["seed_id"]].append(conversation["key"])
    record = {
        "name": "beam-split",
        "source": SOURCE,
        "revision": REVISION,
        "licence": LICENCE,
        "files": {scale: {"path": p, "sha256": d} for scale, (p, d) in FILES.items()},
        "seed": seed,
        "python": platform.python_version(),
        "test_fraction": test_fraction,
        "rule": (
            "whole conversations; conversations sharing a seed id on the same side; each "
            "scale's final-test count exactly round(n * test_fraction)"
        ),
        "read_before_split": "conversation ids, scales and seed ids; no question, answer or rubric",
        "shared_seed_groups": sorted(sorted(g) for g in groups.values() if len(g) > 1),
        "dev": describe(dev),
        "test": describe(test),
    }
    development = {
        "name": "beam-dev",
        "variant": "beam",
        "seed": seed,
        "n": len(question_ids(dev)),
        "note": (
            "BEAM development half: may be read question by question for diagnosis and "
            "candidate selection, and is never reported as a final result. The final half is "
            "beam-test, registered by register_hidden_set.py and answered once."
        ),
        "conversations": dev,
        "question_ids": question_ids(dev),
    }
    candidate = [q for key in test for q in by_key[key]["questions"]]
    return record, development, candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    args = parser.parse_args()
    for path in (SPLIT, DEV, TEST):
        if path.exists():
            print(f"STOP: {path.relative_to(REPO)} exists; cutting again is reconsidering it")
            return 2
    try:
        conversations = load()
        dev, test = split(conversations, args.test_fraction, args.seed)
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as exc:
        print(f"STOP: {exc}")
        return 2
    record, development, candidate = manifests(
        conversations, dev, test, args.test_fraction, args.seed
    )
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    for path, payload in ((SPLIT, record), (DEV, development)):
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2) + "\n")
    CANDIDATE.write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")
    summary = {
        half: record[half]["per_scale"] | {"questions": record[half]["questions"]}
        for half in ("dev", "test")
    }
    print(json.dumps(summary))
    print(
        f"next: python3 tools/register_hidden_set.py {CANDIDATE.relative_to(REPO)} "
        f'--name beam-test --source "{SOURCE}@{REVISION}" --licence "{LICENCE}"'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
