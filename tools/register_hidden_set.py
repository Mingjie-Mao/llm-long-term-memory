"""Register a final test set, and refuse one that is not actually unseen.

LongMemEval-S is exhausted: its 500 questions are partitioned across five disjoint sets
with zero remaining, and `longmemeval_oracle.json` carries the same 500 ids over a
different haystack. So a genuine final test has to come from outside, and the risk is not
that someone knowingly reuses old questions — it is that a "new" corpus turns out to
share questions, or paraphrases of them, with data this project has already read.

This checks that before anything is frozen, because the check is worthless afterwards.

Four refusals, in the order a set is most likely to fail them:

1. **Question-id overlap** with any registered set. The cheap, exact check.
2. **Verbatim question overlap** after normalisation. New ids over old text is the
   failure that an id check alone cannot see, and it is what
   `longmemeval_oracle.json` would have looked like had it been renumbered.
3. **Near-duplicate questions**, by token overlap against every question already read.
   A paraphrase is contamination for a memory benchmark in the same way a copy is.
4. **Re-registration.** A set registered twice is a set someone reconsidered after
   seeing a result, which is the whole thing a freeze exists to prevent.

Passing this is necessary and not sufficient: it cannot tell whether the *source corpus*
overlaps the training data of the model being evaluated, and nothing on this machine can.
That limit is recorded in the manifest rather than left for a reader to infer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFESTS = REPO / "results/manifests"
# Every set whose questions this project has read, in any capacity. `dev60` and the
# probe split are inside train150 and are covered by it.
REGISTERED = ("dev50", "heldout100", "train150", "dev100", "test100")
SOURCES = ("data/longmemeval_s_cleaned.json", "data/longmemeval_oracle.json")

# A question sharing this share of its content words with one already read is treated as
# the same question. Deliberately low: a false refusal costs a conversation, and a false
# acceptance costs the only unseen measurement v4 will ever get.
_NEAR_DUPLICATE = 0.8
_WORD = re.compile(r"[a-z0-9]+")
# fmt: off
_STOP = frozenset([
    "a", "an", "the", "of", "and", "or", "to", "in", "on", "at", "for", "with", "my",
    "i", "you", "it", "is", "was", "were", "do", "did", "does", "how", "what", "when",
    "where", "which", "who", "why",
])
# fmt: on


def _normalise(question: str) -> str:
    return " ".join(_WORD.findall((question or "").lower()))


def _content_words(question: str) -> frozenset[str]:
    return frozenset(w for w in _WORD.findall((question or "").lower()) if w not in _STOP)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seen_questions() -> tuple[set[str], dict[str, str]]:
    """Every question id this project has read, and normalised text -> the set it is in."""
    ids: set[str] = set()
    for name in REGISTERED:
        manifest = MANIFESTS / f"{name}.json"
        if not manifest.is_file():
            raise ValueError(f"missing comparison manifest: {name}")
        recorded = json.loads(manifest.read_text(encoding="utf-8"))["question_ids"]
        if not recorded or any(not isinstance(qid, str) or not qid.strip() for qid in recorded):
            raise ValueError(f"invalid comparison manifest: {name}")
        ids.update(recorded)

    text_to_set: dict[str, str] = {}
    covered: set[str] = set()
    for relative in SOURCES:
        path = REPO / relative
        if not path.is_file():
            continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            if row.get("question_id") in ids:
                question = row.get("question")
                if isinstance(question, str) and _normalise(question):
                    text_to_set[_normalise(question)] = relative
                    covered.add(row["question_id"])
    if covered != ids:
        raise ValueError(f"comparison corpus is incomplete: {len(ids - covered)} questions missing")
    return ids, text_to_set


def check(candidate: list[dict], ids: set[str], texts: dict[str, str]) -> list[str]:
    """Every reason this set may not be used as a final test."""
    problems: list[str] = []
    if not isinstance(candidate, list) or not candidate:
        return ["candidate must be a non-empty list"]
    if any(not isinstance(row, dict) for row in candidate):
        return ["every candidate row must be an object"]

    missing = [
        i
        for i, row in enumerate(candidate)
        if not isinstance(row.get("question_id"), str) or not row["question_id"].strip()
    ]
    if missing:
        problems.append(f"{len(missing)} rows have no question_id")
    if any(
        not isinstance(row.get("question"), str) or not _normalise(row["question"])
        for row in candidate
    ):
        problems.append("every row needs question text supported by the overlap checker")
    if problems:
        return problems

    duplicate_ids = len(candidate) - len({row.get("question_id") for row in candidate})
    if duplicate_ids:
        problems.append(f"{duplicate_ids} question_ids are repeated within the candidate")

    shared = sorted({row.get("question_id") for row in candidate} & ids)
    if shared:
        problems.append(
            f"{len(shared)} question_ids are already in a registered set, e.g. {shared[:3]}"
        )

    verbatim = [
        row["question_id"] for row in candidate if _normalise(row.get("question", "")) in texts
    ]
    if verbatim:
        problems.append(
            f"{len(verbatim)} questions are verbatim repeats of ones already read, "
            f"e.g. {verbatim[:3]} — new ids over old text"
        )

    seen_words = [_content_words(text) for text in texts]
    near: list[str] = []
    for row in candidate:
        words = _content_words(row.get("question", ""))
        if not words:
            continue
        for other in seen_words:
            if other and len(words & other) / max(len(words), len(other)) >= _NEAR_DUPLICATE:
                near.append(row["question_id"])
                break
    if near:
        problems.append(
            f"{len(near)} questions are near-duplicates of ones already read "
            f"(>= {_NEAR_DUPLICATE:.0%} content-word overlap), e.g. {near[:3]}"
        )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path, help="JSON list of question rows")
    parser.add_argument("--name", required=True, help="set name, e.g. v4-hidden")
    parser.add_argument("--source", required=True, help="where it came from, for the record")
    parser.add_argument("--licence", required=True, help="licence it is used under")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.name):
        print("STOP: name must contain only letters, numbers, underscores or hyphens")
        return 2
    manifest_path = MANIFESTS / f"{args.name}.json"
    if manifest_path.exists():
        print(f"STOP: {args.name} is already registered at {manifest_path}")
        print("Registering a final set twice means it was reconsidered after a result.")
        return 2
    if not args.candidate.is_file():
        print(f"STOP: no candidate at {args.candidate}")
        return 2

    try:
        candidate_bytes = args.candidate.read_bytes()
        candidate = json.loads(candidate_bytes)
        ids, texts = seen_questions()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"STOP: cannot validate candidate or comparison corpus ({type(exc).__name__})")
        return 2
    if not isinstance(candidate, list) or not candidate:
        print("STOP: the candidate must be a non-empty JSON list of question rows")
        return 2

    print(f"candidate     : {len(candidate)} questions from {args.candidate}")
    print(f"already read  : {len(ids)} question ids across {', '.join(REGISTERED)}")

    problems = check(candidate, ids, texts)
    if problems:
        print("\nSTOP: this set is not unseen.")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    payload = {
        "name": args.name,
        "n": len(candidate),
        "question_ids": sorted(row["question_id"] for row in candidate),
        "source": args.source,
        "licence": args.licence,
        "file_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "registered_at_utc": datetime.now(UTC).isoformat(),
        "checked_against": list(REGISTERED),
        "one_shot": (
            "Answer once, after every choice is frozen. A second run may only put an "
            "error bar on the result; it may never change the system."
        ),
        "not_verified_here": (
            "Overlap with the evaluated model's own training data cannot be checked "
            "from this repository. A pass here means unseen by this project, which is "
            "a weaker claim than unseen by the model."
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with manifest_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except FileExistsError:
        print("STOP: another registration already created this manifest")
        return 2

    print(f"\nOK: no overlap found. Registered {args.name} at {manifest_path}")
    print("  This set is now frozen. Answer it once, after every v4 choice is fixed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
