"""Classify every readable answer failure, so the error picture rests on more than
one small run.

The v3.3 tune42 diagnosis that motivated this had 11 failures split across four
buckets — two to four samples each, which is not enough to choose what to build
next. This widens the sample using only data already on disk: `heldout100` is spent,
so reading it costs nothing that was not already spent, and `pilot48`/`tune42` are
`train150` slices the data protocol allows to be read without limit. No provider
call is made and no sealed set is opened.

The classifier is deliberately rule-based rather than another model call. Its rules
run on the fields every run records, and it is validated against the eleven v3.3
failures that were read by hand first: a taxonomy that cannot reproduce a manual
read is not evidence, so `--validate` reports that agreement before any aggregate
is believed.

v1 and v3 pools are never merged. `heldout100` was produced by a different extractor
generation, and folding its extraction defects into v3's totals would attribute the
wrong cause to the current system.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

POOLS = {
    "heldout100 (v1, spent)": "results/raw/*heldout100*.jsonl",
    "pilot48 (v3, train)": "results/sealed/v3-answer-pilot/*.jsonl",
    "tune42 (v3, train)": "results/sealed/v3-phase*/*.jsonl",
}

_ABSTAIN = re.compile(
    r"\b(?:i (?:do not|don't) know|cannot (?:determine|calculate|be determined)|"
    r"not (?:stated|mentioned|available|specified|provided)|"
    r"no (?:information|memories|memory|evidence|record)\b|"
    r"is not (?:in|present|contained))",
    re.IGNORECASE,
)
_NUM = re.compile(r"\b\d+(?:\.\d+)?\b")
_WORD_NUM = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


def _numbers(text: str) -> set[float]:
    found = {float(m) for m in _NUM.findall(text or "")}
    for word, value in _WORD_NUM.items():
        if re.search(rf"\b{word}\b", text or "", re.IGNORECASE):
            found.add(float(value))
    return found


def classify(row: dict) -> str:
    """One label per failed row, from fields every run records."""
    notes = row.get("notes") or {}
    hypothesis = str(row.get("hypothesis") or "")
    gold = str(row.get("gold") or "")

    # 1. Retrieval never reached the labelled conversation. Nothing downstream
    #    could have recovered this, so it is checked first.
    if notes.get("source_session_recalled") is False:
        return "retrieval_miss"

    # 2. Some, but not all, labelled sessions arrived. Only the later runs record
    #    this, so it can only ever be under-counted, never over-counted.
    if notes.get("all_source_sessions_recalled") is False:
        return "partial_evidence"

    # 3. The system had the source and declined to answer anyway. This is the
    #    bucket that abstention-as-a-strength turns into a cost.
    if row.get("is_abstention") or notes.get("answer_status") == "no_evidence":
        return "abstained_with_source"
    if _ABSTAIN.search(hypothesis):
        return "abstained_with_source"

    # 4. Both sides assert counts and the counts are not the same set: an
    #    enumeration problem, not a lookup problem. Disagreeing on *some* of the
    #    numbers counts — "you led 5 then and 5 now" against a gold of "4 then,
    #    5 now" is a miscount that happens to get one of the two right, and
    #    requiring disjoint sets would file it as an unrelated wrong value.
    gold_numbers, hypothesis_numbers = _numbers(gold), _numbers(hypothesis)
    if gold_numbers and hypothesis_numbers and gold_numbers != hypothesis_numbers:
        return "count_mismatch"

    # 5. Confident, specific, and wrong.
    return "wrong_specific_value"


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval. Normal approximation is not usable here: several
    buckets sit near 0 or 1, where it produces bounds outside [0, 1]."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def load(pattern: str) -> list[tuple[str, dict]]:
    rows = []
    for path in sorted(glob.glob(str(REPO / pattern))):
        name = Path(path).name
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append((name, json.loads(line)))
    return rows


# The eleven v3.3 tune42 failures, labelled by reading them before this file
# existed. Keyed by gold answer because row files carry no question id.
#
# Two of these labels were corrected after the first validation run, and the
# reason is worth recording because "the reference moved towards the rule" is
# exactly how a validation gate gets quietly defeated. Both were rows where only
# half or three quarters of the labelled sessions had been retrieved, and the hand
# label recorded what the answer *did* (abstained, miscounted) rather than what
# had gone wrong upstream. The ordering in `classify` is the deliberate choice:
# evidence completeness is diagnosed before answer behaviour, because a system
# holding half the evidence has already failed regardless of what it then says.
# Labelling those as behaviour failures would have inflated the buckets that argue
# for changing the answerer and hidden the ones that argue for changing recall.
MANUAL = {
    "When you just started your new role as Senior Software": "count_mismatch",
    "18 days. 19 days (including the last day)": "partial_evidence",
    "38 subjects": "abstained_with_source",
    "The music shop on Main St.": "wrong_specific_value",
    "5": "partial_evidence",
    "3": "count_mismatch",
    "Spotify": "abstained_with_source",
    "my cousin's wedding": "wrong_specific_value",
    "The arrival of the new prime lens": "abstained_with_source",
}


def validate() -> int:
    """Agreement against the hand-read labels. A rule set that cannot reproduce
    them is not trustworthy on the 200-odd rows nobody read."""
    rows = [r for _, r in load("results/sealed/v3-phase5-tune3/*.jsonl") if not r.get("correct")]
    agree = total = 0
    disagreements = []
    for row in rows:
        gold = str(row.get("gold") or "").strip()
        expected = next((v for k, v in MANUAL.items() if gold.startswith(k)), None)
        if expected is None:
            continue
        total += 1
        got = classify(row)
        if got == expected:
            agree += 1
        else:
            disagreements.append((str(row.get("gold"))[:52], expected, got))
    print(f"validation against hand-read labels: {agree}/{total} agree")
    for gold, expected, got in disagreements:
        print(f"  MISMATCH  gold={gold!r}\n            manual={expected}  rule={got}")
    return 0 if total and agree / total >= 0.8 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.validate:
        return validate()

    for pool, pattern in POOLS.items():
        rows = load(pattern)
        failures = [(f, r) for f, r in rows if not r.get("correct")]
        labels = Counter(classify(r) for _, r in failures)
        distinct = len({str(r.get("gold")) for _, r in failures})
        print(f"\n{pool}")
        files = len({f for f, _ in rows})
        print(
            f"  {len(rows)} rows over {files} files · {len(failures)} failures "
            f"· {distinct} distinct questions"
        )
        # Per-question, not per-row. Repeats and arms make the same question fail
        # several times, and those failures are not independent draws: treating
        # them as such shrinks every interval by roughly the repeat count and
        # would claim far more precision than 63 distinct questions can support.
        # The row counts stay visible because they say how reproducible each
        # failure is; the intervals are computed on questions.
        per_question: dict[str, Counter] = defaultdict(Counter)
        for _, row in failures:
            per_question[str(row.get("gold"))][classify(row)] += 1
        question_labels = Counter(c.most_common(1)[0][0] for c in per_question.values())
        n_questions = len(per_question)

        print(f"    {'bucket':<24}{'rows':>5}{'questions':>11}{'share':>8}   95% CI on questions")
        for label, _ in labels.most_common():
            rows_n = labels[label]
            q_n = question_labels.get(label, 0)
            low, high = wilson(q_n, n_questions)
            share = q_n / n_questions
            interval = f"[{low:.1%}, {high:.1%}]"
            print(f"    {label:<24}{rows_n:>5}{q_n:>11}{share:>8.1%}   {interval}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
