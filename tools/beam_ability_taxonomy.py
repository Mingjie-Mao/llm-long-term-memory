"""What each BEAM ability's rubric actually grades, read before any provider call.

`configs/beam-eval.json` proposed keeping `instruction_following`, `preference_following`
and `summarization` out of the primary metric, for the reason that "how the answer is
written moves these as much as what memory kept". That reason is checkable for nothing,
and this checks it: every rubric item of the development half is screened for wording
that could name a *manner of writing* rather than something the conversation established,
and every match is then read once by hand and recorded below.

BEAM's own `instruction_type` label cannot do this job. It marks 31 of the 44 development
instruction-following questions `format_instruction`, and among them are "Always specify
practice durations" and "Always provide player attendance numbers" — content, filed under
format. So the screen reads the rubric items themselves, which are what the judge is given.

The pattern is built to over-match, so that the matched set is small enough to read in
full: 24 items out of 1,165. `--show` prints them. They are not written to the artifact —
BEAM is CC BY-SA 4.0 and nothing that quotes it is committed (`.gitignore`) — so the hand
reading is pinned here instead, by item id, and the tool refuses to report a count if the
screen ever matches an item this table has not judged.

    python3 tools/beam_ability_taxonomy.py [--show]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results/analysis/beam-ability-taxonomy.json"

HALF = "dev"
"""The development half only. The final half is answered once, at the end, and reading
its rubrics to choose a metric would make that half a development set."""

MANNER = re.compile(
    r"step[- ]by[- ]step|sequential step|breaks? down|concise|brief|bullet|numbered"
    r"|\bformat\b|\btone\b|\btable\b|heading|logical steps|framework|presentation style"
    r"|structure|paragraph|length|\blist form",
    re.I,
)
"""Wording that may name how a reply is written. Deliberately loose."""

REVIEWED: dict[str, str] = {
    # Read 2026-09-14. `manner`: the item is satisfied or missed by how the reply is
    # written. `content`: the screen matched a word inside the subject matter.
    "beam-100K-5-instruction_following-0#1": "manner",
    "beam-100K-6-instruction_following-0#1": "manner",
    "beam-100K-10-instruction_following-0#1": "manner",
    "beam-500K-8-instruction_following-0#1": "manner",
    "beam-100K-5-preference_following-0#1": "manner",
    "beam-100K-5-preference_following-1#1": "manner",
    "beam-100K-12-preference_following-0#1": "manner",
    "beam-100K-10-preference_following-1#2": "content",
    "beam-100K-12-preference_following-1#2": "content",
    "beam-100K-18-abstention-1#1": "content",
    "beam-500K-22-abstention-1#1": "content",
    "beam-500K-33-abstention-1#1": "content",
    "beam-100K-10-event_ordering-1#3": "content",
    "beam-100K-6-summarization-1#4": "content",
    "beam-100K-6-summarization-1#5": "content",
    "beam-100K-10-summarization-0#2": "content",
    "beam-100K-17-summarization-0#2": "content",
    "beam-500K-8-summarization-1#2": "content",
    "beam-500K-8-summarization-1#12": "content",
    "beam-500K-20-summarization-1#1": "content",
    "beam-500K-23-summarization-0#3": "content",
    "beam-500K-30-summarization-1#1": "content",
    "beam-500K-33-summarization-1#1": "content",
    "beam-500K-34-summarization-1#5": "content",
}


class Unreviewed(ValueError):
    """The screen matched an item no one has read, so no count can be reported."""


def screen(half: str = HALF, data_dir: Path | None = None):
    from llm_long_term_memory.evaluation.datasets import beam

    instances = beam.load(half, data_dir or REPO / "data")
    items: dict[str, list[int]] = defaultdict(list)
    matched: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for instance in instances:
        items[instance.question_type].append(len(instance.rubric))
        for number, item in enumerate(instance.rubric, start=1):
            if MANNER.search(item):
                matched[instance.question_type].append((f"{instance.question_id}#{number}", item))

    unreviewed = sorted(key for rows in matched.values() for key, _ in rows if key not in REVIEWED)
    if unreviewed:
        raise Unreviewed(
            f"{len(unreviewed)} screened item(s) carry no hand verdict, first "
            f"{unreviewed[0]!r}. Read them with --show and record each in REVIEWED."
        )

    abilities = {}
    for ability in sorted(items):
        counts = items[ability]
        rows = matched[ability]
        abilities[ability] = {
            "questions": len(counts),
            "rubric_items": sum(counts),
            "items_per_question": {
                "mean": round(statistics.mean(counts), 2),
                "median": statistics.median(counts),
                "max": max(counts),
            },
            "screened": len(rows),
            "manner_after_reading": sum(1 for key, _ in rows if REVIEWED[key] == "manner"),
            "manner_item_ids": sorted(key for key, _ in rows if REVIEWED[key] == "manner"),
        }
    payload = {
        "name": "beam-ability-taxonomy",
        "half": half,
        "source": "https://huggingface.co/datasets/Mohammadta/BEAM",
        "licence": "CC BY-SA 4.0",
        "provider_calls": 0,
        "question": "does this ability's rubric grade what memory kept, or how the reply reads?",
        "screen": MANNER.pattern,
        "method": "loose keyword screen over every rubric item, then every match read by hand",
        "quotes_beam": False,
        "abilities": abilities,
    }
    return payload, matched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show",
        action="store_true",
        help="print the screened rubric items so the hand verdicts can be re-checked",
    )
    args = parser.parse_args()
    try:
        payload, matched = screen()
    except Unreviewed as exc:
        print(f"STOP: {exc}")
        return 2
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    total = sum(row["questions"] for row in payload["abilities"].values())
    print(f"{payload['half']} half, {total} questions, zero provider calls")
    header = f"{'ability':28} {'qs':>3} {'items':>6} {'med':>5} {'max':>4}"
    print(f"{header} {'screened':>9} {'manner':>7}")
    for ability, row in payload["abilities"].items():
        per = row["items_per_question"]
        print(
            f"{ability:28} {row['questions']:3d} {row['rubric_items']:6d} "
            f"{per['median']:5.1f} {per['max']:4d} {row['screened']:9d} "
            f"{row['manner_after_reading']:7d}"
        )
    if args.show:
        for ability in sorted(matched):
            print(f"\n--- {ability}")
            for key, item in matched[ability]:
                print(f"  [{REVIEWED[key]:7}] {key}\n      {item}")
    print(f"\nwritten: {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
