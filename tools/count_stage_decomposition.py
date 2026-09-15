"""How far a proposed count member gets toward a quotable answer — no provider calls.

Runs over the review packet only, so it never reads sealed development questions. For
each proposed member it measures, against the *supplied* evidence pool, how much of the
member's name a single turn actually supports, and whether the memory that carries it
sits under a predicate the relation table approves.

Three limits, stated because they bound every number below:

* Proposed members are produced from extracted memories, so a fact extraction dropped
  entirely cannot appear here at all. This instrument cannot measure extraction loss.
* Member names are canonical labels ("Fleetwood Mac - Rumours"), not quotes. Partial
  support usually means the label was composed, not that the evidence is missing.
* Nothing here is an accuracy measurement. A member can be fully supported and still be
  the wrong answer to the question; only human review decides that.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKET = REPO / "results/review/count-v1/packet.json"
OUT = REPO / "results/review/count-v1/stage-decomposition.json"

# Words that carry no identifying weight inside a member label.
NOISE = {"the", "and", "for", "with", "named", "item", "items", "edition", "inch"}

STAGES = (
    "unsupported_in_user_text",
    "only_in_assistant_text",
    "label_partly_supported",
    "carried_by_unapproved_predicate",
    "quotable_under_an_approved_predicate",
)


def tokens(text: str) -> list[str]:
    words = re.sub(r"[^a-z0-9]+", " ", (text or "").casefold()).split()
    return [w for w in words if len(w) >= 3 and w not in NOISE] or words


def coverage(source_text: str, entity: str) -> float:
    """Share of the member label's identifying tokens present in one turn."""
    wanted = tokens(entity)
    if not wanted:
        return 0.0
    present = set(re.sub(r"[^a-z0-9]+", " ", (source_text or "").casefold()).split())
    return sum(1 for w in wanted if w in present) / len(wanted)


def stage_of(entity: str, item: dict, approved: set[str]) -> tuple[str, float]:
    user = max(
        (coverage(s["text"], entity) for s in item["sources"] if s["role"] == "user"),
        default=0.0,
    )
    other = max(
        (coverage(s["text"], entity) for s in item["sources"] if s["role"] != "user"),
        default=0.0,
    )
    # No threshold to tune: a member is either quoted whole in one of the user's own
    # turns, partly matched because the label was composed, or not in that turn at all.
    if user <= 0.0:
        return ("only_in_assistant_text" if other > 0 else "unsupported_in_user_text", user)
    if user < 1.0:
        return "label_partly_supported", user
    carriers = [
        m
        for m in item["memories"]
        if coverage(m.get("content", "") + " " + (m.get("object") or ""), entity) >= 1.0
    ]
    if carriers and not any(m.get("predicate") in approved for m in carriers):
        return "carried_by_unapproved_predicate", user
    return "quotable_under_an_approved_predicate", user


def decompose(packet: dict) -> dict:
    relations = packet["policy"]["relations"]
    questions, members = [], Counter()
    for item in packet["items"]:
        if item["kind"] != "entity":
            continue
        approved = set(relations.get(item["relation"], {}).get("predicates", []))
        rows = []
        for entity in item["proposed_entities"]:
            stage, cover = stage_of(entity, item, approved)
            rows.append({"entity": entity, "stage": stage, "user_coverage": round(cover, 2)})
        members.update(r["stage"] for r in rows)
        blocking = [r for r in rows if r["stage"] != STAGES[-1]]
        questions.append(
            {
                "id": item["id"],
                "relation": item["relation"],
                "proposed": len(rows),
                "first_blocking_stage": blocking[0]["stage"] if blocking else None,
                "members": rows,
                "flags": item["flags"],
            }
        )
    return {
        "schema_version": 1,
        "provider_calls": 0,
        "scope": "proposed members in the review packet; human verdicts are pending",
        "cannot_measure": "facts lost during extraction, and whether a supported member "
        "is the right answer to the question",
        "packet_sha256": packet["packet_sha256"],
        "totals": {
            "questions": len(questions),
            "proposed_members": sum(q["proposed"] for q in questions),
            "questions_with_a_blocking_member": sum(
                1 for q in questions if q["first_blocking_stage"]
            ),
            "questions_carrying_a_packet_flag": sum(1 for q in questions if q["flags"]),
        },
        "members_by_stage": {stage: members.get(stage, 0) for stage in STAGES},
        "questions_by_first_blocking_stage": {
            stage: sum(1 for q in questions if q["first_blocking_stage"] == stage)
            for stage in STAGES[:-1]
        },
        "questions": questions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, default=PACKET)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    report = decompose(json.loads(args.packet.read_text(encoding="utf-8")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("totals", "members_by_stage", "questions_by_first_blocking_stage")
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
