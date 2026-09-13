"""Propose a controlled relation vocabulary for the free-text `predicate` column.

The column is nominally the predicate of a triple and is in practice a label the
extractor invents per fact: 5,054 distinct values over 18,519 memories on
`train150`, 3,578 of them used exactly once, and 3,459 memories whose predicate is
the literal string `none`. The consequences are measurable — 81% of
`(user_id, subject, predicate)` keys hold a single memory, so they can never
supersede, and supersession fires on 0.41% of the store.

String normalisation is not the fix and was measured first: collapsing case,
separators, plurals and word order merges only 5% of the vocabulary. The values are
not spelling variants of each other, they are different inventions.

So this proposes a mapping instead. It costs nothing to run: the encoder is the
local MiniLM the project already loads, so no provider quota is spent, and the
output is a reviewable CSV rather than a migration. Nothing is written to any store.

**The output is a proposal, not a decision.** Every row carries the similarity that
produced it so a human can sort by confidence and correct the tail, and anything
below the threshold is parked in `other` with its original string preserved rather
than forced into a bucket it does not belong in.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Relation types, grouped by the property that actually matters downstream:
# how many values a key may hold at one time. That is what decides whether a new
# fact supersedes an old one or joins a set, and it is the thing the current
# free-text predicate cannot express.
#
# `single` — one value is true at a time; a new value closes the old interval.
# `set`    — values accumulate; "how many X" is an exhaustive scan, never a top-k.
# `event`  — a point in time; nothing is superseded, order is the question.
# `speech` — what the assistant said, filtered by source_role rather than subject.
VOCABULARY: dict[str, tuple[str, str]] = {
    # relation: (arity, description used for the embedding match)
    "lives_in": ("single", "where the person lives, their city, home or residence"),
    "works_as": ("single", "the person's job title, role, profession or occupation"),
    "works_at": ("single", "the employer, company or organisation the person works for"),
    "studies": ("single", "what the person is studying, their course or field of study"),
    "relationship_status": ("single", "marital or relationship status, partner, family status"),
    "health_status": ("single", "current health condition, diagnosis, symptom or treatment"),
    "current_goal": ("single", "an active goal, plan, target or intention being pursued"),
    "uses_tool": ("single", "the tool, framework, software or service currently used"),
    "prefers": ("single", "a stated preference, favourite, or what the person likes best"),
    "avoids": ("single", "something the person dislikes, avoids, is allergic to, or stopped"),
    "owns_pet": ("single", "a pet the person keeps"),
    "drives": ("single", "the vehicle the person owns or drives"),
    "team_size": ("single", "how many people the person manages or works with"),
    "acquired": ("set", "bought, purchased, received or acquired an object or possession"),
    "visited": ("set", "travelled to, visited or stayed at a place"),
    "attended": ("set", "attended an event, class, workshop, conference or ceremony"),
    "read": ("set", "read a book, article or publication"),
    "watched": ("set", "watched a film, show, series or video"),
    "listened_to": ("set", "listened to music, a podcast, an artist or a streaming service"),
    "ate_at": ("set", "ate at a restaurant, cafe or food place, or ordered food"),
    "used_service": ("set", "used a service, subscription, app or provider"),
    "completed": ("set", "finished, completed or achieved a task, project or course"),
    "practises_hobby": ("set", "a hobby, sport, craft or leisure activity practised"),
    "cooked": ("set", "cooked, baked or made a dish or recipe"),
    "grows": ("set", "plants, houseplants or a garden the person keeps"),
    "knows_person": ("set", "a friend, relative, colleague or acquaintance"),
    "event_occurred": ("event", "something that happened on a date in the past"),
    "planned_event": ("event", "an event scheduled or planned for the future"),
    "milestone": ("event", "a birthday, anniversary, graduation, wedding or life milestone"),
    "changed_state": ("event", "a switch, move, change or transition from one state to another"),
    "owns": ("set", "a possession the person has: clothing, gear, equipment, collection"),
    "routine": (
        "single",
        "a habit or recurring routine: morning routine, exercise schedule, wake time",
    ),
    "financial_activity": ("set", "spending, saving, budgeting, investing or a financial decision"),
    "dietary_rule": (
        "single",
        "a diet, food restriction, allergy or eating rule the person follows",
    ),
    "recommended": ("speech", "the assistant recommended or suggested something"),
    "advised_against": ("speech", "the assistant warned against or advised not to do something"),
    "explained": ("speech", "the assistant explained, described or gave information about a topic"),
    "committed_to": ("speech", "the assistant promised, committed or agreed to do something"),
}


def _readable(predicate: str) -> str:
    """`book_recommendations` -> `book recommendations`, so the encoder sees words."""
    return predicate.replace("_", " ").replace("-", " ").strip()


# Morphological families the encoder handles badly and a human reads instantly.
# `book_recommendation` scored 0.439 against `recommended` — below any threshold
# that also keeps the tail honest — because MiniLM is comparing a two-word label
# with a sentence, and the shared head noun carries almost no weight. These are not
# exceptions bolted on to raise a number: the head of the compound *is* the
# relation, and matching on it is more reliable than a similarity score, so the
# rules run first and the embedding only sees what they do not catch.
_HEAD_RULES: tuple[tuple[str, str], ...] = (
    ("recommendation", "recommended"),
    ("recommendations", "recommended"),
    ("suggestion", "recommended"),
    ("advice", "recommended"),
    ("owned", "owns"),
    ("collection", "owns"),
    ("collectibles", "owns"),
    ("gear", "owns"),
    ("equipment", "owns"),
    ("wardrobe", "owns"),
    ("routine", "routine"),
    ("habit", "routine"),
    ("schedule", "routine"),
    ("purchase", "acquired"),
    ("purchases", "acquired"),
    ("bought", "acquired"),
    ("spending", "financial_activity"),
    ("budget", "financial_activity"),
    ("savings", "financial_activity"),
    ("plan", "planned_event"),
    ("plans", "planned_event"),
    ("history", "event_occurred"),
    ("goal", "current_goal"),
    ("goals", "current_goal"),
)


def _rule_match(predicate: str) -> str | None:
    """Match on the head word of the compound, which is where the relation lives."""
    parts = predicate.lower().replace("-", "_").split("_")
    if not parts:
        return None
    for suffix, relation in _HEAD_RULES:
        if parts[-1] == suffix:
            return relation
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default="stores/train150.db")
    parser.add_argument("--out", default="results/analysis/predicate-map.csv")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="below this cosine, park the predicate in `other` rather than guess",
    )
    args = parser.parse_args()

    connection = sqlite3.connect(f"file:{REPO / args.store}?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT predicate, count(*) FROM memories GROUP BY predicate ORDER BY count(*) DESC"
    ).fetchall()
    connection.close()

    from llm_long_term_memory.embed import Encoder

    encoder = Encoder()
    names = list(VOCABULARY)
    targets = encoder.encode([f"{n.replace('_', ' ')}: {VOCABULARY[n][1]}" for n in names])
    targets = targets / (targets**2).sum(axis=1, keepdims=True) ** 0.5

    predicates = [p for p, _ in rows]
    vectors = encoder.encode([_readable(p) for p in predicates])
    vectors = vectors / (vectors**2).sum(axis=1, keepdims=True) ** 0.5
    scores = vectors @ targets.T

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assigned: Counter[str] = Counter()
    memories_by_arity: Counter[str] = Counter()
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["predicate_raw", "memories", "relation_type", "arity", "similarity"])
        for index, (predicate, count) in enumerate(rows):
            ruled = _rule_match(predicate)
            best = int(scores[index].argmax())
            similarity = float(scores[index][best])
            # `none` is not a weak match, it is the extractor declining to key the
            # fact at all. Mapping it by similarity would invent a relation the
            # model never asserted.
            if predicate in {"none", ""}:
                # Not a weak match: the extractor declined to key the fact at all.
                # Mapping it by similarity would invent a relation nobody asserted.
                relation, arity, similarity = "other", "unkeyed", 0.0
            elif ruled is not None:
                relation, arity = ruled, VOCABULARY[ruled][0]
            elif similarity < args.threshold:
                relation, arity = "other", "unknown"
            else:
                relation, arity = names[best], VOCABULARY[names[best]][0]
            assigned[relation] += 1
            memories_by_arity[arity] += count
            writer.writerow([predicate, count, relation, arity, f"{similarity:.3f}"])

    total_memories = sum(c for _, c in rows)
    print(f"predicates mapped : {len(rows):,}  ->  {len(set(assigned) - {'other'})} relation types")
    print(f"parked in `other` : {assigned['other']:,} predicates")
    print(f"\nmemories by arity (of {total_memories:,}):")
    for arity, count in memories_by_arity.most_common():
        print(f"  {arity:<10}{count:>8,}  {count / total_memories:>6.1%}")
    print("\ntop relation types by predicate count:")
    for relation, count in assigned.most_common(12):
        arity = VOCABULARY.get(relation, ("unknown",))[0]
        print(f"  {relation:<22}{arity:<9}{count:>6} predicates")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
