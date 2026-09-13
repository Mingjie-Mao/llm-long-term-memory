"""Count probes over *entities*, and a refusal to ask a question the store cannot answer.

The previous count generator produced a question set that inverted a sign. Re-reading the
paid v4 rows against a corrected gold moved the v4.2 arm difference from +1 to -5, and the
model turned out to have been right where the probe said it was wrong. Four defects caused
it, and each one is refused or fixed here rather than patched per probe.

**Intentions are not members of a set of completed acts.** `COUNT` over a relation swept in
`scope='plan'` and `scope='preference'` rows, so "is looking for thriller recommendations"
counted as a book read and "hardening off seedlings" as a plant grown. 19 of 30 development
probes carried at least one. Membership is now decided by scope, and the two intent scopes
are excluded from every completed-act question.

**One row is not one member.** "has been reading The Huffington Post **and** Politico"
is one memory and two publications; the old gold counted it once. Members are now entities,
extracted from the object and deduplicated, so a row carrying two contributes two and two
rows naming the same thing contribute one.

**A question must not contradict itself.** The old generator pasted the raw predicate names
into the question as its scope, which produced *"How many distinct dishes has the user
cooked? Count only facts recorded under: recipe interest, recipe to make, recipes tried."* —
a completed-act verb over a scope listing intentions. Such a question is refused, not
emitted with a note.

**A question needs evidence a reader could check.** Where an entity cannot be recovered
from a member's object, the probe is refused rather than guessed at, because a gold answer
nobody can verify is worse than one question fewer.

`synthesis-probes.json` is untouched: its bytes are hashed into the held-out split, and the
split is the only unseen check v4 has. This writes a separate set.

Zero provider calls. Ground truth is SQL plus deterministic text handling, never a label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent

# A completed-act question counts what happened or what is held. `plan` records an
# intention and `preference` a taste; neither asserts membership. `commitment` is excluded
# too: "needs to reorder the medication tomorrow" is a promise, not a completed task.
COMPLETED_SCOPES = frozenset({"event", "profile"})
INTENT_SCOPES = frozenset({"plan", "preference", "commitment", "recommendation"})

# Questions this generator knows how to ask, with the tense each one requires. Every one
# asks for something already done or currently held, so every one refuses intent scopes.
PHRASING = {
    "acquired": "things has the user acquired",
    "owns": "possessions does the user have on record",
    "visited": "places has the user visited",
    "attended": "events has the user attended",
    "read": "books or articles has the user read",
    "watched": "films or shows has the user watched",
    "listened_to": "music or podcasts has the user listened to",
    "ate_at": "restaurants or food places has the user eaten at",
    "used_service": "services has the user used",
    "completed": "tasks or projects has the user completed",
    "practises_hobby": "hobbies or activities does the user practise",
    "cooked": "dishes has the user cooked",
    "grows": "plants does the user grow",
    "knows_person": "people does the user know",
}

# Predicate names that describe an intention. A completed-act question whose scope would
# list one of these is self-contradictory and is refused. Matched on the raw predicate,
# because that is what the question would print.
# Listed explicitly rather than stemmed. A `plan\w*` pattern also matches "plants", so
# `garden plants` — a perfectly good completed-act predicate for "how many plants does the
# user grow" — would be refused as an intention. Over-refusal is quieter than
# over-inclusion and therefore more dangerous.
_INTENT_WORDS = (
    "plan", "plans", "planned", "planning",
    "interest", "interests", "interested",
    "want", "wants", "wanted",
    "wish", "wishes", "wishlist",
    "considering", "intend", "intends", "intent", "intention", "intentions",
    "upcoming", "scheduled", "goal", "goals", "aim", "aims",
    "recommendation", "recommendations",
)  # fmt: skip
_INTENT_PREDICATE = re.compile(r"\b(" + "|".join(_INTENT_WORDS) + r")\b")
# Predicates are stored snake_case and printed with spaces, so both must match.
_INTENT_PHRASE = re.compile(r"\bto[\s_](make|read|watch|try|visit|buy|do|see)\b")

# Splitting an object into entities. Only separators that are unambiguous in this corpus:
# a comma list and the word "and". `&` is excluded because it is far more often part of a
# name — "Simon & Schuster", "Barnes & Noble" — than a list separator, and a wrong split
# invents a member. Deliberately not a general parser; where the split is uncertain the
# probe is refused instead.
_SPLIT = re.compile(r"\s*(?:,|;|\band\b)\s+")
# A member that is a measurement rather than a thing: "55 pages per day" is a reading rate,
# not a book. These reached the old gold as members.
_MEASUREMENT = re.compile(
    r"^\d+(\.\d+)?\s*(pages?|hours?|minutes?|days?|weeks?|months?|years?|km|miles?|kg|"
    r"lbs?|percent|%)\b",
    re.IGNORECASE,
)
_ARTICLE = re.compile(r"^(the|a|an|some|several|various|my|their|his|her)\s+", re.IGNORECASE)
_NOISE = re.compile(r"[^\w\s'-]")


class Refusal(Exception):
    """The question cannot be asked honestly. Carries the reason for the record."""


def normalise_entity(text: str) -> str:
    """A deduplication key. Case, articles and surrounding punctuation only.

    Deliberately blunt. Stemming or similarity would merge genuinely different members,
    and over-merging is indistinguishable from the under-counting this set exists to
    measure.
    """
    cleaned = _NOISE.sub(" ", (text or "").lower())
    cleaned = _ARTICLE.sub("", cleaned.strip())
    return " ".join(cleaned.split())


def entities_of(memory_object: str | None) -> list[str]:
    """The entities one stored object names.

    Returns several for "The Huffington Post and Politico", one for "Spaghetti Carbonara",
    and none where the object is a measurement or is empty — which the caller turns into a
    refusal rather than a silent zero.
    """
    raw = (memory_object or "").strip()
    if not raw or _MEASUREMENT.match(raw):
        return []
    parts = [part.strip() for part in _SPLIT.split(raw) if part.strip()]
    # A split that produces a fragment too short to be a thing means the separator was
    # part of a name ("Simon & Schuster"), so the object is kept whole.
    if any(len(normalise_entity(part)) < 3 for part in parts):
        return [raw]
    return parts or [raw]


def scope_conflicts(predicates: list[str]) -> list[str]:
    """Predicates whose names would make a completed-act question contradict itself."""
    # Underscores are normalised to spaces first. A predicate arrives here either raw
    # (`recipe_to_make`) or printed (`recipe to make`), and `_` is a word character, so
    # `\bto` never matches the raw form — the phrase check silently passed it.
    return sorted(p for p in predicates if _is_intent(p.lower().replace("_", " ")))


def _is_intent(text: str) -> bool:
    return bool(_INTENT_PREDICATE.search(text) or _INTENT_PHRASE.search(text))


def build_probe(
    user_id: str,
    relation: str,
    members: list[sqlite3.Row],
    *,
    window: tuple[datetime, datetime] | None = None,
) -> dict[str, Any]:
    """One probe, or a `Refusal` naming what made the question unanswerable."""
    if relation not in PHRASING:
        raise Refusal(f"no vetted phrasing for relation {relation!r}")

    completed = [m for m in members if (m["scope"] or "") in COMPLETED_SCOPES]
    if not completed:
        raise Refusal("every candidate member records an intention rather than an act")

    predicates = sorted({(m["predicate"] or "").replace("_", " ") for m in completed})
    conflicts = scope_conflicts(predicates)
    if conflicts:
        # The old set shipped exactly this: "how many dishes has the user cooked? Count
        # only facts recorded under: recipe interest, recipe to make, recipes tried."
        raise Refusal(f"completed-act question over intent predicates {conflicts}")

    by_key: dict[str, dict[str, Any]] = {}
    for memory in completed:
        found = entities_of(memory["object"])
        if not found:
            raise Refusal(f"member {memory['id']} carries no recoverable entity")
        for entity in found:
            key = normalise_entity(entity)
            if not key:
                raise Refusal(f"member {memory['id']} yields an empty entity key")
            slot = by_key.setdefault(key, {"entity": entity, "memory_ids": []})
            slot["memory_ids"].append(memory["id"])

    if len(by_key) < 3:
        raise Refusal(f"only {len(by_key)} distinct entities after deduplication")
    if len(by_key) > 15:
        raise Refusal(f"{len(by_key)} entities is past the size a reader can check")

    window_clause = ""
    if window is not None:
        start, end = window
        window_clause = f" Count only facts dated between {start:%Y-%m-%d} and {end:%Y-%m-%d}."

    return {
        "kind": "entity_count",
        "namespace": user_id,
        "question": (
            f"How many distinct {PHRASING[relation]}? "
            f"Count only facts recorded under: {', '.join(predicates)}."
            f"{window_clause} Count each distinct thing once, however many times it is "
            f"mentioned."
        ),
        "answer": len(by_key),
        "answer_kind": "integer",
        "scope_predicates": predicates,
        "entities": [by_key[k]["entity"] for k in sorted(by_key)],
        "evidence_memory_ids": sorted({mid for v in by_key.values() for mid in v["memory_ids"]}),
        "derivation": (
            f"COUNT of DISTINCT entities over active memories where subject='user', "
            f"the mapped relation_type is {relation!r}, and scope is one of "
            f"{sorted(COMPLETED_SCOPES)}; entities split from the stored object and "
            f"deduplicated case-insensitively; scope stated in the question as {predicates}"
        ),
    }


def _fingerprint(connection: sqlite3.Connection) -> str:
    parts = [
        str(connection.execute("SELECT count(*) FROM memories").fetchone()[0]),
        str(connection.execute("SELECT count(*) FROM turns").fetchone()[0]),
        str(connection.execute("SELECT count(*) FROM sessions").fetchone()[0]),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def generate(store: Path, relation_map: Path, seed: int, limit: int) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    with open(relation_map, encoding="utf-8", newline="") as handle:
        relations = {row["predicate_raw"]: row["relation_type"] for row in csv.DictReader(handle)}

    groups: dict[tuple[str, str], list[sqlite3.Row]] = defaultdict(list)
    for row in connection.execute(
        "SELECT id, user_id, predicate, object, scope, status, subject FROM memories"
    ):
        if row["status"] != "active" or row["subject"] != "user":
            continue
        relation = relations.get(row["predicate"] or "")
        if relation in PHRASING:
            groups[(row["user_id"], relation)].append(row)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    probes: list[dict[str, Any]] = []
    refusals: list[dict[str, str]] = []
    for user_id, relation in keys:
        try:
            probes.append(build_probe(user_id, relation, groups[(user_id, relation)]))
        except Refusal as why:
            refusals.append({"namespace": user_id, "relation": relation, "reason": str(why)})
        if len(probes) >= limit:
            break

    for index, probe in enumerate(probes):
        probe["probe_id"] = f"entity_count_{index:04d}"

    return {
        "schema_version": 1,
        "not_a_benchmark": (
            "Ground truth is SQL over stored facts plus deterministic entity handling, so "
            "it inherits extraction loss. These are instrument readings and must never be "
            "reported beside a LongMemEval accuracy figure."
        ),
        "supersedes": (
            "The `count` probes in synthesis-probes.json, whose gold counted intentions as "
            "members and one row as one member. That file is unchanged: its bytes are "
            "hashed into the held-out split."
        ),
        "provider_calls": 0,
        "seed": seed,
        "limit": limit,
        "store": store.name,
        "store_fingerprint": _fingerprint(connection),
        "relation_map_sha256": hashlib.sha256(relation_map.read_bytes()).hexdigest(),
        "completed_scopes": sorted(COMPLETED_SCOPES),
        "intent_scopes": sorted(INTENT_SCOPES),
        "probes": probes,
        "refused": refusals,
        "refusal_reasons": _reason_counts(refusals),
    }


def _reason_counts(refusals: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in refusals:
        counts[item["reason"].split(" ")[0] + " …"] += 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=REPO / "stores/train150.db")
    parser.add_argument(
        "--relation-map", type=Path, default=REPO / "results/analysis/predicate-map.csv"
    )
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument(
        "--out", type=Path, default=REPO / "results/analysis/entity-count-probes.json"
    )
    args = parser.parse_args()

    if not args.store.is_file():
        print(f"STOP: no store at {args.store}")
        return 2
    payload = generate(args.store, args.relation_map, args.seed, args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"generated : {len(payload['probes'])} entity-count probes")
    print(f"refused   : {len(payload['refused'])}")
    for reason, count in payload["refusal_reasons"].items():
        print(f"  {count:4d}  {reason}")
    print(f"\nwrote {args.out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
