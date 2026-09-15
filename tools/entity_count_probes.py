"""Count probes over *entities*, and a refusal to ask a question the store cannot answer.

The previous count generator produced unreliable membership labels. Re-reading the paid
v4 rows against a provisional model-made overlay moved the v4.2 arm difference from +1 to
-5; that overlay still awaits human review and does not replace the registered result.
The rules below reject some known defects. Relation mappings, name splits and membership
still need semantic review before the generated probes can measure an answerer.

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

Zero provider calls. Proposed gold is SQL plus deterministic text handling over extracted
facts; determinism and content fingerprints establish reproducibility, not semantic truth.
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

# Objects the splitter must not be trusted on. The comment above has always said an
# uncertain split is refused; until the split audit these were the four shapes where it
# silently was not, and every one of them put a non-member into a shipped gold answer.
#
# Month names are whole words. The first version matched any word that merely began with a
# month's three letters, so 'decorating', 'Marketing' and 'novels' were dates: none of them
# reached the shipped refusals, but a rule that calls a hobby a date refuses a sound probe on
# the next store and records the wrong reason for it.
_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b\.?"
)
_ORDINAL = r"(?:st|nd|rd|th)?"
# An object that ends in when it happened is a record, not a list. `'MoMA, Dec 2023'` gave
# "how many places has the user visited?" a month as a place; `'TechFest, San Francisco,
# February 2023'` gave "how many events attended?" a city and a month. Refused whole rather
# than repaired by dropping the date: once an object is known to carry provenance, its
# other commas are provenance too, and 'San Francisco' is not an event either.
_DATE_MEMBER = re.compile(
    rf"^(?:{_MONTH}\s*(?:\d{{1,2}}{_ORDINAL},?\s*)?(?:(?:19|20)\d{{2}})?|(?:19|20)\d{{2}}|"
    # A decade split off a shared noun: '1970s and 1980s cameras' made '1970s' a possession.
    rf"(?:19|20)\d0s|"
    rf"\d{{1,2}}{_ORDINAL}\s+(?:of\s+)?{_MONTH}(?:\s*(?:19|20)\d{{2}})?|"
    rf"(?:19|20)\d{{2}}-\d{{2}}-\d{{2}})$",
    re.IGNORECASE,
)
# Provenance written into a single object instead of split off by a separator. The rules
# here used to look only at members a separator produced, so a record that never split
# passed whole: the second emission of the set counted 'January 15 2023' as a place
# visited, one city twice through two itinerary lines ('arrive Calgary June 14 2023 11:30
# am'), one pair of sneakers twice through two receipts ('sneakers on 2023-04-10 for $80
# from Amazon'), and 'raised $150' as an event attended. A day or a year after a month, an
# ISO date, a clock time or a price marks a record of an occasion, and two records of one
# occasion never deduplicate. A leading model year does not: '1995 Honda Civic' names a car.
_PROVENANCE = re.compile(
    rf"\b{_MONTH}\s*\d{{1,2}}{_ORDINAL}\b|\b{_MONTH},?\s*(?:19|20)\d{{2}}\b|"
    rf"\b\d{{1,2}}{_ORDINAL}\s+(?:of\s+)?{_MONTH}|\b(?:19|20)\d{{2}}-\d{{2}}-\d{{2}}\b|"
    r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*(?:am|pm)\b|[$€£¥]\s?\d",
    re.IGNORECASE,
)
# A fragment that is the tail of a sentence, not a noun. `'attended on March 22nd, 2023'`
# is not two events; it is not even one entity.
_CLAUSE_MEMBER = re.compile(
    r"^(?:attended|held|hosted|visited|located|organi[sz]ed|purchased|bought|read|on|in|"
    r"at|for|since|from|during|to|by)\b",
    re.IGNORECASE,
)
# A work with its author attached. `'Milk and Filth by Carmen Giménez Smith'` is one poetry
# collection whose title contains "and"; the split made it two books. Any byline means the
# object names a titled work, so its internal separators belong to the title.
_BYLINE = re.compile(r"\bby\s+[A-Z]", re.UNICODE)
# Companions, not members. `'basketball with Tom and Alex'` is one hobby and two people;
# the split counted Alex as a hobby.
_COMPANION = re.compile(r"\bwith\s+\S", re.IGNORECASE)


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


def split_refusal(memory_object: str | None) -> str | None:
    """Why this object's split cannot be trusted, or None if it can.

    Separate from `entities_of` on purpose. `entities_of` answers "what does this object
    say", which the split audit needs to report on objects this function rejects; this
    answers "may a gold answer be built on that", which only the generator needs. Keeping
    them apart is what lets the audit show a reviewer the bad split it refused.
    """
    raw = (memory_object or "").strip()
    parts = entities_of(raw)
    if len(parts) < 2:
        # Nothing was split, so there is no split to distrust — but the object can still be
        # a clause rather than a thing, or a record of an occasion rather than a name. A bare
        # year is not refused here: '1984' is a book.
        if parts and _CLAUSE_MEMBER.match(parts[0]):
            return "reads as a clause, not a thing"
        if _PROVENANCE.search(raw):
            return (
                "carries its own date, time or price, so it records an occasion rather than "
                "naming a member"
            )
        return None
    if _BYLINE.search(raw):
        return "carries a byline, so its separators belong to a title"
    if _COMPANION.search(raw):
        return "names companions, which are not members of the asked-about set"
    for part in parts:
        if _DATE_MEMBER.match(part):
            return f"splits into a date ({part!r}), so the object records provenance"
        if _CLAUSE_MEMBER.match(part):
            return f"splits into a clause fragment ({part!r}), not an entity"
    if _PROVENANCE.search(raw):
        return "carries its own date, time or price, so its separators are provenance too"
    return None


def scope_conflicts(predicates: list[str]) -> list[str]:
    """Predicates whose names would make a completed-act question contradict itself."""
    # Underscores are normalised to spaces first. A predicate arrives here either raw
    # (`recipe_to_make`) or printed (`recipe to make`), and `_` is a word character, so
    # `\bto` never matches the raw form — the phrase check silently passed it.
    return sorted(p for p in predicates if _is_intent(p.lower().replace("_", " ")))


def _is_intent(text: str) -> bool:
    return bool(_INTENT_PREDICATE.search(text) or _INTENT_PHRASE.search(text))


def _event_time(memory: Any) -> datetime | None:
    """A member's event time, or None where the row cannot supply one.

    `sqlite3.Row` raises rather than returning None for a column the query did not select,
    so a caller that forgot `event_time` must be told, not silently handed an unbounded
    count. Both that case and a NULL column come back as None here and become a refusal.
    """
    try:
        raw = memory["event_time"]
    except (IndexError, KeyError):
        return None
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw
    try:
        return datetime.fromisoformat(str(raw))
    except ValueError:
        return None


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

    if window is not None:
        # The window used to reach only the question text. It said "count only facts dated
        # between …" while the answer stayed the count of every member, so any probe built
        # with one carried a gold answer that contradicted its own question. Filtering here
        # is what makes the sentence true; a row that cannot be dated is refused, because
        # counting it would answer a time-bounded question with an undated fact.
        start, end = window
        dated = []
        for memory in completed:
            when = _event_time(memory)
            if when is None:
                raise Refusal(f"member {memory['id']} has no event time to bound")
            if start <= when <= end:
                dated.append(memory)
        completed = dated
        if not completed:
            raise Refusal(f"no member falls inside {start:%Y-%m-%d}..{end:%Y-%m-%d}")

    predicates = sorted({(m["predicate"] or "").replace("_", " ") for m in completed})
    conflicts = scope_conflicts(predicates)
    if conflicts:
        # The old set shipped exactly this: "how many dishes has the user cooked? Count
        # only facts recorded under: recipe interest, recipe to make, recipes tried."
        raise Refusal(f"completed-act question over intent predicates {conflicts}")

    by_key: dict[str, dict[str, Any]] = {}
    for memory in completed:
        # Before the members are counted, not after: a gold answer built on a bad split is
        # wrong in a way no later check can see, because the count and the entity list
        # agree with each other. This is the check the module docstring always claimed.
        untrustworthy = split_refusal(memory["object"])
        if untrustworthy:
            raise Refusal(f"member {memory['id']} {untrustworthy}")
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
            + (
                # Said in the derivation because it limits what a time-bounded answer can
                # mean: the engine's event time mostly inherits the session date, so this
                # counts facts stated in the window, not acts that happened in it.
                f"; bounded to event_time in [{window[0]:%Y-%m-%d}, {window[1]:%Y-%m-%d}], "
                f"which inherits the session date rather than a parsed event date"
                if window is not None
                else ""
            )
        ),
    }


def _fingerprint(connection: sqlite3.Connection) -> str:
    """Bind fact and provenance contents, including committed WAL changes.

    Row counts miss edits to an existing fact, scope or source. Hash logical rows from
    the same read transaction used to generate the probes, in a stable order; neither
    physical SQLite layout nor insertion order is part of the evidence identity.
    """
    digest = hashlib.sha256(b"sqlite-evidence-content-v1\n")
    for table in ("memories", "turns", "sessions", "evidence", "entities", "memory_entities"):
        # Names come only from the fixed inventory above, never from a caller.
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not exists:
            if table in {"memories", "turns", "sessions"}:
                raise ValueError(f"missing evidence table: {table}")
            digest.update(f"{table}:absent\n".encode())
            continue
        columns = [
            item[0] for item in connection.execute(f'SELECT * FROM "{table}" LIMIT 0').description
        ]
        header = json.dumps([table, columns], ensure_ascii=False, separators=(",", ":"))
        digest.update((header + "\n").encode("utf-8"))
        order = ", ".join(str(i) for i in range(1, len(columns) + 1))
        for row in connection.execute(f'SELECT * FROM "{table}" ORDER BY {order}'):
            values = [
                value if not isinstance(value, bytes) else {"blob_hex": value.hex()}
                for value in row
            ]
            encoded = json.dumps(values, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            digest.update((encoded + "\n").encode("utf-8"))
    return digest.hexdigest()


def generate(store: Path, relation_map: Path, seed: int, limit: int) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        # Without a transaction, facts could be read before an edit and fingerprinted
        # after it. SQLite's read snapshot also includes committed WAL records.
        connection.execute("BEGIN")
        return _generate(connection, store, relation_map, seed, limit)
    finally:
        connection.close()


def _generate(
    connection: sqlite3.Connection, store: Path, relation_map: Path, seed: int, limit: int
) -> dict[str, Any]:
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
            probe = build_probe(user_id, relation, groups[(user_id, relation)])
        except Refusal as why:
            refusals.append({"namespace": user_id, "relation": relation, "reason": str(why)})
        else:
            # Named for what it asks, not for where it landed in this emission. Ids used to be
            # positions, so every refusal a rule change added renumbered the probes after it —
            # one regeneration changed the id of 49 of the 55 probes it kept — and a review
            # decision recorded against an id would attach to a different question the next
            # time the set was built. Namespace and relation are the probe's identity already.
            probe["probe_id"] = f"entity_count_{user_id}_{relation}"
            probe["relation"] = relation
            probes.append(probe)
        if len(probes) >= limit:
            break

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
        "split_rules": (
            "An object whose split yields a date, a clause fragment, a byline or a "
            "companion is refused whole rather than repaired, because once an object is "
            "known to carry provenance its other separators are provenance too. The first "
            "emission of this file predated the rule and shipped four gold answers that "
            "counted a month as a place visited or a city as an event attended; "
            "tools/audit_entity_probe_splits.py found them and tests/"
            "test_entity_count_probes.py pins each one. The second emission checked "
            "provenance only on members a separator produced, so a record that never split "
            "passed whole and three more gold answers counted a vet visit's date as a place, "
            "one city and one pair of sneakers twice, and a sum raised as an event; an "
            "object carrying a dated day or month, an ISO date, a clock time or a price is "
            "now refused whether or not it splits. Splits that survive the rules are "
            "still only machine-checked: 'Daisy Jones and The Six' is indistinguishable "
            "from two real titles, and no rule here decides it."
        ),
        "provider_calls": 0,
        "seed": seed,
        "limit": limit,
        "store": store.name,
        "store_fingerprint_kind": "sqlite-evidence-content-v1",
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
    try:
        shown = args.out.resolve().relative_to(REPO)
    except ValueError:
        shown = args.out
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
