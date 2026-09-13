"""Generate synthesis probes whose ground truth is a SQL query, not a label.

The v4 target is arithmetic and enumeration over evidence the system already holds.
Measuring that on LongMemEval gives two to four samples per operation, which is where
the failure taxonomy ran out of resolution — every interval it produced was twenty to
forty points wide. These probes exist to raise the sample size for one narrow question
at a time.

**Ground truth is derived, never labelled.** A count probe's answer is a `COUNT` over
stored facts; a duration probe's is a subtraction of two `event_time` values. No model
is asked what the answer is, so there is no judge to disagree with and no label to be
wrong in the same direction as the system under test.

**What a probe score is not.** It measures synthesis *conditional on the store being
right*. Where extraction lost a fact, the probe's ground truth inherits that loss. A
number from here is therefore not accuracy, must never appear beside a LongMemEval
figure, and cannot be reported as a benchmark result — see
`results/prereg-v4-data-protocol.md`.

**Regenerated, not curated.** A probe set is a deterministic function of
`(seed, store fingerprint, per_kind)`. `--verify` regenerates and compares, so a set
that was quietly edited after someone saw a result fails to reproduce.
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

# How each `set`-arity relation is spoken about in a question. Templates rather than
# a model: a generator that called an LLM would put the thing under test inside the
# instrument. The cost is that phrasing variety is low, which is stated as a limit
# rather than hidden — these probes isolate an operation, they do not simulate users.
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
    "financial_activity": "financial activities are on record for the user",
}


_MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
_MONTHS = {m.lower(): i for i, m in enumerate(_MONTH_NAMES, 1)}
_MONTH_RE = "|".join(_MONTHS)


def _stated_months(text: str) -> set[tuple[int, int]]:
    """Year/month pairs written in the memory's own text."""
    found = set()
    for match in re.finditer(
        rf"\b({_MONTH_RE})\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", text, re.IGNORECASE
    ):
        found.add((int(match.group(2)), _MONTHS[match.group(1).lower()]))
    for match in re.finditer(rf"\b({_MONTH_RE})\s+(\d{{4}})\b", text, re.IGNORECASE):
        found.add((int(match.group(2)), _MONTHS[match.group(1).lower()]))
    for match in re.finditer(r"\b(\d{4})-(\d{2})-\d{2}\b", text):
        found.add((int(match.group(1)), int(match.group(2))))
    return found


def _anchor_disagrees(row) -> bool:
    """Does the stored `event_time` contradict a date written in the same memory?

    The first probe set kept a memory reading "watched Parasite on March 17th, 2021"
    whose `event_time` was `2021-12-10` — the session date, not the event date. The
    arithmetic over `event_time` was right and the question was unanswerable: a model
    reading the text computes from March and is marked wrong. Measured on the first
    set: 22% of duration and comparison probes carried at least one such anchor.

    Only an explicit month-and-year in the text counts. A memory that states no date
    is fine — nothing in it contradicts the stored one.
    """
    stated = _stated_months(row["content"] or "")
    if not stated:
        return False
    when = _date(row["event_time"])
    return when is not None and (when.year, when.month) not in stated


# Relations a natural-language count question cannot hold apart. "How many
# possessions" reads on `owns` and `acquired` alike, and no phrasing of the
# question tells the answerer which one the ground truth meant. Where a namespace
# holds both, the group is dropped rather than phrased around: the first probe set
# scored the answerer wrong for counting furniture recorded under `acquired` when
# the truth counted only `owns`.
CONFUSABLE = [
    frozenset({"owns", "acquired"}),
    frozenset({"visited", "attended"}),
    frozenset({"ate_at", "cooked"}),
    frozenset({"completed", "practises_hobby"}),
    frozenset({"read", "watched", "listened_to"}),
]


def _rivals(relation: str) -> set[str]:
    """Relations that would read as members of the same question."""
    return {r for family in CONFUSABLE if relation in family for r in family} - {relation}


def _fingerprint(connection: sqlite3.Connection) -> str:
    """Identify the store a probe set was derived from.

    Row counts plus the recorded ingest fingerprint, hashed. A probe set carried to a
    different store has to fail loudly rather than silently measure something else.
    """
    parts = [
        str(connection.execute("SELECT count(*) FROM memories").fetchone()[0]),
        str(connection.execute("SELECT count(*) FROM turns").fetchone()[0]),
        str(connection.execute("SELECT count(*) FROM sessions").fetchone()[0]),
    ]
    for key in ("ingest_fingerprint", "extractor_version"):
        row = connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        parts.append(row[0] if row else "")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _relation_map(path: Path) -> dict[str, tuple[str, str]]:
    with open(path, encoding="utf-8") as handle:
        return {
            row["predicate_raw"]: (row["relation_type"], row["arity"])
            for row in csv.DictReader(handle)
        }


def _memories(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    connection.row_factory = sqlite3.Row
    return connection.execute(
        "SELECT id, user_id, subject, source_role, predicate, object, content, event_time, status "
        "FROM memories ORDER BY id"
    ).fetchall()


def _date(value: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(value) if value else None
    except ValueError:
        return None


def _count_probes(rows, relations, rng, limit) -> list[dict[str, Any]]:
    """How many members does one set-valued relation hold for one subject?

    Two rules make the question answerable, both added after the first set was read
    against v3.3 and both costing no provider call:

    **The question names its own scope.** Ground truth is a `COUNT` over one mapped
    relation, but the store reaches that relation through several raw predicates and
    the plain question named none of them. Asked "how many distinct possessions",
    the answerer counted everything possession-shaped in its context and was marked
    wrong against a narrower set. The predicates are now listed in the question, so
    the boundary the truth uses is the boundary the answerer is given.

    **A namespace holding a confusable relation is skipped.** Naming predicates
    fixes the boundary but not the ambiguity of the noun; see `CONFUSABLE`.

    **Only the user's own facts are counted.** Every phrasing in `PHRASING` asks
    what *the user* did, and six of the first sixty probes counted facts whose
    subject was the assistant — the same mismatch the `_dated` filter was written
    for, arriving through a different door.
    """
    groups: dict[tuple[str, str], list[sqlite3.Row]] = defaultdict(list)
    present: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["status"] != "active" or row["subject"] != "user":
            continue
        relation, arity = relations.get(row["predicate"], ("other", "unknown"))
        present[row["user_id"]].add(relation)
        if arity == "set" and relation in PHRASING:
            groups[(row["user_id"], relation)].append(row)
    # Three is the smallest size where miscounting is distinguishable from an
    # off-by-one on a pair, and it is where the observed failures actually sat.
    usable = sorted(
        (key, members)
        for key, members in groups.items()
        if 3 <= len(members) <= 15 and not (_rivals(key[1]) & present[key[0]])
    )
    rng.shuffle(usable)
    probes = []
    for (user_id, relation), members in usable[:limit]:
        labels = sorted({m["predicate"].replace("_", " ") for m in members})
        probes.append(
            {
                "kind": "count",
                "namespace": user_id,
                "question": (
                    f"How many distinct {PHRASING[relation]}? "
                    f"Count only facts recorded under: {', '.join(labels)}."
                ),
                "answer": len(members),
                "answer_kind": "integer",
                "scope_predicates": labels,
                "evidence_memory_ids": sorted(m["id"] for m in members),
                "derivation": (
                    f"COUNT of active memories where subject='user' and the mapped "
                    f"relation_type is {relation!r}; scope stated in the question as "
                    f"{labels}"
                ),
            }
        )
    return probes


def _dated(rows, relations) -> dict[str, list[sqlite3.Row]]:
    """Dated facts that are actually *events*, per namespace.

    Filtering matters here and was added after reading the first output. Taking any
    dated memory produced pairs like "which happened first: Abraham Lincoln, Oprah
    Winfrey and Steve Jobs are famous ambiverts, or ...", which is not a question
    about ordering at all — an assistant's general statement carries the session date
    in `event_time`, not the date of anything that happened. The arithmetic was sound
    and the question was nonsense, which is the worst combination for a probe because
    a wrong answer would look like a reasoning failure.

    So: facts the *user* stated, on relations whose arity means something occurred,
    and whose stored date does not contradict a date written in their own text
    (see `_anchor_disagrees`).
    """
    by_namespace: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        if row["status"] != "active" or not _date(row["event_time"]):
            continue
        if row["source_role"] != "user":
            continue
        if relations.get(row["predicate"], ("other", "unknown"))[1] not in {"set", "event"}:
            continue
        if _anchor_disagrees(row):
            continue
        by_namespace[row["user_id"]].append(row)
    return by_namespace


def _duration_probes(rows, relations, rng, limit) -> list[dict[str, Any]]:
    """How many days between two dated facts? Arithmetic the store can check.

    **The subtraction is over dates, not timestamps.** `event_time` carries a time of
    day, but the answerer never sees one: `render_memory` formats every anchor as
    `%Y-%m-%d`. Flooring a timestamp difference therefore produced a gold one day
    below what is computable whenever the later event's clock time was the earlier
    of the two — 45% of the first set's duration probes, every one of them scoring a
    correct reader wrong. Ground truth must be reachable from the context that is
    actually rendered, so the times are dropped before subtracting.
    """
    probes = []
    namespaces = sorted(_dated(rows, relations).items())
    rng.shuffle(namespaces)
    for user_id, dated in namespaces:
        if len(probes) >= limit:
            break
        distinct = {row["event_time"]: row for row in dated}
        if len(distinct) < 2:
            continue
        pair = rng.sample(sorted(distinct.values(), key=lambda r: r["id"]), 2)
        first, second = sorted(pair, key=lambda r: _date(r["event_time"]))
        days = (_date(second["event_time"]).date() - _date(first["event_time"]).date()).days
        if days <= 0:
            continue
        probes.append(
            {
                "kind": "duration",
                "namespace": user_id,
                "question": (
                    f"How many days passed between these two events? "
                    f"A: {first['content']} B: {second['content']}"
                ),
                "answer": days,
                "answer_kind": "integer",
                "evidence_memory_ids": sorted([first["id"], second["id"]]),
                # Dates only, and written as dates, so the answer can be re-checked
                # from the probe against exactly what the answerer was shown.
                "derivation": (
                    f"({second['event_time'][:10]} - {first['event_time'][:10]}).days "
                    f"over dates, times dropped"
                ),
            }
        )
    return probes


def _comparison_probes(rows, relations, rng, limit) -> list[dict[str, Any]]:
    """Which of two facts happened first? Ordering without arithmetic."""
    probes = []
    namespaces = sorted(_dated(rows, relations).items())
    rng.shuffle(namespaces)
    for user_id, dated in namespaces:
        if len(probes) >= limit:
            break
        distinct = {row["event_time"]: row for row in dated}
        if len(distinct) < 2:
            continue
        pair = rng.sample(sorted(distinct.values(), key=lambda r: r["id"]), 2)
        first, second = sorted(pair, key=lambda r: _date(r["event_time"]))
        # Present in random order so the answer is not always the first one shown.
        shown = [first, second]
        rng.shuffle(shown)
        probes.append(
            {
                "kind": "comparison",
                "namespace": user_id,
                "question": (
                    f"Which of these happened first? "
                    f"A: {shown[0]['content']} B: {shown[1]['content']}"
                ),
                "answer": "A" if shown[0]["id"] == first["id"] else "B",
                "answer_kind": "choice",
                "evidence_memory_ids": sorted([first["id"], second["id"]]),
                # Both shown options are dated in the derivation so the answer
                # letter can be re-checked from the probe alone, without trusting
                # the generator that produced it.
                "derivation": (
                    f"A={shown[0]['event_time']} B={shown[1]['event_time']}; "
                    f"earlier is {'A' if shown[0]['id'] == first['id'] else 'B'}"
                ),
            }
        )
    return probes


def _current_state_probes(rows, rng, limit) -> list[dict[str, Any]]:
    """Which value is current when a key has been superseded?

    The only probe kind that needs a real supersession chain, and the one that tests
    whether the temporal semantics the store already computed survive into the answer.
    """
    chains: dict[tuple[str, str, str], list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        chains[(row["user_id"], row["subject"], row["predicate"])].append(row)
    usable = []
    for key, members in sorted(chains.items()):
        active = [m for m in members if m["status"] == "active"]
        superseded = [m for m in members if m["status"] == "superseded"]
        if len(active) == 1 and superseded:
            usable.append((key, active[0], superseded))
    rng.shuffle(usable)
    probes = []
    for (user_id, subject, predicate), active, superseded in usable[:limit]:
        probes.append(
            {
                "kind": "current_state",
                "namespace": user_id,
                "question": (
                    f"What is currently true about {subject}'s {predicate.replace('_', ' ')}?"
                ),
                "answer": active["object"] or active["content"],
                "answer_kind": "text",
                "evidence_memory_ids": sorted([active["id"], *(m["id"] for m in superseded)]),
                "derivation": (
                    f"the single active memory on ({subject!r}, {predicate!r}); "
                    f"{len(superseded)} superseded value(s) precede it"
                ),
            }
        )
    return probes


def generate(store: Path, relation_map: Path, seed: int, per_kind: int) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
    try:
        fingerprint = _fingerprint(connection)
        rows = _memories(connection)
    finally:
        connection.close()
    relations = _relation_map(relation_map)

    probes: list[dict[str, Any]] = []
    # One generator per kind, each with its own stream, so changing `per_kind` for one
    # kind cannot shift the probes another kind would have drawn.
    for name, build in (
        ("count", lambda r: _count_probes(rows, relations, r, per_kind)),
        ("duration", lambda r: _duration_probes(rows, relations, r, per_kind)),
        ("comparison", lambda r: _comparison_probes(rows, relations, r, per_kind)),
        ("current_state", lambda r: _current_state_probes(rows, r, per_kind)),
    ):
        stream = random.Random(f"{seed}:{fingerprint}:{name}")
        for index, probe in enumerate(build(stream)):
            probe["probe_id"] = f"{name}_{index:04d}"
            probes.append(probe)

    return {
        "schema_version": 1,
        "seed": seed,
        "per_kind": per_kind,
        "store": store.name,
        "store_fingerprint": fingerprint,
        "relation_map_sha256": hashlib.sha256(relation_map.read_bytes()).hexdigest(),
        "not_a_benchmark": (
            "Synthesis probes measure operations over stored facts. Ground truth is "
            "derived by SQL and inherits any extraction loss, so this is not accuracy "
            "and must not be reported beside a LongMemEval figure."
        ),
        "probes": probes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=REPO / "stores/train150.db")
    parser.add_argument(
        "--relation-map", type=Path, default=REPO / "results/analysis/predicate-map.csv"
    )
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--per-kind", type=int, default=60)
    parser.add_argument("--out", type=Path, default=REPO / "results/analysis/synthesis-probes.json")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="regenerate and compare against --out instead of writing it",
    )
    args = parser.parse_args()

    built = generate(args.store, args.relation_map, args.seed, args.per_kind)

    if args.verify:
        if not args.out.is_file():
            print(f"STOP: no probe set at {args.out}")
            return 2
        existing = json.loads(args.out.read_text(encoding="utf-8"))
        if existing != built:
            print("STOP: the probe set does not reproduce from its own seed and store")
            return 2
        print(f"PASS: {len(built['probes'])} probes reproduce exactly")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(built, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    counts: dict[str, int] = defaultdict(int)
    for probe in built["probes"]:
        counts[probe["kind"]] += 1
    print(f"store fingerprint: {built['store_fingerprint'][:16]}…  seed: {args.seed}")
    for kind in ("count", "duration", "comparison", "current_state"):
        print(f"  {kind:<16}{counts[kind]:>4}")
    print(f"  {'TOTAL':<16}{len(built['probes']):>4}   -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
