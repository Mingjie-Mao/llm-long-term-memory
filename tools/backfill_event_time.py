"""Recompute `event_time` on a store that was written before it meant anything.

Splitting `event_time` from `observed_at` gave new ingests a real signal: a memory's
time is either something the fact states, or nothing at all. Stores written before that
carry the conversation's date in `event_time` for every row, so
`Memory.event_time_is_stated` reads True on 8,968 of 8,968 memories and discriminates
nothing. A mechanism that wants to prefer a dated fact over a restatement of it cannot
be evaluated on such a store at all.

Re-ingesting would answer it and cost a quota day. It is not necessary: the two inputs
`stated_event_time` needs — the fact's own words, and the date it was said — are both
already columns. This reads them back and writes what the extractor would write today,
with no model calls.

**Not a migration.** `_migrate` deliberately leaves `event_time` alone, because
rewriting a column across every existing research store is not something that should
happen because somebody opened a database. This is opt-in, reports without writing by
default, and refuses a store that lacks the column it reads from.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import write_report  # noqa: E402

from llm_long_term_memory.ingest.event_time import temporal_evidence  # noqa: E402


def _parse(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def recompute(path: Path) -> tuple[dict, list[tuple[str | None, str]]]:
    """The report, and the `(new event_time, id)` pairs that would be written."""
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        columns = {r["name"] for r in connection.execute("PRAGMA table_info(memories)")}
        if "observed_at" not in columns:
            raise SystemExit(
                f"{path} has no `observed_at` column. Open it once with the store, which "
                "adds it and backfills it from `event_time`, then re-run."
            )
        rows = connection.execute(
            "SELECT id, content, event_time, observed_at FROM memories"
        ).fetchall()
    finally:
        connection.close()

    pending: list[tuple[str | None, str]] = []
    kinds: Counter[str] = Counter()
    examples: list[dict] = []
    stated_after = 0

    for row in rows:
        # `observed_at` is the anchor a bare "February 14" resolves against. On a row
        # the migration has touched it equals `event_time`; the fallback is for a store
        # migrated by some other route.
        observed = _parse(row["observed_at"]) or _parse(row["event_time"])
        before = _parse(row["event_time"])
        # The same call ingestion makes, so a backfilled store matches a new one.
        after = temporal_evidence(row["content"] or "", observed).exact
        if after is not None:
            stated_after += 1
        if before == after:
            kinds["unchanged"] += 1
            continue
        if after is None:
            kinds["assumed_date_withdrawn"] += 1
        elif before is None:
            kinds["date_recovered"] += 1
        else:
            kinds["corrected_to_the_stated_date"] += 1
        pending.append((after.isoformat() if after else None, row["id"]))
        if len(examples) < 20 and after is not None:
            examples.append(
                {
                    "id": row["id"],
                    "content": row["content"],
                    "was": row["event_time"],
                    "now": after.isoformat(),
                }
            )

    report = {
        "store": str(path),
        "memories": len(rows),
        "changed": len(pending),
        "by_kind": dict(sorted(kinds.items())),
        "carrying_a_time_before": sum(1 for r in rows if r["event_time"]),
        "carrying_a_stated_time_after": stated_after,
        "recovered_examples": examples,
    }
    return report, pending


def apply(path: Path, pending: list[tuple[str | None, str]]) -> int:
    """Write the recomputed times. Only rows that move are touched."""
    connection = sqlite3.connect(path)
    try:
        connection.executemany("UPDATE memories SET event_time = ? WHERE id = ?", pending)
        connection.commit()
    finally:
        connection.close()
    return len(pending)


EVIDENCE_COLUMNS = (
    "event_time_expression",
    "event_time_estimate",
    "event_time_precision",
    "event_time_source_expression",
)


def recompute_evidence(path: Path) -> tuple[dict, list[tuple]]:
    """The four evidence columns new ingests fill and old stores leave NULL.

    `event_time_expression`, `_estimate` and `_precision` come from the fact's own
    words; `event_time_source_expression` from the raw span the fact was anchored to,
    exactly as `provenance.attach_source_span` computes it. Rows without an anchor keep
    NULL there.
    """
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        columns = {r["name"] for r in connection.execute("PRAGMA table_info(memories)")}
        missing = [c for c in ("observed_at", *EVIDENCE_COLUMNS) if c not in columns]
        if missing:
            raise SystemExit(
                f"{path} lacks {', '.join(missing)}. Open it once read-write with the "
                "store, whose migration adds them, then re-run."
            )
        rows = connection.execute(
            "SELECT m.id, m.content, m.event_time, m.observed_at, "
            + ", ".join(f"m.{c}" for c in EVIDENCE_COLUMNS)
            + ", t.content AS turn_text, m.source_char_start, m.source_char_end "
            "FROM memories m LEFT JOIN turns t "
            "ON t.session_id = m.source_session_id AND t.turn_index = m.source_turn_index"
        ).fetchall()
    finally:
        connection.close()

    pending: list[tuple] = []
    precision: Counter[str] = Counter()
    for row in rows:
        observed = _parse(row["observed_at"]) or _parse(row["event_time"])
        evidence = temporal_evidence(row["content"] or "", observed)
        source = None
        if row["turn_text"] is not None and row["source_char_start"] is not None:
            span = row["turn_text"][row["source_char_start"] : row["source_char_end"]]
            source = temporal_evidence(span, observed).expression
        values = (
            evidence.expression,
            evidence.estimate.isoformat() if evidence.estimate else None,
            evidence.precision,
            source,
        )
        precision[evidence.precision or "none"] += 1
        if values != tuple(row[c] for c in EVIDENCE_COLUMNS):
            pending.append((*values, row["id"]))
    report = {
        "store": str(path),
        "memories": len(rows),
        "evidence_rows_changed": len(pending),
        "precision": dict(sorted(precision.items())),
    }
    return report, pending


def apply_evidence(path: Path, pending: list[tuple]) -> int:
    connection = sqlite3.connect(path)
    try:
        connection.executemany(
            "UPDATE memories SET "
            + ", ".join(f"{c} = ?" for c in EVIDENCE_COLUMNS)
            + " WHERE id = ?",
            pending,
        )
        connection.commit()
    finally:
        connection.close()
    return len(pending)


def render(result: dict) -> str:
    lines = [
        "# Event-time backfill",
        "",
        f"> Store: `{result['store']}`. No model calls — every input is already a column.",
        "",
        f"- memories: **{result['memories']:,}**",
        f"- rows whose `event_time` moves: **{result['changed']:,}**",
        f"- rows carrying a time before: **{result['carrying_a_time_before']:,}** "
        "— every row, which is the defect",
        f"- rows carrying a *stated* time after: **{result['carrying_a_stated_time_after']:,}**",
        "",
        "| what happened | rows |",
        "|---|---:|",
    ]
    lines += [f"| `{kind}` | {count:,} |" for kind, count in result["by_kind"].items()]
    lines += [
        "",
        "`assumed_date_withdrawn` is the bulk of it, and is the point: those facts never",
        "stated a time and the conversation's date was standing in for one. They keep it",
        "in `observed_at`, so ordering is unchanged — what changes is that they stop",
        "claiming to be evidence about *when*.",
    ]
    if result["recovered_examples"]:
        lines += [
            "",
            "## Facts that turn out to state their own date",
            "",
            "| was | now | fact |",
            "|---|---|---|",
        ]
        lines += [
            f"| {e['was']} | **{e['now']}** | {e['content'][:90]} |"
            for e in result["recovered_examples"]
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("store", type=Path)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Apply the change. Without it nothing is written and the report is printed.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args()

    report, pending = recompute(args.store)
    evidence_report, evidence_pending = recompute_evidence(args.store)
    report["evidence"] = evidence_report
    print(write_report(report, render, json_out=args.json_out, md_out=args.md_out), end="")
    print(
        f"evidence columns: {len(evidence_pending):,} rows would change; "
        f"precision {evidence_report['precision']}"
    )
    if args.write:
        written = apply(args.store, pending)
        evidence_written = apply_evidence(args.store, evidence_pending)
        print(f"\nwrote event_time on {written:,} rows and evidence on {evidence_written:,}")
    else:
        print("\nnothing written. Pass --write to apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
