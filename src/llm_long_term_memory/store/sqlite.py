"""SQLite-backed MemoryStore.

Chosen over Postgres for the first version because a single file makes the demo
trivially reproducible and FTS5 provides BM25 without a second service. The
`MemoryStore` Protocol exists so a pgvector backend can be added in V2 without
touching the retrieval or packing layers.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .base import LexicalHit, Memory, MemoryStatus, MemoryType, Session, Turn

_SCHEMA = Path(__file__).with_name("schema.sql")

# FTS5 treats these as syntax. A raw user question containing one of them raises
# sqlite3.OperationalError, so queries are tokenized and re-quoted rather than
# passed through. Escaping (not stripping) keeps recall on hyphenated terms.
_FTS_TOKEN = re.compile(r"[A-Za-z0-9_]+")

_MEMORY_COLUMNS = (
    "id, user_id, type, content, subject, predicate, object, source_role, scope, "
    "importance, confidence, "
    "event_time, valid_from, valid_to, ingested_at, update_op, replaces_previous, "
    "superseded_by, status, "
    "strength, "
    "access_count, last_accessed_at, strength_updated_at, token_count, source_session_id, "
    "source_turn_index, source_char_start, source_char_end"
)


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _parse(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _fts_match(query: str) -> str:
    """Tokenize and quote a free-text query for FTS5.

    FTS5 treats punctuation as syntax, so a raw user question containing one of its
    operators raises OperationalError. Quoting each token escapes rather than strips,
    which keeps recall on hyphenated terms.
    """
    tokens = _FTS_TOKEN.findall(query)
    return " OR ".join(f'"{t}"' for t in tokens)


class SQLiteMemoryStore:
    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        if read_only:
            self._conn = sqlite3.connect(
                self.path.resolve().as_uri() + "?mode=ro", uri=True, check_same_thread=False
            )
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def initialize(self) -> None:
        if self.read_only:
            self._conn.execute("SELECT id FROM memories LIMIT 0")
            return
        self._conn.executescript(_SCHEMA.read_text(encoding="utf-8"))
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Apply additive migrations to stores created by earlier project phases.

        The schema file is deliberately a clean description of a new database, but
        SQLite's `CREATE TABLE IF NOT EXISTS` never adds columns to an existing
        table. Without this small migration runner, opening a v1 store after adding
        Stage B's `update_op` fails at the next insert — or, worse, leaves a store
        whose code and metadata silently disagree.
        """
        columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(memories)").fetchall()
        }
        if "update_op" not in columns:
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN update_op TEXT NOT NULL DEFAULT 'coexists'"
            )
        if "strength_updated_at" not in columns:
            self._conn.execute("ALTER TABLE memories ADD COLUMN strength_updated_at TEXT")
        if "source_turn_index" not in columns:
            self._conn.execute("ALTER TABLE memories ADD COLUMN source_turn_index INTEGER")
        if "source_char_start" not in columns:
            self._conn.execute("ALTER TABLE memories ADD COLUMN source_char_start INTEGER")
        if "source_char_end" not in columns:
            self._conn.execute("ALTER TABLE memories ADD COLUMN source_char_end INTEGER")
        # P10. `scope` stays NULL on pre-P10 rows: it is genuinely unknown, and a
        # made-up value would be indistinguishable from a real one at query time.
        #
        # `source_role` can be recovered exactly, though. The old extractor set
        # subject='assistant' precisely when the fact string began "The assistant",
        # so those rows *were* assistant-spoken; everything else was written under a
        # user-profile framing. Backfilling from that is a derivation, not a guess,
        # and without it every recovered assistant fact in an existing store would be
        # mislabelled the moment source_role became queryable.
        if "source_role" not in columns:
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN source_role TEXT NOT NULL DEFAULT 'user'"
            )
            self._conn.execute(
                "UPDATE memories SET source_role = 'assistant' WHERE subject = 'assistant'"
            )
        if "scope" not in columns:
            self._conn.execute("ALTER TABLE memories ADD COLUMN scope TEXT")

        # Idempotency keys gained a request fingerprint and an explicit state. Existing
        # rows are rows whose reply was already recorded, so they backfill to 'done' — the
        # default in the schema — and to a NULL fingerprint, which the claim path reads as
        # "claimed before fingerprints existed" and lets through rather than rejecting a
        # legitimate retry of a write that predates this column.
        write_key_columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(write_keys)").fetchall()
        }
        if write_key_columns and "fingerprint" not in write_key_columns:
            self._conn.execute("ALTER TABLE write_keys ADD COLUMN fingerprint TEXT")
        if write_key_columns and "state" not in write_key_columns:
            self._conn.execute(
                "ALTER TABLE write_keys ADD COLUMN state TEXT NOT NULL DEFAULT 'done'"
            )
        if write_key_columns and "claimed_at" not in write_key_columns:
            self._conn.execute("ALTER TABLE write_keys ADD COLUMN claimed_at TEXT")

        # The turns index is created empty by the schema and its triggers only fire
        # on rows inserted afterwards, so a store ingested before the index existed
        # would search an empty archive and silently find nothing — the worst failure
        # mode for a fallback, since "no evidence" and "never indexed" are
        # indistinguishable from outside.
        #
        # Detected with an explicit marker rather than by counting rows in the index:
        # `turns_fts` is an external-content table, so `SELECT count(*)` reads the
        # underlying `turns` table and reports full coverage even when nothing has
        # been indexed at all.
        built = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'turns_fts_built'"
        ).fetchone()
        if not built and self._conn.execute("SELECT count(*) FROM turns").fetchone()[0]:
            self._conn.execute("INSERT INTO turns_fts(turns_fts) VALUES ('rebuild')")
            self._conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('turns_fts_built', '1')"
            )

    # ---------------------------------------------------------------- sessions

    def add_session(self, session: Session) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO sessions(id, user_id, started_at, source) VALUES (?,?,?,?)",
                (session.id, session.user_id, _dt(session.started_at), session.source),
            )
            self._conn.executemany(
                "INSERT OR REPLACE INTO turns(id, session_id, turn_index, role, content, ts) "
                "VALUES (?,?,?,?,?,?)",
                [
                    (t.id, t.session_id, t.turn_index, t.role, t.content, _dt(t.ts))
                    for t in session.turns
                ],
            )

    def remove_turn(self, turn_id: str) -> bool:
        """Delete one turn. True when a row was actually removed.

        Exists for the write path's rollback. A turn is persisted before extraction runs,
        so a provider failure leaves a turn behind with nothing extracted from it; without
        this, the retry either appends a duplicate turn or has to be refused forever. The
        `turns_fts` trigger removes the index entry with the row.
        """
        with self._conn:
            cursor = self._conn.execute("DELETE FROM turns WHERE id = ?", (turn_id,))
        return cursor.rowcount == 1

    def remove_memories(self, memory_ids: list[str]) -> int:
        """Delete memory rows written by a write that then failed. Returns rows removed.

        The other half of `remove_turn`. Extraction succeeding is not the end of the
        write: the index still has to accept the vectors, save them, and the resolver
        still has to run. A failure in any of those left the memories durable while the
        caller was told the write failed, and the retry — which re-extracts from scratch —
        wrote them a second time.

        `memory_entities` and `evidence` cascade. `superseded_by` does not: it is a plain
        reference, so a row pointing at one of these has to be restored *first* or the
        delete leaves a dangling id. `restore_memory_states` is what does that, and the
        write path calls it before this.
        """
        if not memory_ids:
            return 0
        marks = ",".join("?" * len(memory_ids))
        with self._conn:
            cursor = self._conn.execute(
                f"DELETE FROM memories WHERE id IN ({marks})", tuple(memory_ids)
            )
        return cursor.rowcount

    def restore_memory_states(self, states: list[tuple[str, str, str | None, Any, Any]]) -> None:
        """Put back the status, supersede link and validity window of existing rows.

        Resolution edits rows that were already there — closing their validity interval
        and pointing them at a successor. When the write those edits belong to does not
        complete, the edits have to come back off, and they cannot be recomputed: the
        resolver is not an inverse function. So the write path snapshots the rows on the
        keys it is about to touch, and this restores that snapshot verbatim.
        """
        if not states:
            return
        with self._conn:
            self._conn.executemany(
                "UPDATE memories SET status=?, superseded_by=?, valid_from=?, valid_to=? "
                "WHERE id=?",
                [
                    (status, superseded_by, _dt(valid_from), _dt(valid_to), memory_id)
                    for memory_id, status, superseded_by, valid_from, valid_to in states
                ],
            )

    def count_turns(self, user_id: str) -> int:
        row = self._conn.execute(
            "SELECT count(*) FROM turns WHERE session_id IN "
            "(SELECT id FROM sessions WHERE user_id = ?)",
            (user_id,),
        ).fetchone()
        return int(row[0])

    # ------------------------------------------------------------ idempotent writes

    def write_key(self, user_id: str, key: str) -> dict[str, Any] | None:
        """What is on record for one idempotency key, or None if it is unclaimed.

        Returned rather than interpreted here: whether a pending claim is live or an
        orphan is a policy question about how long a write may take, and that belongs
        with the caller that knows.
        """
        row = self._conn.execute(
            "SELECT response, created_at, fingerprint, state, claimed_at "
            "FROM write_keys WHERE user_id = ? AND key = ?",
            (user_id, key),
        ).fetchone()
        return dict(row) if row else None

    def remembered_write(self, user_id: str, key: str) -> str | None:
        """The reply a previous identical write returned, if there was one."""
        row = self._conn.execute(
            "SELECT response FROM write_keys WHERE user_id = ? AND key = ?",
            (user_id, key),
        ).fetchone()
        return row[0] if row else None

    def remember_write(
        self, user_id: str, key: str, response: str, fingerprint: str | None = None
    ) -> bool:
        """Claim a key for a write in progress. False when the key was already taken.

        `INSERT OR IGNORE` rather than a check-then-write: two concurrent retries of the
        same request would both pass a check and both proceed, which is the race the key
        exists to close.
        """
        with self._conn:
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO write_keys "
                "(user_id, key, response, created_at, fingerprint, state, claimed_at) "
                "VALUES (?, ?, ?, ?, ?, 'pending', ?)",
                # `created_at` keeps the store's local convention. `claimed_at` is UTC because
                # the takeover window is measured from it, and in local wall-clock time that
                # window is wrong for an hour twice a year: across the spring-forward gap a
                # claim seconds old reads as an hour old and is taken over while still running.
                (
                    user_id,
                    key,
                    response,
                    datetime.now().isoformat(),
                    fingerprint,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return cursor.rowcount == 1

    def take_over_write(
        self,
        user_id: str,
        key: str,
        fingerprint: str | None,
        *,
        expected_state: str,
        expected_claimed_at: str | None,
    ) -> bool:
        """Re-claim a key whose previous attempt failed or was abandoned.

        A compare-and-swap on exactly what the caller read — the state *and* the claim time
        — in one statement. Conditioning on the state alone did not close the race this
        exists for: the first taker sets the row back to 'pending', which a condition of
        `state IN ('failed', 'pending')` still matches, so a second retry that had read the
        same orphan took it over as well and both wrote. Every takeover stamps a new
        `claimed_at`, so a row anyone has re-claimed since the read no longer matches.
        """
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE write_keys SET state = 'pending', claimed_at = ?, fingerprint = ?, "
                "response = ? WHERE user_id = ? AND key = ? AND state = ? AND claimed_at IS ?",
                (
                    datetime.now(UTC).isoformat(),
                    fingerprint,
                    json.dumps({"pending": True}),
                    user_id,
                    key,
                    expected_state,
                    expected_claimed_at,
                ),
            )
        return cursor.rowcount == 1

    def record_write_reply(self, user_id: str, key: str, response: str) -> None:
        """Replace a claimed key's placeholder with the reply it produced."""
        with self._conn:
            self._conn.execute(
                "UPDATE write_keys SET response = ?, state = 'done' WHERE user_id = ? AND key = ?",
                (response, user_id, key),
            )

    def record_write_failure(self, user_id: str, key: str, detail: str) -> None:
        """Mark a claimed key as failed so the caller can retry it.

        The row is kept rather than deleted. A deleted key is indistinguishable from one
        that was never claimed, and the operator asking "did that write ever run?" needs
        the difference; it is also what makes a retry visible as a retry.
        """
        with self._conn:
            self._conn.execute(
                "UPDATE write_keys SET state = 'failed', response = ? "
                "WHERE user_id = ? AND key = ? AND state = 'pending'",
                (json.dumps({"failed": True, "detail": detail[:500]}), user_id, key),
            )

    # ------------------------------------------------------------ account budgets

    def accounts_with_quota(self, day: str | None = None) -> list[str]:
        """Accounts that hold a cap or have spent today.

        Not the same set as `user_ids()`, which lists namespaces holding memories. An
        account can be spending quota with no memory to show for it — every write failed,
        or it only ever searched — and those are exactly the accounts an operator wants
        named before they are refused.
        """
        sql = "SELECT user_id FROM account_budgets"
        params: tuple = ()
        if day is not None:
            sql += " UNION SELECT user_id FROM account_usage WHERE day = ?"
            params = (day,)
        else:
            sql += " UNION SELECT user_id FROM account_usage"
        return sorted(row[0] for row in self._conn.execute(sql, params))

    def account_budget(self, user_id: str, model: str, operation: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT daily_calls, daily_tokens FROM account_budgets "
            "WHERE user_id = ? AND model = ? AND operation = ?",
            (user_id, model, operation),
        ).fetchone()
        return dict(row) if row else None

    def set_account_budget(
        self,
        user_id: str,
        model: str,
        operation: str,
        daily_calls: int | None,
        daily_tokens: int | None,
    ) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO account_budgets "
                "(user_id, model, operation, daily_calls, daily_tokens, updated_at) "
                "VALUES (?,?,?,?,?,?)",
                (user_id, model, operation, daily_calls, daily_tokens, datetime.now().isoformat()),
            )

    def account_spend(self, user_id: str, day: str) -> Any:
        """One account's spend for one quota day, summed across models and operations."""
        from llm_long_term_memory.api.budget import Spend

        row = self._conn.execute(
            "SELECT COALESCE(sum(calls), 0) AS calls, "
            "COALESCE(sum(failed_calls), 0) AS failed_calls, "
            "COALESCE(sum(input_tokens), 0) AS input_tokens, "
            "COALESCE(sum(output_tokens), 0) AS output_tokens "
            "FROM account_usage WHERE user_id = ? AND day = ?",
            (user_id, day),
        ).fetchone()
        return Spend(
            int(row["calls"]),
            int(row["failed_calls"]),
            int(row["input_tokens"]),
            int(row["output_tokens"]),
        )

    def record_account_usage(
        self,
        user_id: str,
        day: str,
        model: str,
        operation: str,
        *,
        calls: int,
        failed_calls: int,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Add to one ledger row, creating it if today is its first call.

        `ON CONFLICT ... DO UPDATE` rather than read-modify-write: two concurrent calls
        doing the latter would each read the same count and each write it back plus one,
        losing a call from the ledger — and a ledger that undercounts is a budget that
        does not hold.
        """
        with self._conn:
            self._conn.execute(
                "INSERT INTO account_usage "
                "(user_id, day, model, operation, calls, failed_calls, input_tokens, "
                "output_tokens, updated_at) VALUES (?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(user_id, day, model, operation) DO UPDATE SET "
                "calls = calls + excluded.calls, "
                "failed_calls = failed_calls + excluded.failed_calls, "
                "input_tokens = input_tokens + excluded.input_tokens, "
                "output_tokens = output_tokens + excluded.output_tokens, "
                "updated_at = excluded.updated_at",
                (
                    user_id,
                    day,
                    model,
                    operation,
                    calls,
                    failed_calls,
                    input_tokens,
                    output_tokens,
                    datetime.now().isoformat(),
                ),
            )

    def hard_delete_user(self, user_id: str) -> dict[str, int | list[str]]:
        """Physically remove one namespace from every SQLite-backed data plane.

        Vector deletion remains the caller's responsibility because the vector
        index is a separate resource; the removed memory ids are returned for it.
        """
        session_ids = [
            row[0]
            for row in self._conn.execute("SELECT id FROM sessions WHERE user_id = ?", (user_id,))
        ]
        memory_ids = [
            row[0]
            for row in self._conn.execute("SELECT id FROM memories WHERE user_id = ?", (user_id,))
        ]
        turns = self.count_turns(user_id)
        with self._conn:
            self._conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
            self._conn.execute(
                "DELETE FROM turns WHERE session_id IN (SELECT id FROM sessions WHERE user_id = ?)",
                (user_id,),
            )
            self._conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            self._conn.execute("DELETE FROM entities WHERE user_id = ?", (user_id,))
            # Idempotency keys go too: replaying one after erasure would return a reply
            # describing memories that no longer exist.
            self._conn.execute("DELETE FROM write_keys WHERE user_id = ?", (user_id,))
            # `account_usage` and `account_budgets` are deliberately NOT deleted here.
            # Erasure removes the user's content; the ledger holds counts, not content,
            # and clearing it would make "delete my data" a way to reset the day's spend
            # to zero and start again. The budget row is operator configuration and is
            # not the tenant's to remove either. Both are dropped by the operator when an
            # account is genuinely closed, which is a different action from an erasure.
        return {
            "sessions": len(session_ids),
            "memories": len(memory_ids),
            "turns": turns,
            "ids": memory_ids,
        }

    # ---------------------------------------------------------------- memories

    def add_memories(self, memories: list[Memory]) -> None:
        if not memories:
            return
        rows = [
            (
                m.id,
                m.user_id,
                m.type,
                m.content,
                m.subject,
                m.predicate,
                m.object,
                m.source_role,
                m.scope,
                m.importance,
                m.confidence,
                _dt(m.event_time),
                _dt(m.valid_from),
                _dt(m.valid_to),
                _dt(m.ingested_at or datetime.now()),
                m.update_op,
                int(m.replaces_previous),
                m.superseded_by,
                m.status,
                m.strength,
                m.access_count,
                _dt(m.last_accessed_at),
                _dt(m.strength_updated_at),
                m.token_count,
                m.source_session_id,
                m.source_turn_index,
                m.source_char_start,
                m.source_char_end,
            )
            for m in memories
        ]
        placeholders = ",".join(["?"] * len(_MEMORY_COLUMNS.split(",")))
        with self._conn:
            self._conn.executemany(
                f"INSERT OR REPLACE INTO memories({_MEMORY_COLUMNS}) VALUES ({placeholders})",
                rows,
            )
            for m in memories:
                self._link_entities(m)

    def _link_entities(self, memory: Memory) -> None:
        for surface in memory.entities:
            canonical = surface.strip().lower()
            if not canonical:
                continue
            entity_id = f"{memory.user_id}:{canonical}"
            self._conn.execute(
                "INSERT OR IGNORE INTO entities(id, user_id, canonical_name, surface_form) "
                "VALUES (?,?,?,?)",
                (entity_id, memory.user_id, canonical, surface),
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO memory_entities(memory_id, entity_id) VALUES (?,?)",
                (memory.id, entity_id),
            )

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        entities = [
            r["surface_form"]
            for r in self._conn.execute(
                "SELECT e.surface_form FROM entities e "
                "JOIN memory_entities me ON me.entity_id = e.id WHERE me.memory_id = ?",
                (row["id"],),
            )
        ]
        return Memory(
            id=row["id"],
            user_id=row["user_id"],
            type=row["type"],
            content=row["content"],
            token_count=row["token_count"],
            subject=row["subject"],
            predicate=row["predicate"],
            object=row["object"],
            source_role=row["source_role"] or "user",
            scope=row["scope"],
            importance=row["importance"],
            confidence=row["confidence"],
            event_time=_parse(row["event_time"]),
            valid_from=_parse(row["valid_from"]),
            valid_to=_parse(row["valid_to"]),
            ingested_at=_parse(row["ingested_at"]),
            update_op=row["update_op"],
            replaces_previous=bool(row["replaces_previous"]),
            superseded_by=row["superseded_by"],
            status=row["status"],
            strength=row["strength"],
            access_count=row["access_count"],
            last_accessed_at=_parse(row["last_accessed_at"]),
            strength_updated_at=_parse(row["strength_updated_at"]),
            source_session_id=row["source_session_id"],
            source_turn_index=row["source_turn_index"],
            source_char_start=row["source_char_start"],
            source_char_end=row["source_char_end"],
            entities=entities,
        )

    def get(self, memory_id: str) -> Memory | None:
        row = self._conn.execute(
            f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return self._row_to_memory(row) if row else None

    def get_many(self, memory_ids: list[str]) -> list[Memory]:
        if not memory_ids:
            return []
        marks = ",".join(["?"] * len(memory_ids))
        rows = self._conn.execute(
            f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE id IN ({marks})", memory_ids
        ).fetchall()
        by_id = {r["id"]: self._row_to_memory(r) for r in rows}
        # Preserve caller ordering — retrieval passes ids in ranked order.
        return [by_id[mid] for mid in memory_ids if mid in by_id]

    def iter_active(self, user_id: str) -> list[Memory]:
        rows = self._conn.execute(
            f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE user_id = ? AND status = 'active'",
            (user_id,),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def set_meta(self, key: str, value: str) -> None:
        """Record a property of the store itself.

        The extractor version lives here rather than in a config file because it is a
        property of *this data*, not of the code that happens to be checked out. A
        store built by one extractor and evaluated a month later under another must
        still report which one wrote it.
        """
        with self._conn:
            self._conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?,?)", (key, value))

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def iter_all(self, user_id: str) -> list[Memory]:
        """Every memory in a namespace, whatever its status.

        `iter_active` hides superseded rows, which is right for answering and wrong
        for browsing: the API's memory list and the inspector's timeline exist
        precisely to show that a fact was replaced rather than to pretend the old
        value never held.
        """
        rows = self._conn.execute(
            f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE user_id = ?", (user_id,)
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get_session(self, session_id: str) -> Session | None:
        row = self._conn.execute(
            "SELECT id, user_id, started_at, source FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return Session(
            id=row["id"],
            user_id=row["user_id"],
            started_at=_parse(row["started_at"]),
            source=row["source"],
            turns=self.turns_for_session(session_id),
        )

    def find_by_predicate(
        self, user_id: str, subject: str, predicate: str, include_superseded: bool = False
    ) -> list[Memory]:
        """Memories sharing a (subject, predicate) key.

        `include_superseded` exists for the temporal resolver, which has to rebuild
        an entire timeline rather than compare against the current head: sessions
        are ingested in arbitrary order, so a memory arriving now can be *older*
        than one already stored and has to slot in before it.
        """
        # Evicted memories remain in the database for auditability but must never
        # re-enter a rebuilt timeline. `include_superseded` means active plus
        # superseded, not every historical status.
        clause = " AND status != 'evicted'"
        if not include_superseded:
            clause += " AND status = 'active'"
        rows = self._conn.execute(
            f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE user_id = ? AND subject = ? "
            f"AND predicate = ?{clause}",
            (user_id, subject, predicate),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def user_ids(self) -> list[str]:
        """Every namespace present in the store.

        The corpus is not one user: LongMemEval questions are independent
        simulated people, so memories are namespaced per question (D25). Any
        whole-store pass has to walk all of them — `resolve_all("user")` against a
        50-namespace store silently resolves nothing and reports success.
        """
        return [r[0] for r in self._conn.execute("SELECT DISTINCT user_id FROM memories")]

    def predicate_keys(self, user_id: str) -> list[tuple[str, str]]:
        """Every (subject, predicate) pair present, for a full re-resolution pass."""
        rows = self._conn.execute(
            "SELECT DISTINCT subject, predicate FROM memories "
            "WHERE user_id = ? AND subject IS NOT NULL AND predicate IS NOT NULL",
            (user_id,),
        ).fetchall()
        return [(r["subject"], r["predicate"]) for r in rows]

    def turns_for_memories(self, memories: list[Memory]) -> list[Turn]:
        """Level 1 of the raw fallback: the source turns of memories already found.

        Cheap and precise. When retrieval located the right memory but extraction
        dropped the detail — "the assistant recommended a Mayo Clinic resource",
        without the URL — the answer is in the turn that memory came from, and
        nowhere else needs searching.
        """
        session_ids = {m.source_session_id for m in memories if m.source_session_id}
        if not session_ids:
            return []
        wanted = {(m.source_session_id, m.source_turn_index) for m in memories}
        out: list[Turn] = []
        for session_id in sorted(session_ids):
            for turn in self.turns_for_session(session_id):
                # The anchored turn, or the whole session when no anchor was stored.
                if (session_id, turn.turn_index) in wanted or (session_id, None) in wanted:
                    out.append(turn)
        return out

    def search_turns(self, user_id: str, query: str, limit: int = 3) -> list[Turn]:
        """Level 2 of the raw fallback: BM25 over the whole archive for one namespace.

        Used only when structured memory found nothing at all, because extraction
        can miss a fact entirely. Namespace-scoped like every other read — the join
        onto `sessions` is what enforces that, since `turns` has no user_id of its
        own.
        """
        match = _fts_match(query)
        if not match:
            return []
        rows = self._conn.execute(
            """
            SELECT t.id, t.session_id, t.turn_index, t.role, t.content, t.ts
            FROM turns_fts f
            JOIN turns t ON t.rowid = f.rowid
            JOIN sessions s ON s.id = t.session_id
            WHERE turns_fts MATCH ? AND s.user_id = ?
            ORDER BY bm25(turns_fts) LIMIT ?
            """,
            (match, user_id, limit),
        ).fetchall()
        return [
            Turn(
                id=r["id"],
                session_id=r["session_id"],
                turn_index=r["turn_index"],
                role=r["role"],
                content=r["content"],
                ts=_parse(r["ts"]),
            )
            for r in rows
        ]

    # --------------------------------------------------------------- retrieval

    def search_lexical(
        self, user_id: str, query: str, limit: int, include_superseded: bool = False
    ) -> list[LexicalHit]:
        match = _fts_match(query)
        if not match:
            return []
        status_clause = "m.status != 'evicted'"
        if not include_superseded:
            status_clause += " AND m.status = 'active'"
        rows = self._conn.execute(
            "SELECT m.id AS id, bm25(memories_fts) AS score "
            "FROM memories_fts JOIN memories m ON m.rowid = memories_fts.rowid "
            f"WHERE memories_fts MATCH ? AND m.user_id = ? AND {status_clause} "
            "ORDER BY score LIMIT ?",  # bm25() is negative; more negative = better
            (match, user_id, limit),
        ).fetchall()
        return [LexicalHit(memory_id=r["id"], score=r["score"]) for r in rows]

    # ----------------------------------------------------------------- updates

    def mark_superseded(self, memory_id: str, superseded_by: str, valid_to: datetime) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE memories SET status='superseded', superseded_by=?, valid_to=? WHERE id=?",
                (superseded_by, _dt(valid_to), memory_id),
            )

    def mark_current(self, memory_id: str) -> None:
        """Reopen a memory as the live value of its key.

        Needed because resolution is not monotonic: ingesting an *older* fact for a
        key can demote the memory that was previously the head, and re-running
        resolution after a correction must be able to promote one back. Without
        this, a key could only ever accumulate superseded rows.
        """
        with self._conn:
            self._conn.execute(
                "UPDATE memories SET status='active', superseded_by=NULL, valid_to=NULL WHERE id=?",
                (memory_id,),
            )

    def set_validity(self, memory_id: str, valid_to: datetime | None) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE memories SET valid_to=? WHERE id=?", (_dt(valid_to), memory_id)
            )

    def record_access(
        self, memory_ids: list[str], at: datetime, reinforcement: float = 0.30
    ) -> None:
        if not memory_ids:
            return
        reinforcement = min(1.0, max(0.0, reinforcement))
        with self._conn:
            self._conn.executemany(
                "UPDATE memories SET access_count = access_count + 1, "
                "strength = MIN(1.0, strength + ? * (1.0 - strength)), "
                "last_accessed_at = ?, strength_updated_at = ? "
                "WHERE id = ?",
                [(reinforcement, _dt(at), _dt(at), mid) for mid in memory_ids],
            )

    def set_strengths(self, strengths: dict[str, float], at: datetime | None = None) -> None:
        if not strengths:
            return
        values = [
            (min(1.0, max(0.0, strength)), memory_id) for memory_id, strength in strengths.items()
        ]
        with self._conn:
            if at is None:
                self._conn.executemany("UPDATE memories SET strength = ? WHERE id = ?", values)
            else:
                self._conn.executemany(
                    "UPDATE memories SET strength = ?, strength_updated_at = ? WHERE id = ?",
                    [(strength, _dt(at), memory_id) for strength, memory_id in values],
                )

    def mark_evicted(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        with self._conn:
            self._conn.executemany(
                "UPDATE memories SET status = 'evicted' WHERE id = ? AND status = 'active'",
                [(memory_id,) for memory_id in memory_ids],
            )

    def add_evidence(self, memory_id: str, source_memory_ids: list[str]) -> None:
        if not source_memory_ids:
            return
        with self._conn:
            self._conn.executemany(
                "INSERT OR IGNORE INTO evidence(memory_id, source_memory_id) VALUES (?, ?)",
                [
                    (memory_id, source_id)
                    for source_id in source_memory_ids
                    if source_id != memory_id
                ],
            )

    def evidence_for(self, memory_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT source_memory_id FROM evidence WHERE memory_id = ? ORDER BY source_memory_id",
            (memory_id,),
        ).fetchall()
        return [row["source_memory_id"] for row in rows]

    def evidence_source_ids(self) -> set[str]:
        rows = self._conn.execute("SELECT DISTINCT source_memory_id FROM evidence").fetchall()
        return {row["source_memory_id"] for row in rows}

    def turns_for_session(self, session_id: str) -> list[Turn]:
        rows = self._conn.execute(
            "SELECT id, session_id, turn_index, role, content, ts FROM turns "
            "WHERE session_id = ? ORDER BY turn_index",
            (session_id,),
        ).fetchall()
        return [
            Turn(
                id=row["id"],
                session_id=row["session_id"],
                turn_index=row["turn_index"],
                role=row["role"],
                content=row["content"],
                ts=_parse(row["ts"]),
            )
            for row in rows
        ]

    def count(self, user_id: str | None = None, status: MemoryStatus | None = None) -> int:
        clauses, params = [], []
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return self._conn.execute(f"SELECT COUNT(*) FROM memories{where}", params).fetchone()[0]

    def count_by_type(self, user_id: str) -> dict[MemoryType, int]:
        rows = self._conn.execute(
            "SELECT type, COUNT(*) AS n FROM memories WHERE user_id = ? AND status = 'active' "
            "GROUP BY type",
            (user_id,),
        ).fetchall()
        return {r["type"]: r["n"] for r in rows}

    def session_ids(self) -> set[str]:
        return {row["id"] for row in self._conn.execute("SELECT id FROM sessions")}

    def session_ids_for_user(self, user_id: str) -> set[str]:
        """One namespace's sessions. Scoped in SQL rather than filtered afterwards, so
        an export cannot be made to walk every tenant's sessions to build one tenant's
        document."""
        return {
            row["id"]
            for row in self._conn.execute("SELECT id FROM sessions WHERE user_id = ?", (user_id,))
        }

    def zero_yield_sessions(
        self,
        min_turns: int = 6,
        *,
        include_session_ids: set[str] | None = None,
    ) -> tuple[int, int]:
        """Sessions that were archived but produced no memory at all.

        Returns `(zero_yield, total)` counting only sessions of at least `min_turns`,
        so that a one-line stub does not flatter the number. When
        `include_session_ids` is supplied, raw-only batches that have not reached a
        checkpoint terminal state can be excluded from an interim report.

        This is computed from the store rather than counted during ingestion,
        because a counter only sees sessions processed after it was added and this
        needs to describe a store that was built over several days.

        It exists because the failure was invisible. Ingestion reported memories
        written, duplicates dropped and bad session indices, and nothing at all
        about sessions that yielded silence — so 16.7% of substantive sessions in
        the first store, and 13.4% in the second, produced no memory while every
        number on the report looked healthy. Measured 2026-08-15 across both stores;
        the profile of a zero-yield session is indistinguishable from a normal one
        (same median turn count, same assistant share), so this is extractor
        variance rather than a property of the conversation.
        """
        rows = self._conn.execute(
            "SELECT s.id, "
            "(SELECT COUNT(*) FROM turns t WHERE t.session_id = s.id) AS turn_count, "
            "EXISTS(SELECT 1 FROM memories m WHERE m.source_session_id = s.id) AS has_memory "
            "FROM sessions s"
        )
        eligible = [
            row
            for row in rows
            if row["turn_count"] >= min_turns
            and (include_session_ids is None or row["id"] in include_session_ids)
        ]
        total = len(eligible)
        zero = sum(not row["has_memory"] for row in eligible)
        return zero, total

    def close(self) -> None:
        self._conn.close()
