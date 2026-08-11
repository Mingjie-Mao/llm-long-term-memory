"""Checkpointed ingestion driver.

D5 measured a full pass over LongMemEval-S at ~1,920 extraction requests, which
spans more than one day of free-tier quota. So the driver is built around stopping:
progress is committed after every batch, exhausting the daily budget is a clean
return rather than an exception, and a resumed run skips sessions already ingested.

**Each question is its own user.** LongMemEval builds a question's haystack by
padding its evidence sessions with distractors drawn from unrelated conversations,
so two questions share no history — measured: Q1 and Q2 have zero sessions in
common. Ingesting the union of all sessions under one `user_id` merged fifty
different people into one memory store, which produced a `lives_in` chain running
Toronto -> Greenville -> Seattle -> Tokyo -> Shanghai -> Hyderabad -> Las Vegas and
let retrieval for one question return another question's evidence.

So ingestion is namespaced by `question_id`. That gives up the 1.24x session-sharing
saving from D5 — which is what motivated the union in the first place — and the
trade is not close: 20% fewer requests is worth nothing if the store is wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from chronomem.embed import Encoder
from chronomem.evaluation.datasets.longmemeval import HaystackSession, Instance
from chronomem.llm.client import DailyQuotaExhausted
from chronomem.store import MemoryStore, NumpyFlatIndex

from .dedup import Deduplicator
from .extract import Extractor


@dataclass
class IngestProgress:
    done_sessions: set[str] = field(default_factory=set)
    memories_written: int = 0
    duplicates_dropped: int = 0
    updates_detected: int = 0
    bad_session_index: int = 0
    extraction_requests: int = 0
    adjudication_requests: int = 0
    superseded: int = 0
    undated: int = 0

    def to_dict(self) -> dict:
        return {
            "done_sessions": sorted(self.done_sessions),
            "memories_written": self.memories_written,
            "duplicates_dropped": self.duplicates_dropped,
            "updates_detected": self.updates_detected,
            "bad_session_index": self.bad_session_index,
            "extraction_requests": self.extraction_requests,
            "adjudication_requests": self.adjudication_requests,
            "superseded": self.superseded,
            "undated": self.undated,
        }

    @classmethod
    def load(cls, path: Path) -> IngestProgress:
        if not path.exists():
            return cls()
        d = json.loads(path.read_text())
        p = cls(done_sessions=set(d.get("done_sessions", [])))
        for key in (
            "memories_written",
            "duplicates_dropped",
            "updates_detected",
            "bad_session_index",
            "extraction_requests",
            "adjudication_requests",
            "superseded",
            "undated",
        ):
            setattr(p, key, d.get(key, 0))
        return p

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.replace(path)  # atomic, so a kill mid-write cannot corrupt the checkpoint


def namespaced_sessions(instances: list[Instance]) -> list[tuple[str, HaystackSession]]:
    """(namespace, session) pairs, one namespace per question.

    Deliberately not deduplicated across questions: a session appearing in two
    haystacks belongs to two different simulated users and must be extracted into
    both stores. Deduplicating it merges the personas.
    """
    out: list[tuple[str, HaystackSession]] = []
    for inst in instances:
        seen: set[str] = set()
        for sess in inst.sessions:
            if sess.session_id in seen:
                continue
            seen.add(sess.session_id)
            out.append((inst.question_id, sess))
    return out


def group_by_namespace(
    pairs: list[tuple[str, HaystackSession]],
) -> list[tuple[str, list[HaystackSession]]]:
    """Batches never straddle a namespace: extraction of one question's sessions
    must not attribute a fact to another question's user."""
    groups: dict[str, list[HaystackSession]] = {}
    for ns, sess in pairs:
        groups.setdefault(ns, []).append(sess)
    return list(groups.items())


def fit_batch_size(
    configured: int, tpm: int, tokens_per_session: float, headroom: float = 0.8
) -> int:
    """Largest batch that fits under the model's per-minute token allowance.

    D5 picked ten sessions per request against a 250k TPM budget. That number does
    not transfer: Gemma's free-tier allowance is 16k tokens/minute, so a ten-session
    batch (~25.6k tokens) can never be sent at all. Deriving the batch from the
    model's own limit keeps the extractor swappable — otherwise every model change
    silently needs a matching config change nobody remembers to make.

    `headroom` leaves room for the prompt scaffolding and for the estimate being an
    estimate.
    """
    if tpm <= 0 or tokens_per_session <= 0:
        return configured
    affordable = int((tpm * headroom) // tokens_per_session)
    return max(1, min(configured, affordable))


def _key(namespace: str, session: HaystackSession) -> str:
    """Checkpoint key. Namespaced, because the same session in two questions is two
    separate units of work."""
    return f"{namespace}:{session.session_id}"


def batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


@dataclass(slots=True)
class IngestOutcome:
    progress: IngestProgress
    completed: bool
    stopped_reason: str | None = None


class IngestionPipeline:
    def __init__(
        self,
        extractor: Extractor,
        deduplicator: Deduplicator,
        store: MemoryStore,
        index: NumpyFlatIndex,
        encoder: Encoder,
        checkpoint_path: str | Path,
        sessions_per_request: int = 10,
        checkpoint_every: int = 5,
        resolver=None,
    ) -> None:
        self.extractor = extractor
        self.deduplicator = deduplicator
        self.store = store
        self.index = index
        self.encoder = encoder
        self.checkpoint_path = Path(checkpoint_path)
        self.sessions_per_request = sessions_per_request
        self.checkpoint_every = checkpoint_every
        # Optional so the ablation can run ingestion with temporal
        # resolution switched off and compare against the same store.
        self.resolver = resolver

    def run(
        self,
        pairs: list[tuple[str, HaystackSession]],
        resume: bool = True,
        on_batch=None,
    ) -> IngestOutcome:
        """`pairs` is (namespace, session); one namespace per question."""
        progress = IngestProgress.load(self.checkpoint_path) if resume else IngestProgress()
        pending = [p for p in pairs if _key(*p) not in progress.done_sessions]

        # Batches never straddle a namespace, so every fact in a call belongs to the
        # user the call is attributed to.
        batches: list[tuple[str, list[HaystackSession]]] = []
        for namespace, sessions in group_by_namespace(pending):
            batches += [
                (namespace, chunk) for chunk in batched(sessions, self.sessions_per_request)
            ]

        for n, (namespace, batch) in enumerate(batches, start=1):
            try:
                self._ingest_batch(namespace, batch, progress)
            except DailyQuotaExhausted as exc:
                self._commit(progress)
                return IngestOutcome(progress, completed=False, stopped_reason=str(exc))
            except Exception as exc:
                # Anything else — the network dropping for longer than the client's
                # retry budget, a malformed response, a bug — is also a stop rather
                # than a loss. A run costs ~1,920 requests across more than a day of
                # quota; letting an exception escape here would discard every batch
                # since the last checkpoint and, worse, present as a traceback rather
                # than as resumable state. Sessions are marked done only after their
                # batch succeeds, so committing here never claims work that failed.
                self._commit(progress)
                return IngestOutcome(
                    progress,
                    completed=False,
                    stopped_reason=f"{type(exc).__name__}: {exc}",
                )

            progress.done_sessions.update(_key(namespace, s) for s in batch)
            if n % self.checkpoint_every == 0:
                self._commit(progress)
            if on_batch:
                on_batch(n, len(batches), progress)

        self._commit(progress)
        return IngestOutcome(progress, completed=True)

    def _record_sessions(self, namespace: str, batch: list[HaystackSession]) -> None:
        """Register the source sessions before their memories reference them.

        `memories.source_session_id` is a real foreign key, which is what keeps
        provenance honest — a memory cannot claim to come from a session that was
        never ingested. Turn bodies are omitted by default: storing them would copy
        the entire 245MB corpus into SQLite for no gain, since the demo can reload
        turns from the dataset by id when it needs to show one.
        """
        from datetime import datetime

        from chronomem.store import Session

        from .extract import _parse_date

        for sess in batch:
            self.store.add_session(
                Session(
                    id=sess.session_id,
                    user_id=namespace,
                    started_at=_parse_date(sess.date) or datetime.now(),
                    source=f"longmemeval:{sess.session_id}",
                    turns=[],
                )
            )

    def _ingest_batch(
        self, namespace: str, batch: list[HaystackSession], progress: IngestProgress
    ) -> None:
        self._record_sessions(namespace, batch)
        # The extractor stamps every memory it produces with this id, so it is set
        # per batch rather than per pipeline.
        self.extractor.user_id = namespace
        outcome = self.extractor.extract(batch)
        progress.extraction_requests += 1
        progress.bad_session_index += outcome.dropped_bad_index
        if not outcome.memories:
            return

        vectors = self.encoder.encode([m.content for m in outcome.memories])
        deduped = self.deduplicator.process(outcome.memories, vectors)
        progress.duplicates_dropped += deduped.duplicates
        progress.updates_detected += len(deduped.updates)
        progress.adjudication_requests += deduped.adjudications
        if not deduped.kept:
            return

        # Re-embed only the survivors so the index and the store stay in step.
        kept_vectors = self.encoder.encode([m.content for m in deduped.kept])
        self.store.add_memories(deduped.kept)
        self.index.add([m.id for m in deduped.kept], kept_vectors)
        progress.memories_written += len(deduped.kept)

        if self.resolver is not None:
            # After the write, so the batch is reconciled against itself as well as
            # against history — ten sessions routinely contain both sides of a
            # change. Costs no requests: resolution is pure SQL.
            stats = self.resolver.resolve_memories(deduped.kept)
            progress.superseded += stats.superseded
            progress.undated += stats.skipped_undated

    def _commit(self, progress: IngestProgress) -> None:
        self.index.save()
        progress.save(self.checkpoint_path)
