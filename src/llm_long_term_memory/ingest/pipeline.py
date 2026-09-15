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

from llm_long_term_memory.embed import Encoder
from llm_long_term_memory.evaluation.datasets.longmemeval import HaystackSession, Instance
from llm_long_term_memory.llm.client import ContentBlocked, DailyQuotaExhausted
from llm_long_term_memory.store import MemoryStore, NumpyFlatIndex

from . import fingerprint
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
    blocked_sessions: set[str] = field(default_factory=set)
    """Sessions in a batch the provider refused on content policy. They yielded no
    memories and never will, so the gap is named rather than left to look like an
    extraction failure."""
    blocked_batches: int = 0

    def to_dict(self) -> dict:
        return {
            "done_sessions": sorted(self.done_sessions),
            "blocked_sessions": sorted(self.blocked_sessions),
            "blocked_batches": self.blocked_batches,
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
        d = json.loads(path.read_text(encoding="utf-8"))
        p = cls(
            done_sessions=set(d.get("done_sessions", [])),
            # Restored, or a resumed run reports zero blocked batches while the
            # store is missing their sessions — the gap would lose its name at the
            # first quota stop.
            blocked_sessions=set(d.get("blocked_sessions", [])),
        )
        for key in (
            "blocked_batches",
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
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(path)  # atomic, so a kill mid-write cannot corrupt the checkpoint


def namespaced_sessions(instances: list[Instance]) -> list[tuple[str, HaystackSession]]:
    """(namespace, session) pairs, one namespace per store.

    Deliberately not deduplicated across namespaces: a session appearing in two
    LongMemEval haystacks belongs to two different simulated users and must be
    extracted into both stores. Deduplicating it merges the personas.

    Within a namespace it is deduplicated, because questions can share one. A
    conversation carries twenty questions over the same sessions, and keying on the
    question would extract that conversation twenty times into one store.
    """
    out: list[tuple[str, HaystackSession]] = []
    seen: set[tuple[str, str]] = set()
    for inst in instances:
        namespace = inst.store_namespace
        for sess in inst.sessions:
            if (namespace, sess.session_id) in seen:
                continue
            seen.add((namespace, sess.session_id))
            out.append((namespace, sess))
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


TOKENS_PER_SESSION = 2_560
"""Measured in D5. Named because two callers need the same number."""


def resolved_sessions_per_request(configured: int, tpm: int) -> int:
    """The batch size ingestion will actually use.

    Batch size is part of the ingest fingerprint, and the configured value is not
    it — the model's token budget can lower it. Both the ingestion driver and the
    preflight check resolve it through here, so a store whose fingerprint says 15
    is not compared against a configuration that merely asked for 15.
    """
    return fit_batch_size(configured, tpm, tokens_per_session=TOKENS_PER_SESSION)


def _key(namespace: str, session: HaystackSession) -> str:
    """Checkpoint key. Namespaced, because the same session in two questions is two
    separate units of work."""
    return f"{namespace}:{session.session_id}"


def batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def namespace_batch_count(
    pairs: list[tuple[str, HaystackSession]], sessions_per_request: int
) -> int:
    """Count batches using the same namespace boundary as the ingestion driver.

    ``ceil(total_sessions / batch_size)`` undercounts whenever a namespace ends
    with a partial batch. LongMemEval dev-50 has fifty namespaces, so that error is
    large enough to make a run appear to fit within a daily request quota when it
    does not.
    """
    return sum(
        -(-len(sessions) // sessions_per_request) for _, sessions in group_by_namespace(pairs)
    )


@dataclass(slots=True)
class IngestOutcome:
    progress: IngestProgress
    completed: bool
    stopped_reason: str | None = None


class ExtractorChanged(RuntimeError):
    """A resume would append memories from a different extractor than the store holds."""


class ConfigurationMismatch(RuntimeError):
    """The objects about to run are not the ones the resolved configuration describes."""


class IngestionPipeline:
    source_label = "longmemeval"
    """Written as `<label>:<session id>` into each stored session's `source`, which is how a
    store records the dataset its sessions came from."""

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
        config_spec: fingerprint.IngestSpec | None = None,
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
        # The spec derived from resolved configuration, when the caller has one.
        # Preflight validates a store against the *configuration*; this is what
        # makes that validation mean something about the run — see `run()`.
        self.config_spec = config_spec

    def run(
        self,
        pairs: list[tuple[str, HaystackSession]],
        resume: bool = True,
        on_batch=None,
    ) -> IngestOutcome:
        """`pairs` is (namespace, session); one namespace per question."""
        # Stamp the store with the extractor that wrote it, so a result row can
        # report the version of the *data* rather than of the checked-out code.
        #
        # Stamping used to be unconditional, on the reasoning that "a re-ingest under
        # a new extractor correctly relabels the store it is rebuilding". That is
        # true of a rebuild and false of a resume, and this code could not tell them
        # apart. It cost a real store: 4,843 memories written by the pre-P10
        # extractor were resumed to completion by the P10 extractor, which files
        # ~53% of facts under `assistant` where its predecessor filed ~7%. Both runs
        # succeeded, the checkpoint stayed consistent, and the store passed every
        # structural check while being the product of two systems. Section 10 of the
        # engineering report is the post-mortem.
        #
        # So: a resume into a non-empty store must match what is already there.
        # `--fresh` remains the way to rebuild under a new extractor, and it is the
        # honest one, because it replaces the data rather than relabelling it.
        #
        # The comparison is against a fingerprint of the *inputs* — prompt text,
        # schema text, model, batch size — not against `extractor_version` alone.
        # A version string is edited by hand, so it catches a deliberate generation
        # change and misses the likelier one: a prompt reworded with the version left
        # alone, which produces exactly the same mixed store with nothing to notice
        # it by.
        if hasattr(self.store, "set_meta"):
            spec = fingerprint.from_runtime(
                self.extractor,
                sessions_per_request=self.sessions_per_request,
                dedup=self.deduplicator,
            )
            # The second layer. Preflight compares the store against a fingerprint
            # derived from configuration, which is only evidence about this run if
            # the objects built from that configuration match it. Checking it here
            # is what closes the gap: preflight answers "is the config what wrote
            # this store?", and this answers "is the runtime what the config says?".
            #
            # It is cheap and it runs before any request is spent, so a wiring
            # mistake — an extractor constructed with a different model than the
            # config names — stops here rather than at the end of a day of quota.
            if self.config_spec is not None:
                drifted = fingerprint.differences(self.config_spec.as_dict(), spec.as_dict())
                if drifted:
                    raise ConfigurationMismatch(
                        "the ingestion objects do not match the resolved "
                        "configuration:\n  "
                        + "\n  ".join(drifted)
                        + "\nThe fingerprint written to the store would describe "
                        "something other than what ran, and every check that reads "
                        "it downstream would inherit that."
                    )

            current = spec.as_dict()
            previous = fingerprint.loads(self.store.get_meta("ingest_fingerprint"))
            if resume and previous and self.store.count():
                moved = fingerprint.differences(previous, current)
                if moved:
                    raise ExtractorChanged(
                        "this store was written by a different ingestion setup:\n  "
                        + "\n  ".join(moved)
                        + "\nContinuing would produce a store whose memories come "
                        "from two different systems, labelled as one. Rebuild with "
                        "--fresh, or ingest into a new --store-name."
                    )
            self.store.set_meta("ingest_fingerprint", fingerprint.dumps(current))
            # Kept as its own key: result rows report it, and it stays readable
            # without parsing JSON.
            self.store.set_meta("extractor_version", current["extractor_version"])

        progress = IngestProgress.load(self.checkpoint_path) if resume else IngestProgress()
        # A content-policy refusal is deterministic for an unchanged prompt and
        # temperature, so retrying it on every resume can never fill the gap. Keep
        # it separately reported, but treat it as terminal work for scheduling.
        terminal_sessions = progress.done_sessions | progress.blocked_sessions
        pending = [p for p in pairs if _key(*p) not in terminal_sessions]

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
            except ContentBlocked:
                # The provider refused this prompt on content policy. Unlike every
                # other failure here it will refuse identically on the next run, so
                # halting turns one batch into a permanently stuck ingest — which is
                # what it did on the held-out corpus at batch 76 of 367.
                #
                # Recorded rather than skipped silently. These sessions produce no
                # memories, and downstream that is indistinguishable from the
                # extractor having found nothing, which is exactly the confusion a
                # held-out result cannot afford. The ids go in the checkpoint so the
                # gap has a name and a size.
                #
                # Keep the refused source text for lossless raw fallback. Other
                # failed calls are deliberately not archived: they remain pending,
                # so storing them would make an unfinished batch look like a
                # completed zero-memory result.
                self._record_sessions(namespace, batch)
                progress.blocked_sessions.update(_key(namespace, s) for s in batch)
                progress.blocked_batches += 1
                if on_batch:
                    print(f"  blocked by content policy: {namespace}, {len(batch)} sessions")
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
        never ingested. Turn bodies are deliberately retained: structured memories
        are an index and temporal state, not a lossy replacement for the answer-
        bearing source evidence that a later query may need to hydrate.
        """
        from datetime import datetime

        from llm_long_term_memory.store import Session, Turn, scoped_session_id

        from .extract import _parse_date

        for sess in batch:
            stored_session_id = scoped_session_id(namespace, sess.session_id)
            self.store.add_session(
                Session(
                    id=stored_session_id,
                    user_id=namespace,
                    started_at=_parse_date(sess.date) or datetime.now(),
                    source=f"{self.source_label}:{sess.session_id}",
                    turns=[
                        Turn(
                            id=f"{stored_session_id}:{turn_index}",
                            session_id=stored_session_id,
                            turn_index=turn_index,
                            role=turn.role,
                            content=turn.content,
                            ts=_parse_date(sess.date) or datetime.now(),
                        )
                        for turn_index, turn in enumerate(sess.turns)
                    ],
                )
            )

    def _ingest_batch(
        self, namespace: str, batch: list[HaystackSession], progress: IngestProgress
    ) -> None:
        # The extractor stamps every memory it produces with this id, so it is set
        # per batch rather than per pipeline.
        self.extractor.user_id = namespace
        outcome = self.extractor.extract(batch)
        # Archive only once extraction has succeeded. A quota/network failure is
        # pending work, not a genuine zero-memory session.
        self._record_sessions(namespace, batch)
        from llm_long_term_memory.store import scoped_session_id

        for memory in outcome.memories:
            if memory.source_session_id:
                memory.source_session_id = scoped_session_id(namespace, memory.source_session_id)
        progress.extraction_requests += outcome.requests
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
