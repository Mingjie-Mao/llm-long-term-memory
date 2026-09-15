"""Two dedup policies, one extraction pass.

Deduplication is not namespace-scoped: `Deduplicator._neighbours` searches the shared index
with no `user_id` condition while retrieval applies one, so a fact from one conversation can
be dropped as a DUPLICATE of a fact in another
([`dedup-namespace-leak.json`](../results/analysis/dedup-namespace-leak.json)). Fixing it
changes which memories are written, so the fixed system is not the frozen one and the two
have to be measurable side by side.

Ingesting twice would cost 1,024 extraction requests and about 16M input tokens. It does not
have to: **extraction happens before deduplication and is identical under both policies.**
So this records what the extractor returned on the first pass and replays it on the second,
where the only thing that runs again is dedup.

    pass 1   real extractor, frozen dedup      512 extraction requests + its adjudications
    pass 2   cached extractor, scoped dedup    0 extraction requests + its adjudications

The saving is the expensive, predictable half. Adjudication still runs twice, because a
dropped memory changes the store and therefore every later neighbour search — the two
policies genuinely diverge there, and pretending otherwise would produce a second store that
no run could have produced.

Both passes go through `lltm ingest run` itself rather than a private copy of the pipeline,
so what is measured is the path the real run takes. `--execute` is required to spend anything;
without it a fake provider drives both passes and nothing is sent.

    python3 tools/dual_policy_ingest.py --questions results/manifests/beam-dev.json
    python3 tools/dual_policy_ingest.py --questions results/manifests/beam-dev.json --execute
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

PREREG = "results/prereg-beam-v1.md"


def stop_reason(settings, model: str = "gemini-3.1-flash-lite") -> dict:
    """Why the ingest stopped, from the persisted daily counter rather than from the word
    "quota" in a message.

    The pipeline returns the same incomplete outcome whether the day's budget ran out or
    the provider answered 503, and the two call for opposite responses: one is "come back
    tomorrow", the other is "run it again now". Reporting the first when it was the second
    costs a day for nothing, which is what happened on the first attempt.
    """
    import json as _json

    state = settings.store_dir / "quota" / f"quota-{model}.json"
    limits = settings.store_dir / "quota" / "observed-limits.json"
    used = cap = None
    if state.is_file():
        used = _json.loads(state.read_text(encoding="utf-8")).get("count")
    if limits.is_file():
        cap = (_json.loads(limits.read_text(encoding="utf-8")).get(model) or {}).get("rpd")
    exhausted = used is not None and cap is not None and used >= cap
    return {
        "model": model,
        "requests_today": used,
        "daily_cap": cap,
        "daily_budget_exhausted": exhausted,
        "advice": (
            "the day's budget is spent; rerun tomorrow and it resumes"
            if exhausted
            else "not a quota stop — the provider failed or the run was interrupted. "
            "Rerun now; it resumes from the checkpoint and does not re-extract."
        ),
    }


def _quota_stop_types() -> tuple[type, ...]:
    """Everything `lltm ingest run` can raise to mean "stopped, checkpoint written"."""
    import typer

    return tuple({SystemExit, typer.Exit})


_QUOTA_STOP = _quota_stop_types()
OUT = REPO / "results/analysis/dual-policy-ingest.json"

_DATETIME_FIELDS = (
    "event_time",
    "valid_from",
    "valid_to",
    "ingested_at",
    "last_accessed_at",
    "strength_updated_at",
)


def batch_key(namespace: str, sessions) -> str:
    """Identifies a batch by what went into it, not by its position.

    Batches are formed from the session list and never from the store, so the same key
    appears in both passes even though the stores diverge.
    """
    raw = "|".join(f"{namespace}:{session.session_id}" for session in sessions)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def to_record(memory) -> dict:
    row = dataclasses.asdict(memory)
    for field in _DATETIME_FIELDS:
        if isinstance(row.get(field), datetime):
            row[field] = row[field].isoformat()
    row["type"] = str(row["type"])
    row["status"] = str(row["status"])
    return row


def from_record(row: dict):
    from llm_long_term_memory.store import Memory

    row = dict(row)
    for field in _DATETIME_FIELDS:
        if row.get(field):
            row[field] = datetime.fromisoformat(row[field])
    return Memory(**row)


class RecordingExtractor:
    """The real extractor, writing down what it returned.

    Delegates `version`, `model` and `prompt_texts` so the ingest fingerprint describes the
    extractor that actually ran; a wrapper that fingerprinted as itself would make the store
    claim it was written by something that does not exist.
    """

    def __init__(self, inner, cache_path: Path) -> None:
        self.inner = inner
        self.cache_path = cache_path
        self.user_id = getattr(inner, "user_id", "user")

    version = property(lambda self: self.inner.version)
    model = property(lambda self: self.inner.model)

    def prompt_texts(self):
        return self.inner.prompt_texts()

    def extract(self, sessions):
        self.inner.user_id = self.user_id
        outcome = self.inner.extract(sessions)
        with self.cache_path.open("a", encoding="utf-8") as sink:
            sink.write(
                json.dumps(
                    {
                        "key": batch_key(self.user_id, sessions),
                        "namespace": self.user_id,
                        "requests": outcome.requests,
                        "dropped_bad_index": outcome.dropped_bad_index,
                        "memories": [to_record(memory) for memory in outcome.memories],
                    }
                )
                + "\n"
            )
            sink.flush()  # a quota stop must not lose the batch that was just paid for
        return outcome


class CachedExtractor:
    """Replays a recorded pass. Costs nothing and cannot reach a provider."""

    def __init__(self, inner, cache_path: Path) -> None:
        self.inner = inner
        self.user_id = "user"
        self.batches = {}
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                self.batches[record["key"]] = record

    version = property(lambda self: self.inner.version)
    model = property(lambda self: self.inner.model)

    def prompt_texts(self):
        return self.inner.prompt_texts()

    def extract(self, sessions):
        from llm_long_term_memory.ingest.extract import ExtractionOutcome

        key = batch_key(self.user_id, sessions)
        record = self.batches.get(key)
        if record is None:
            # Refused rather than extracted: falling through to the real extractor would
            # spend quota the second pass was built to avoid, and silently.
            raise KeyError(
                f"batch {key} for namespace {self.user_id!r} is not in the recorded pass; "
                "the two passes were given different sessions"
            )
        return ExtractionOutcome(
            memories=[from_record(row) for row in record["memories"]],
            dropped_bad_index=record["dropped_bad_index"],
            requests=0,
        )


def run_pass(
    cli, ingest_module, dedup_module, args, *, store_name, extractor_factory, scoped, fresh
):
    """One ingest through the real CLI, under one dedup policy.

    Returns the objects it built, plus whether the pass reached the end. A daily-quota stop
    is the normal path here — 512 extraction requests do not fit in a 500-request day — and
    the CLI signals it by exiting 2 after checkpointing. That is caught rather than
    propagated so the caller can record the state and say "resume tomorrow"."""
    import dedup_fix_behavioral_diff as diff_tool

    real = ingest_module.TwoStageExtractor
    neighbours = dedup_module.Deduplicator._neighbours
    made = {}

    def factory(client, model, *a, **kw):
        inner = real(client, model, *a, **kw)
        wrapper = extractor_factory(inner)
        made["extractor"] = wrapper
        made["client"] = client
        return wrapper

    ingest_module.TwoStageExtractor = factory
    if scoped:
        dedup_module.Deduplicator._neighbours = diff_tool.namespace_scoped(neighbours)
    try:
        cli.ingest_run(
            config=args.config,
            limit=None,
            sessions=None,
            fresh=fresh,
            store_name=store_name,
            questions=args.questions,
        )
        made["complete"] = True
    except _QUOTA_STOP as exc:
        # `typer.Exit(2)` on a quota stop. Typer's Exit derives from click's, not from
        # SystemExit, so catching SystemExit here caught nothing and the tool died with a
        # traceback instead of saying "resume tomorrow" — found by drilling the stop
        # against a fake provider rather than by meeting it 500 requests into a real run.
        # Never `fresh` again after this: the checkpoint is the only thing standing
        # between a resumed run and a second full extraction.
        if getattr(exc, "exit_code", getattr(exc, "code", None)) != 2:
            raise
        made["complete"] = False
    finally:
        ingest_module.TwoStageExtractor = real
        dedup_module.Deduplicator._neighbours = neighbours
    return made


def store_summary(settings, store_name: str) -> dict:
    from llm_long_term_memory.store import SQLiteMemoryStore

    store = SQLiteMemoryStore(settings.store_dir / f"{store_name}.db", read_only=True)
    store.initialize()
    by_namespace = {
        namespace: sum(1 for _ in store.iter_all(namespace))
        for namespace in sorted(store.user_ids())
    }
    ids = {memory.id for namespace in by_namespace for memory in store.iter_all(namespace)}
    store.close()
    return {"memories": sum(by_namespace.values()), "by_namespace": by_namespace, "ids": ids}


def _run(args) -> int:
    if args.execute and os.environ.get("BEAM_PREREG") != PREREG:
        print(f"STOP: --execute needs BEAM_PREREG={PREREG}")
        return 2

    if not args.execute:
        work = Path(args.work or REPO / "stores/dual-policy-dry").resolve()
        (work / "stores").mkdir(parents=True, exist_ok=True)
        (work / "results").mkdir(parents=True, exist_ok=True)
        os.environ["LLTM_STORE_DIR"] = str(work / "stores")
        os.environ["LLTM_RESULTS_DIR"] = str(work / "results")
        os.environ.setdefault("GEMINI_API_KEY", "dry-run-no-request-is-sent")

    import beam_dry_run

    from llm_long_term_memory import cli
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.ingest import dedup as dedup_module
    from llm_long_term_memory.llm import client as client_module

    ingest_module = sys.modules["llm_long_term_memory.ingest"]
    settings = Settings()
    if not args.execute:
        client_module.GeminiClient = lambda *a, **k: beam_dry_run.FakeProvider()

    cache = settings.store_dir / f"{args.store_name}-extraction-cache.jsonl"
    frozen_name, scoped_name = f"{args.store_name}-frozen", f"{args.store_name}-scoped"
    state_path = settings.store_dir / f"{args.store_name}-dual-policy-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}

    def remember(**fields):
        state.update(fields)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    first: dict = {}
    if not state.get("recording_complete"):
        # The existence of a cache file says a pass started, never that it finished. A
        # partial cache replayed as if it were whole would build a second store missing
        # whatever the quota stop interrupted, and nothing downstream would notice.
        started = bool(state.get("recording_started"))
        print(f"pass 1: real extractor, frozen dedup -> {frozen_name}")
        if started:
            print("  resuming from the checkpoint; the recorded batches are kept")
        remember(recording_started=True)
        first = run_pass(
            cli,
            ingest_module,
            dedup_module,
            args,
            store_name=frozen_name,
            extractor_factory=lambda inner: RecordingExtractor(inner, cache),
            scoped=False,
            fresh=not started,
        )
        remember(recording_complete=bool(first.get("complete")))
        if not first.get("complete"):
            reason = stop_reason(settings)
            remember(last_stop=reason)
            print(
                f"\nSTOPPED with the checkpoint written after "
                f"{reason['requests_today']} of {reason['daily_cap']} requests today.\n"
                f"{reason['advice']}"
            )
            return 2
    else:
        print(f"the recorded extraction pass is complete ({cache})")

    print(f"\npass 2: cached extractor, namespace-scoped dedup -> {scoped_name}")
    second = run_pass(
        cli,
        ingest_module,
        dedup_module,
        args,
        store_name=scoped_name,
        extractor_factory=lambda inner: CachedExtractor(inner, cache),
        scoped=True,
        fresh=not state.get("replay_started"),
    )
    remember(replay_started=True, replay_complete=bool(second.get("complete")))
    if not second.get("complete"):
        reason = stop_reason(settings)
        remember(last_stop=reason)
        print(
            f"\nSTOPPED during the replay after {reason['requests_today']} of "
            f"{reason['daily_cap']} requests today. Extraction is already paid for.\n"
            f"{reason['advice']}"
        )
        return 2

    frozen, scoped = store_summary(settings, frozen_name), store_summary(settings, scoped_name)
    saved = sorted(scoped["ids"] - frozen["ids"])
    lost = sorted(frozen["ids"] - scoped["ids"])
    batches = sum(1 for line in cache.read_text(encoding="utf-8").splitlines() if line.strip())

    def extraction_calls(made: dict) -> int | None:
        """What the pass actually asked a provider for, counted from the client itself."""
        client = made.get("client")
        kinds = getattr(client, "by_kind", None)
        if kinds is None:
            return None
        return kinds.get("extract_stage_a", 0) + kinds.get("extract_stage_b", 0)

    replay_calls = extraction_calls(second)
    if replay_calls:
        # The replay exists to spend nothing. If it reached a provider, the cache missed
        # and the saving is imaginary — worth failing over rather than reporting.
        print(f"STOP: the replay pass made {replay_calls} extraction request(s)")
        return 1
    payload = {
        "name": "dual-policy-ingest",
        "store_name": args.store_name,
        "executed": args.execute,
        "recorded_batches": batches,
        "extraction_requests": {
            "recording_pass": extraction_calls(first),
            "replay_pass": replay_calls,
        },
        "frozen": {k: v for k, v in frozen.items() if k != "ids"},
        "scoped": {k: v for k, v in scoped.items() if k != "ids"},
        "memories_the_fix_saved": len(saved),
        "memories_only_in_the_frozen_store": len(lost),
        "conversations_affected": sorted(
            {
                namespace
                for namespace in scoped["by_namespace"]
                if scoped["by_namespace"][namespace] != frozen["by_namespace"].get(namespace)
            }
        ),
        "note": "Extraction ran once. Deduplication ran twice, because a dropped memory "
        "changes the store and therefore every later neighbour search.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\n" + json.dumps(payload, indent=2))
    print(f"\nwritten: {OUT.relative_to(REPO)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True, help="manifest of question ids")
    parser.add_argument("--config", default="configs/v2.yaml")
    parser.add_argument("--store-name", default="beam-dev")
    parser.add_argument("--execute", action="store_true", help="spend quota; needs BEAM_PREREG")
    parser.add_argument("--work", default=None, help="dry-run work directory")
    parser.add_argument(
        "--retry-transient",
        type=int,
        default=0,
        help="resume this many times after a non-quota stop (a provider 5xx)",
    )
    parser.add_argument("--retry-wait", type=float, default=120.0, help="seconds between tries")
    args = parser.parse_args()

    from llm_long_term_memory.config import Settings

    attempt = 0
    while True:
        code = _run(args)
        if code != 2 or attempt >= args.retry_transient:
            return code
        # Only a transient stop is retried. A spent day is not a failure to work around:
        # continuing past it is what the daily cap exists to prevent, and the checkpoint is
        # already written.
        reason = stop_reason(Settings()) if args.execute else {"daily_budget_exhausted": False}
        if reason.get("daily_budget_exhausted"):
            print("the day's budget is spent; not retrying")
            return code
        attempt += 1
        print(
            f"\ntransient stop; retry {attempt} of {args.retry_transient} "
            f"in {args.retry_wait:.0f}s\n",
            flush=True,
        )
        time.sleep(args.retry_wait)


if __name__ == "__main__":
    raise SystemExit(main())
