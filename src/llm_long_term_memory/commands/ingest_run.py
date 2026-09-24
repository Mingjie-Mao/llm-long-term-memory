"""The corpus ingestion command.

It owns the store lock, the resumable checkpoint, the quota ledger and the usage
artifact, so it is kept whole rather than split across helpers.
"""

from __future__ import annotations

import typer
from rich.table import Table

from llm_long_term_memory.commands.common import cli_lock, console, manifest_instances
from llm_long_term_memory.config import Settings
from llm_long_term_memory.evaluation.datasets import longmemeval as lme


def ingest_run(
    config: str = typer.Option("configs/baselines.yaml", "--config", "-c"),
    limit: int | None = typer.Option(None, help="Override dataset_limit from config"),
    sessions: int | None = typer.Option(None, help="Cap unique sessions (for a trial run)"),
    fresh: bool = typer.Option(False, help="Discard the checkpoint and start over"),
    store_name: str | None = typer.Option(
        None, help="Store filename stem; two-stage extraction defaults to two-stage"
    ),
    questions: str | None = typer.Option(
        None,
        "--questions",
        help=(
            "Manifest of question ids. Ingests exactly these questions' haystacks. "
            "`--limit` takes a prefix of the corpus, which is the wrong selection for "
            "a frozen split: the held-out hundred are scattered through it, not at "
            "the front."
        ),
    ),
) -> None:
    """Extract memories from the corpus. Resumes after a daily-quota stop."""
    from llm_long_term_memory.config import ExperimentConfig
    from llm_long_term_memory.embed import Encoder
    from llm_long_term_memory.ingest import (
        Deduplicator,
        Extractor,
        IngestionPipeline,
        TwoStageExtractor,
        namespace_batch_count,
        namespaced_sessions,
    )
    from llm_long_term_memory.llm import Limits, QuotaManager, UsageTracker
    from llm_long_term_memory.llm.client import GeminiClient
    from llm_long_term_memory.store import NumpyFlatIndex, SQLiteMemoryStore, scoped_session_id
    from llm_long_term_memory.temporal import TemporalResolver

    settings = Settings()
    cfg = ExperimentConfig.from_yaml(config)
    store_name = store_name or ("two-stage" if cfg.ingest.two_stage else "memories")

    # A second ingest into the same store is the failure that has actually happened
    # here: two processes ran concurrently, each recorded its own usage, and the
    # totals under-reported the spend by roughly a third — 211 requests logged
    # against ~320 actually made. The store looked fine, so the only symptom was
    # the daily quota ending early.
    with cli_lock(settings.store_dir / f"{store_name}.db", what="ingest"):
        quota = QuotaManager(
            state_dir=settings.store_dir / "quota",
            default=Limits(rpm=cfg.quota.rpm, tpm=cfg.quota.tpm, rpd=cfg.quota.rpd),
        )
        quota.load_learned()
        usage = UsageTracker()
        client = GeminiClient(settings.require_api_key(), quota=quota, usage=usage)
        encoder = Encoder(cfg.models.embedder)

        store = SQLiteMemoryStore(settings.store_dir / f"{store_name}.db")
        store.initialize()
        index = NumpyFlatIndex(
            settings.store_dir / f"{store_name}-index", dim=cfg.models.embedding_dim
        )

        extractor = (
            TwoStageExtractor(client, cfg.models.extractor)
            if cfg.ingest.two_stage
            else Extractor(client, cfg.models.extractor)
        )
        dedup = Deduplicator(
            client,
            cfg.models.extractor,
            encoder,
            store=store,
            index=index,
            threshold=cfg.ingest.dedupe_similarity_threshold,
        )
        repair = None
        if cfg.ingest.specificity_repair:
            from llm_long_term_memory.ingest.grounded import GroundedExtractor
            from llm_long_term_memory.ingest.repair import SpecificityRepair

            repair = SpecificityRepair(GroundedExtractor(client, cfg.models.extractor))
            console.print(
                "[yellow]specificity repair on[/yellow] [dim]— one grounded call for "
                "each session that lost a specific; measured 44.8% -> 59.3% retention "
                "on its registered cohort[/dim]"
            )

        resolver = TemporalResolver(store) if cfg.temporal_resolution else None
        from llm_long_term_memory.ingest.pipeline import fit_batch_size

        # ~2,560 tokens per session, measured in D5.
        per_request = fit_batch_size(
            cfg.ingest.sessions_per_request,
            quota.for_model(cfg.models.extractor).limits.tpm,
            tokens_per_session=2_560,
        )
        if per_request != cfg.ingest.sessions_per_request:
            console.print(
                f"[yellow]batch size {cfg.ingest.sessions_per_request} -> {per_request}[/yellow] "
                f"[dim]({cfg.models.extractor} allows "
                f"{quota.for_model(cfg.models.extractor).limits.tpm:,} tokens/min)[/dim]"
            )

        pipeline = IngestionPipeline(
            extractor,
            dedup,
            store,
            index,
            encoder,
            resolver=resolver,
            checkpoint_path=settings.store_dir / f"{store_name}-ingest.json",
            sessions_per_request=per_request,
            checkpoint_every=cfg.ingest.checkpoint_every,
            repair=repair,
        )

        with console.status("Loading corpus…"):
            if questions:
                # Load the whole split and filter, exactly as `eval run --questions`
                # does. A stratified sample of the manifest's size would return a
                # different set of questions, and for a held-out split the haystacks
                # ingested have to be the ones its questions are asked about.
                from llm_long_term_memory.evaluation.manifest import load_manifest

                manifest = load_manifest(questions)
                by_id = {i.question_id: i for i in manifest_instances(manifest, settings)}
                unknown = [q for q in manifest.question_ids if q not in by_id]
                if unknown:
                    raise typer.BadParameter(
                        f"{len(unknown)} question id(s) not in the {manifest.variant!r} "
                        f"split, first: {unknown[0]!r}"
                    )
                instances = [by_id[q] for q in manifest.question_ids]
                console.print(f"[dim]manifest {manifest.name}: {len(instances)} questions[/dim]")
            else:
                instances = lme.load(
                    cfg.dataset_variant, settings.data_dir, limit=limit or cfg.dataset_limit
                )
            all_sessions = namespaced_sessions(instances)
        if sessions:
            all_sessions = all_sessions[:sessions]

        n_batches = namespace_batch_count(all_sessions, per_request)
        extraction_calls_per_batch = 2 if cfg.ingest.two_stage else 1
        console.print(
            f"[bold]ingest[/bold] · {len(all_sessions):,} unique sessions · "
            f"{per_request}/batch = "
            f"~{n_batches * extraction_calls_per_batch:,} extraction requests · "
            f"extractor={cfg.models.extractor}\n"
        )

        def on_batch(n, total, progress) -> None:
            console.print(
                f"[{n}/{total}] +{progress.memories_written} memories  "
                f"dup={progress.duplicates_dropped} upd={progress.updates_detected}  "
                f"reqs={progress.extraction_requests + progress.adjudication_requests}",
                highlight=False,
            )

        usage_name = (
            "ingest.usage.json" if store_name == "memories" else f"{store_name}.ingest.usage.json"
        )
        usage_path = settings.results_dir / "raw" / usage_name
        try:
            outcome = pipeline.run(all_sessions, resume=not fresh, on_batch=on_batch)
        except BaseException:
            # Normal quota/network stops return an outcome. This covers Ctrl-C or
            # a truly unexpected crash after metered calls, so their cost is not
            # silently lost before the user resumes.
            usage.save(usage_path, merge=not fresh)
            raise
        p = outcome.progress
        usage.save(usage_path, merge=not fresh)

        t = Table(title="ingestion", show_header=False)
        t.add_column(style="cyan")
        t.add_column(justify="right")
        t.add_row("sessions ingested", f"{len(p.done_sessions):,}")
        t.add_row("content-blocked sessions", f"{len(p.blocked_sessions):,}")
        t.add_row(
            "sessions with terminal status",
            f"{len(p.done_sessions | p.blocked_sessions):,}",
        )
        terminal_keys = p.done_sessions | p.blocked_sessions
        expected_ids = {
            scoped_session_id(namespace, session.session_id) for namespace, session in all_sessions
        }
        terminal_ids = {
            scoped_session_id(namespace, session.session_id)
            for namespace, session in all_sessions
            if f"{namespace}:{session.session_id}" in terminal_keys
        }
        raw_only_pending = len((store.session_ids() & expected_ids) - terminal_ids)
        t.add_row("raw-only pending sessions", f"{raw_only_pending:,}")
        t.add_row("memories written", f"{p.memories_written:,}")
        t.add_row("duplicates dropped", f"{p.duplicates_dropped:,}")
        t.add_row("updates detected", f"{p.updates_detected:,}")
        t.add_row("bad session_index", f"{p.bad_session_index:,}")
        # Sessions that were archived and produced nothing. Reported because it was
        # invisible: every other number here looked healthy while 13-17% of
        # substantive sessions yielded no memory at all, which is where a large part
        # of the extraction-class failures come from.
        zero, subst = store.zero_yield_sessions(include_session_ids=terminal_ids)
        if subst:
            t.add_row(
                "zero-memory sessions",
                f"[{'red' if zero / subst > 0.15 else 'yellow'}]{zero:,} of {subst:,} "
                f"({zero / subst:.1%})[/]",
            )
        t.add_row("superseded", f"{p.superseded:,}")
        t.add_row("extraction requests", f"{p.extraction_requests:,}")
        t.add_row("adjudication requests", f"{p.adjudication_requests:,}")
        if repair is not None:
            # Reported whether or not it fired. A repair that silently did nothing and
            # one that was never switched on look identical in the totals otherwise.
            called = p.repair_requests
            sessions_done = max(1, len(p.done_sessions))
            t.add_row("repair requests", f"{called:,} ({called / sessions_done:.0%} of sessions)")
            t.add_row(
                "repair memories",
                f"{p.repair_memories:,} ({p.repair_memories / sessions_done:.2f}/session)",
            )
        if p.memories_written:
            t.add_row(
                "memories / session", f"{p.memories_written / max(1, len(p.done_sessions)):.1f}"
            )
        if resolver is not None:
            # A final pass over every key: incremental resolution can leave a key stale
            # when a fact arrives in a later batch than the one it belongs before.
            final = resolver.resolve_everything()
            t.add_row("final pass: superseded", f"{final.superseded:,}")
            t.add_row("final pass: restatements", f"{final.restatements:,}")
            t.add_row("undated (unresolvable)", f"{final.skipped_undated:,}")
        console.print(t)
        by_type: dict[str, int] = {}
        for ns in store.user_ids():
            for k, v in store.count_by_type(ns).items():
                by_type[k] = by_type.get(k, 0) + v
        console.print(dict(sorted(by_type.items(), key=lambda kv: -kv[1])))

        if not outcome.completed:
            console.print(f"\n[yellow]Stopped:[/yellow] {outcome.stopped_reason}")
            console.print("[dim]Rerun the same command tomorrow — it resumes.[/dim]")
        store.close()
        if not outcome.completed:
            # A quota stop is expected and resumable, but it is not success. Cron,
            # CI and the Codex continuation must be able to distinguish "checkpoint
            # safely written" from "all requested sessions reached a terminal
            # state" without scraping prose from the terminal.
            raise typer.Exit(code=2)
