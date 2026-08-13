"""ChronoMem command line.

P0 commands are the ones that need no API key: download the benchmark, measure it,
and plan the ingestion budget against the free-tier quota.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from chronomem.config import Settings
from chronomem.evaluation.datasets import longmemeval as lme

app = typer.Typer(add_completion=False, help="ChronoMem — long-term memory for LLM agents")
data_app = typer.Typer(help="Benchmark data: download, inspect, plan")
app.add_typer(data_app, name="data")

console = Console()


def _mask(secret: str) -> str:
    """Enough to confirm which key is loaded, not enough to leak it."""
    s = secret.strip()
    return f"{s[:4]}…{s[-4:]} ({len(s)} chars)" if len(s) >= 12 else "set (too short?)"


@app.command()
def doctor() -> None:
    """Check that credentials and data are where the pipeline expects them."""
    settings = Settings()
    t = Table(show_header=False)
    t.add_column(style="cyan")
    t.add_column()

    if settings.has_api_key:
        t.add_row("GEMINI_API_KEY", f"[green]✓[/green] {_mask(settings.gemini_api_key)}")
    else:
        t.add_row("GEMINI_API_KEY", "[red]✗ not found[/red]")

    for label, path in (
        ("data dir", settings.data_dir),
        ("store dir", settings.store_dir),
        ("results dir", settings.results_dir),
    ):
        mark = "[green]✓[/green]" if path.exists() else "[dim]— (created on first use)[/dim]"
        t.add_row(label, f"{mark} {path}")

    for variant, filename in lme.VARIANTS.items():
        p = settings.data_dir / filename
        if p.exists():
            t.add_row(f"longmemeval_{variant}", f"[green]✓[/green] {p.stat().st_size / 1e6:.0f} MB")
        else:
            t.add_row(f"longmemeval_{variant}", "[dim]— not downloaded[/dim]")

    console.print(t)

    if not settings.has_api_key:
        console.print(
            "\n[yellow]No API key.[/yellow] Get a free one (no card) at "
            "https://aistudio.google.com/apikey, then:\n"
            "  [cyan]cp .env.example .env[/cyan]  and put the key in it.\n"
            "[dim].env is gitignored. Never commit it or paste it into a chat.[/dim]"
        )


@data_app.command("download")
def data_download(
    variant: str = typer.Option("s", help="s | m | oracle"),
    data_dir: str = typer.Option("data", help="Where to store the dataset"),
    force: bool = typer.Option(False, help="Re-download even if present"),
) -> None:
    """Fetch a LongMemEval variant from HuggingFace (no auth required)."""
    with console.status(f"Downloading longmemeval_{variant}…"):
        path = lme.download(variant, data_dir, force=force)
    size_mb = path.stat().st_size / 1e6
    console.print(f"[green]✓[/green] {path}  ({size_mb:.1f} MB)")


@data_app.command("stats")
def data_stats(
    variant: str = typer.Option("s", help="s | m | oracle"),
    data_dir: str = typer.Option("data"),
    limit: int | None = typer.Option(None, help="Only load the first N questions"),
) -> None:
    """Measure the corpus. This is what decides the ingestion strategy."""
    with console.status("Loading…"):
        instances = lme.load(variant, data_dir, limit=limit)
        stats = lme.compute_stats(instances, variant=variant)

    t = Table(title=f"longmemeval_{variant}", show_header=False)
    t.add_column(style="cyan")
    t.add_column(justify="right")
    t.add_row("questions", f"{stats.n_questions:,}")
    t.add_row("  of which abstention", f"{stats.n_abstention:,}")
    t.add_row("sessions (with repeats)", f"{stats.n_sessions:,}")
    t.add_row("sessions (unique)", f"{stats.n_unique_sessions:,}")
    t.add_row(
        "[bold]session sharing factor[/bold]", f"[bold]{stats.session_sharing_factor:.2f}x[/bold]"
    )
    t.add_row("turns", f"{stats.n_turns:,}")
    t.add_row("characters", f"{stats.total_chars:,}")
    t.add_row("est. tokens", f"{stats.est_total_tokens:,}")
    t.add_row("median sessions / question", f"{stats.median_sessions_per_q:,.0f}")
    t.add_row("median est. tokens / question", f"{stats.median_tokens_per_q:,.0f}")
    console.print(t)

    qt = Table(title="question types")
    qt.add_column("type", style="cyan")
    qt.add_column("n", justify="right")
    for name, n in stats.question_types.items():
        qt.add_row(name, str(n))
    console.print(qt)

    console.print(
        "\n[dim]Token counts are a chars/4 estimate. Exact counts require the "
        "provider tokenizer and are recomputed once an API key is configured.[/dim]"
    )


@data_app.command("plan")
def data_plan(
    variant: str = typer.Option("s"),
    data_dir: str = typer.Option("data"),
    rpd: int = typer.Option(1_500, help="Requests/day allowed by your quota tier"),
) -> None:
    """How many days does one full ingestion take at each batch size?

    On a request-capped free tier this, not cost, is the schedule.
    """
    with console.status("Loading…"):
        instances = lme.load(variant, data_dir)
        stats = lme.compute_stats(instances, variant=variant)

    t = Table(title=f"Ingestion budget — longmemeval_{variant} @ {rpd:,} requests/day")
    t.add_column("sessions/request", justify="right", style="cyan")
    t.add_column("requests", justify="right")
    t.add_column("est. tokens/request", justify="right")
    t.add_column("days", justify="right")

    for spr in (1, 5, 10, 20, 40):
        plan = lme.plan_ingestion(stats, spr, rpd=rpd, dedupe_sessions=True)
        days = f"{plan.days_at_quota:.1f}"
        style = (
            "green" if plan.fits_in_one_day else ("yellow" if plan.days_at_quota <= 3 else "red")
        )
        t.add_row(
            str(spr),
            f"{plan.n_requests:,}",
            f"{plan.est_tokens_per_request:,}",
            f"[{style}]{days}[/{style}]",
        )
    console.print(t)
    console.print(
        "\n[dim]Assumes each unique session is extracted once and its memories reused "
        "across every question referencing it. Larger batches cost fewer requests but "
        "degrade extraction quality — pick the smallest batch that fits the schedule.[/dim]"
    )


_ALL_VARIANTS = typer.Argument(None, help="Defaults to every run found")

eval_app = typer.Typer(help="Run and report evaluations")
app.add_typer(eval_app, name="eval")


def _build(variant: str, cfg_path: str):
    """Wire up client, judge, and runner for one variant."""
    from chronomem.config import ExperimentConfig
    from chronomem.embed import Encoder
    from chronomem.evaluation.judge import Judge
    from chronomem.evaluation.runners.full_context import FullContextRunner
    from chronomem.evaluation.runners.memory import MemoryRunner
    from chronomem.evaluation.runners.naive_rag import NaiveRAGRunner
    from chronomem.llm import Limits, QuotaManager, UsageTracker
    from chronomem.llm.client import GeminiClient

    settings = Settings()
    cfg = ExperimentConfig.from_yaml(cfg_path)

    quota = QuotaManager(
        state_dir=settings.store_dir / "quota",
        default=Limits(rpm=cfg.quota.rpm, tpm=cfg.quota.tpm, rpd=cfg.quota.rpd),
    )
    learned = quota.load_learned()
    if learned:
        console.print(
            "[dim]observed limits: "
            + ", ".join(f"{m} rpd={lim.rpd} rpm={lim.rpm}" for m, lim in sorted(learned.items()))
            + "[/dim]"
        )
    usage = UsageTracker()
    client = GeminiClient(settings.require_api_key(), quota=quota, usage=usage)
    judge = Judge(client, model=cfg.models.judge)

    if variant == "full_context":
        runner = FullContextRunner(client, model=cfg.models.answerer)
    elif variant == "naive_rag":
        runner = NaiveRAGRunner(client, model=cfg.models.answerer, encoder=Encoder())
    elif variant in ("chronomem", "chronomem_no_temporal"):
        from chronomem.store import NumpyFlatIndex, SQLiteMemoryStore

        store = SQLiteMemoryStore(settings.store_dir / "memories.db")
        store.initialize()
        index = NumpyFlatIndex(settings.store_dir / "memories-index", dim=cfg.models.embedding_dim)
        if not len(index):
            raise typer.BadParameter("the memory store is empty — run `chronomem ingest run` first")
        runner = MemoryRunner(
            client,
            model=cfg.models.answerer,
            encoder=Encoder(),
            store=store,
            index=index,
            top_k=cfg.retrieval.top_k,
            temporal=(variant == "chronomem"),
        )
        runner.name = variant
    else:
        raise typer.BadParameter(f"unknown variant {variant!r}")
    return cfg, settings, runner, judge, usage


@eval_app.command("run")
def eval_run(
    variant: str = typer.Argument(
        ..., help="full_context | naive_rag | chronomem | chronomem_no_temporal"
    ),
    config: str = typer.Option("configs/baselines.yaml", "--config", "-c"),
    limit: int | None = typer.Option(None, help="Override dataset_limit from config"),
    fresh: bool = typer.Option(False, help="Ignore existing results and start over"),
) -> None:
    """Evaluate one variant on LongMemEval. Resumes automatically if interrupted."""
    from chronomem.evaluation.harness import run_eval
    from chronomem.evaluation.report import render_summary

    cfg, settings, runner, judge, usage = _build(variant, config)
    n = limit if limit is not None else cfg.dataset_limit

    with console.status("Loading dataset…"):
        instances = lme.load(cfg.dataset_variant, settings.data_dir, limit=n)

    out = settings.results_dir / "raw" / f"{variant}.jsonl"
    console.print(
        f"[bold]{variant}[/bold] · {len(instances)} questions · "
        f"answerer={cfg.models.answerer} judge={cfg.models.judge}\n"
        f"[dim]→ {out}[/dim]\n"
    )

    total = len(instances)

    def progress(result, report) -> None:
        mark = "[green]✓[/green]" if result.correct else "[red]✗[/red]"
        console.print(
            f"{mark} [{report.n}/{total}] {result.question_id[:20]:<20} "
            f"{result.question_type:<26} acc={report.accuracy:.1%} "
            f"ctx={result.context_tokens:,}",
            highlight=False,
        )

    report = run_eval(
        runner, judge, instances, out, usage=usage, resume=not fresh, on_progress=progress
    )

    console.print()
    console.print(render_summary(report))
    if not report.completed:
        console.print(
            f"\n[yellow]Stopped on quota:[/yellow] {report.stopped_reason}\n"
            f"[dim]Rerun the same command tomorrow — it resumes from {report.n}.[/dim]"
        )


@eval_app.command("report")
def eval_report(
    variants: list[str] = _ALL_VARIANTS,
    out: str = typer.Option("results/table.md", help="Where to write the markdown table"),
) -> None:
    """Regenerate the results table from run artifacts."""
    from chronomem.evaluation.report import load_report, render_table

    settings = Settings()
    raw = settings.results_dir / "raw"
    if not raw.exists():
        console.print("[yellow]No runs found.[/yellow] Try `chronomem eval run full_context`.")
        raise typer.Exit(1)

    order = variants or [p.stem for p in sorted(raw.glob("*.jsonl"))]
    reports = [
        load_report(raw / f"{v}.jsonl", variant=v) for v in order if (raw / f"{v}.jsonl").exists()
    ]
    if not reports:
        console.print("[yellow]No matching runs.[/yellow]")
        raise typer.Exit(1)

    table = render_table(reports)
    console.print(table)
    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(table + "\n")
    console.print(f"\n[green]✓[/green] {dest}")


@eval_app.command("label")
def eval_label(
    variant: str = typer.Argument(..., help="Which run to sample from"),
    n: int = typer.Option(50, help="How many questions to hand-label"),
    config: str = typer.Option("configs/baselines.yaml", "--config", "-c"),
) -> None:
    """Write a worksheet for measuring judge reliability (D4).

    The judge's verdict is withheld from the sheet so that labeling is not anchored
    to it.
    """
    from chronomem.config import ExperimentConfig
    from chronomem.evaluation.agreement import write_worksheet
    from chronomem.evaluation.report import load_report

    settings = Settings()
    cfg = ExperimentConfig.from_yaml(config)
    src = settings.results_dir / "raw" / f"{variant}.jsonl"
    if not src.exists():
        console.print(f"[red]No run at {src}[/red]")
        raise typer.Exit(1)

    report = load_report(src, variant=variant)
    instances = lme.load(cfg.dataset_variant, settings.data_dir, limit=cfg.dataset_limit)
    questions = {i.question_id: i.question for i in instances}

    dest = write_worksheet(
        report.results, questions, settings.results_dir / f"labels-{variant}.csv", n=n
    )
    console.print(
        f"[green]✓[/green] {dest}\n\n"
        f"Fill in the [cyan]human[/cyan] column with 1 (correct) or 0 (wrong) for each "
        f"row, judging the [cyan]hypothesis[/cyan] against the [cyan]gold[/cyan] answer.\n"
        f"Then run: [cyan]chronomem eval agreement {variant}[/cyan]"
    )


@eval_app.command("agreement")
def eval_agreement(
    variant: str = typer.Argument(..., help="Which run to score"),
) -> None:
    """Compare the judge's verdicts against hand labels."""
    from chronomem.evaluation.agreement import save_agreement, score_worksheet
    from chronomem.evaluation.report import load_report

    settings = Settings()
    sheet = settings.results_dir / f"labels-{variant}.csv"
    if not sheet.exists():
        console.print(f"[red]No worksheet at {sheet}[/red] — run `eval label {variant}` first.")
        raise typer.Exit(1)

    report = load_report(settings.results_dir / "raw" / f"{variant}.jsonl", variant=variant)
    agreement = score_worksheet(sheet, report.results)
    if agreement.n == 0:
        console.print("[yellow]No rows labeled yet.[/yellow]")
        raise typer.Exit(1)

    console.print(agreement.summary())
    if agreement.disagreements:
        t = Table(title="disagreements")
        t.add_column("question_id", style="cyan")
        t.add_column("judge")
        t.add_column("human")
        for qid, judge, human in agreement.disagreements:
            t.add_row(qid, "correct" if judge else "wrong", "correct" if human else "wrong")
        console.print(t)

    save_agreement(agreement, settings.results_dir / f"agreement-{variant}.json")


ingest_app = typer.Typer(help="Build the memory store from a corpus")
app.add_typer(ingest_app, name="ingest")


@ingest_app.command("run")
def ingest_run(
    config: str = typer.Option("configs/baselines.yaml", "--config", "-c"),
    limit: int | None = typer.Option(None, help="Override dataset_limit from config"),
    sessions: int | None = typer.Option(None, help="Cap unique sessions (for a trial run)"),
    fresh: bool = typer.Option(False, help="Discard the checkpoint and start over"),
    store_name: str = typer.Option("memories", help="Store filename stem"),
) -> None:
    """Extract memories from the corpus. Resumes after a daily-quota stop."""
    from chronomem.config import ExperimentConfig
    from chronomem.embed import Encoder
    from chronomem.ingest import Deduplicator, Extractor, IngestionPipeline, namespaced_sessions
    from chronomem.llm import Limits, QuotaManager, UsageTracker
    from chronomem.llm.client import GeminiClient
    from chronomem.store import NumpyFlatIndex, SQLiteMemoryStore
    from chronomem.temporal import TemporalResolver

    settings = Settings()
    cfg = ExperimentConfig.from_yaml(config)

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
    index = NumpyFlatIndex(settings.store_dir / f"{store_name}-index", dim=cfg.models.embedding_dim)

    extractor = Extractor(client, cfg.models.extractor)
    dedup = Deduplicator(
        client,
        cfg.models.extractor,
        encoder,
        store=store,
        index=index,
        threshold=cfg.ingest.dedupe_similarity_threshold,
    )
    resolver = TemporalResolver(store) if cfg.temporal_resolution else None
    from chronomem.ingest.pipeline import fit_batch_size

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
    )

    with console.status("Loading corpus…"):
        instances = lme.load(
            cfg.dataset_variant, settings.data_dir, limit=limit or cfg.dataset_limit
        )
        all_sessions = namespaced_sessions(instances)
    if sessions:
        all_sessions = all_sessions[:sessions]

    n_batches = -(-len(all_sessions) // per_request)
    console.print(
        f"[bold]ingest[/bold] · {len(all_sessions):,} unique sessions · "
        f"{per_request}/request = ~{n_batches:,} requests · "
        f"extractor={cfg.models.extractor}\n"
    )

    def on_batch(n, total, progress) -> None:
        console.print(
            f"[{n}/{total}] +{progress.memories_written} memories  "
            f"dup={progress.duplicates_dropped} upd={progress.updates_detected}  "
            f"reqs={progress.extraction_requests + progress.adjudication_requests}",
            highlight=False,
        )

    outcome = pipeline.run(all_sessions, resume=not fresh, on_batch=on_batch)
    p = outcome.progress
    usage.save(settings.results_dir / "raw" / "ingest.usage.json")

    t = Table(title="ingestion", show_header=False)
    t.add_column(style="cyan")
    t.add_column(justify="right")
    t.add_row("sessions ingested", f"{len(p.done_sessions):,}")
    t.add_row("memories written", f"{p.memories_written:,}")
    t.add_row("duplicates dropped", f"{p.duplicates_dropped:,}")
    t.add_row("updates detected", f"{p.updates_detected:,}")
    t.add_row("bad session_index", f"{p.bad_session_index:,}")
    t.add_row("superseded", f"{p.superseded:,}")
    t.add_row("extraction requests", f"{p.extraction_requests:,}")
    t.add_row("adjudication requests", f"{p.adjudication_requests:,}")
    if p.memories_written:
        t.add_row("memories / session", f"{p.memories_written / max(1, len(p.done_sessions)):.1f}")
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


@ingest_app.command("coverage")
def ingest_coverage(
    config: str = typer.Option("configs/baselines.yaml", "--config", "-c"),
    n: int = typer.Option(20, help="How many questions to check"),
    show_misses: int = typer.Option(5, help="Print this many failures in full"),
) -> None:
    """Does extraction preserve the answers? Run this before the full ingest.

    Uses the `oracle` variant (evidence sessions only), so it costs a few requests
    and isolates extraction quality from retrieval quality.
    """
    from chronomem.config import ExperimentConfig
    from chronomem.ingest import Extractor
    from chronomem.ingest.coverage import UNMEASURABLE_TYPES, evaluate_coverage
    from chronomem.llm import Limits, QuotaManager, UsageTracker
    from chronomem.llm.client import GeminiClient

    settings = Settings()
    cfg = ExperimentConfig.from_yaml(config)
    quota = QuotaManager(
        state_dir=settings.store_dir / "quota",
        default=Limits(rpm=cfg.quota.rpm, tpm=cfg.quota.tpm, rpd=cfg.quota.rpd),
    )
    quota.load_learned()
    usage = UsageTracker()
    client = GeminiClient(settings.require_api_key(), quota=quota, usage=usage)
    extractor = Extractor(client, cfg.models.extractor)

    with console.status("Loading oracle set…"):
        instances = lme.load("oracle", settings.data_dir, limit=n)

    console.print(f"[bold]coverage[/bold] · {len(instances)} questions · {cfg.models.extractor}\n")

    def progress(case, report) -> None:
        mark = "[green]✓[/green]" if case.covered else "[red]✗[/red]"
        console.print(
            f"{mark} [{report.n}] {case.question_type:<26} {case.n_memories:>2} mems  "
            f"gold={case.gold[:40]!r}",
            highlight=False,
        )

    report = evaluate_coverage(
        instances, lambda sessions: extractor.extract(sessions).memories, on_case=progress
    )

    console.print()
    t = Table(title="answer coverage", show_header=False)
    t.add_column(style="cyan")
    t.add_column(justify="right")
    t.add_row("[bold]coverage (measurable)[/bold]", f"[bold]{report.rate:.1%}[/bold]")
    t.add_row("[dim]coverage (all types)[/dim]", f"[dim]{report.rate_all:.1%}[/dim]")
    t.add_row("measurable questions", str(len(report.measurable)))
    t.add_row("questions", str(report.n))
    t.add_row("memories / question", f"{report.memories_per_session:.1f}")
    console.print(t)

    bt = Table(title="by question type")
    bt.add_column("type", style="cyan")
    bt.add_column("n", justify="right")
    bt.add_column("covered", justify="right")
    for qtype, (total, ok) in report.by_type().items():
        note = " [dim](not string-matchable)[/dim]" if qtype in UNMEASURABLE_TYPES else ""
        bt.add_row(qtype + note, str(total), f"{ok}/{total}")
    console.print(bt)

    for case in report.misses()[:show_misses]:
        console.print(f"\n[red]MISS[/red] {case.question_type} · gold=[yellow]{case.gold}[/yellow]")
        console.print(f"  Q: {case.question[:150]}")
        for m in case.memories[:6]:
            console.print(f"  [dim]- {m[:130]}[/dim]")

    usage.save(settings.results_dir / "raw" / "coverage.usage.json")

    # Persisted so the failures can be studied without re-spending quota — the
    # analysis loop on a metric like this is many passes over the same output.
    import json as _json
    from dataclasses import asdict as _asdict

    dest = settings.results_dir / "raw" / "coverage.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        _json.dumps(
            {
                "rate_measurable": report.rate,
                "rate_all": report.rate_all,
                "memories_per_question": report.memories_per_session,
                "cases": [_asdict(c) for c in report.cases],
            },
            indent=2,
        )
    )
    console.print(f"\n[green]✓[/green] {dest}")


@app.command("resolve")
def resolve(
    config: str = typer.Option("configs/baselines.yaml", "--config", "-c"),
    store_name: str = typer.Option("memories"),
) -> None:
    """Re-run temporal resolution over the whole store.

    Free — pure SQL, no API calls. This is what makes the arity list in
    `SINGLE_VALUED_PREDICATES` cheap to correct: getting it wrong costs a
    re-resolution, not a 2,880-request re-ingest.
    """
    from chronomem.store import SQLiteMemoryStore
    from chronomem.temporal import TemporalResolver

    settings = Settings()
    store = SQLiteMemoryStore(settings.store_dir / f"{store_name}.db")
    store.initialize()
    stats = TemporalResolver(store).resolve_everything()

    t = Table(title="temporal resolution", show_header=False)
    t.add_column(style="cyan")
    t.add_column(justify="right")
    t.add_row("keys examined", f"{stats.keys_examined:,}")
    t.add_row("keys resolved", f"{stats.keys_resolved:,}")
    t.add_row("superseded", f"{stats.superseded:,}")
    t.add_row("restatements", f"{stats.restatements:,}")
    t.add_row("promoted back", f"{stats.promoted:,}")
    t.add_row("undated (unresolvable)", f"{stats.skipped_undated:,}")
    t.add_row("active memories", f"{store.count(status='active'):,}")
    console.print(t)
    store.close()


@eval_app.command("compare")
def eval_compare(
    variant_a: str = typer.Argument(..., help="Baseline variant"),
    variant_b: str = typer.Argument(..., help="Variant under test"),
) -> None:
    """Paired McNemar test between two runs on the same questions.

    Comparing headline accuracies cannot separate a real improvement from
    re-running the same configuration. This can: it ignores every question the two
    variants agree on and tests only the disagreements.
    """
    from chronomem.evaluation.compare import compare as paired
    from chronomem.evaluation.report import load_report

    settings = Settings()
    raw = settings.results_dir / "raw"
    reports = {}
    for v in (variant_a, variant_b):
        path = raw / f"{v}.jsonl"
        if not path.exists():
            console.print(f"[red]No run at {path}[/red]")
            raise typer.Exit(1)
        reports[v] = load_report(path, variant=v)

    c = paired(reports[variant_a], reports[variant_b])

    t = Table(title=f"{variant_a} vs {variant_b}", show_header=False)
    t.add_column(style="cyan")
    t.add_column(justify="right")
    t.add_row("paired questions", str(c.n_paired))
    t.add_row("both right", str(c.both_right))
    t.add_row("both wrong", str(c.both_wrong))
    t.add_row(f"[green]{variant_b} wins[/green]", str(c.b_wins))
    t.add_row(f"[red]{variant_b} losses[/red]", str(c.b_losses))
    t.add_row("headline delta", f"{c.accuracy_delta:+.1%}")
    t.add_row("[bold]p-value[/bold]", f"[bold]{c.p_value:.4f}[/bold]")
    console.print(t)
    style = "green" if c.significant else "yellow"
    console.print(f"\n[{style}]{c.verdict()}[/{style}]")


@eval_app.command("variability")
def eval_variability(
    variant: str = typer.Argument(..., help="Variant with repeat runs on disk"),
) -> None:
    """Spread across repeats of one unchanged configuration.

    Reads `<variant>.jsonl` plus any `<variant>.rep*.jsonl`. This is the yardstick
    every reported difference has to clear.
    """
    from chronomem.evaluation.compare import Variability
    from chronomem.evaluation.report import load_report

    settings = Settings()
    raw = settings.results_dir / "raw"
    paths = sorted(raw.glob(f"{variant}.rep*.jsonl"))
    base = raw / f"{variant}.jsonl"
    if base.exists():
        paths = [base, *paths]
    if len(paths) < 2:
        console.print(
            f"[yellow]Need at least two runs.[/yellow] Found {len(paths)}. "
            f"Re-run with --fresh and copy to {variant}.rep1.jsonl, etc."
        )
        raise typer.Exit(1)

    accuracies = [load_report(p).accuracy for p in paths]
    v = Variability(variant, accuracies)
    console.print(v.summary())


@ingest_app.command("fidelity")
def ingest_fidelity(
    config: str = typer.Option("configs/baselines.yaml", "--config", "-c"),
    sessions: int = typer.Option(30, help="How many sessions to extract and score"),
    show_misses: int = typer.Option(5, help="Missed values to print per facet"),
    holdout: bool = typer.Option(
        True, help="Score sessions from the held-out split, never the dev questions"
    ),
    batch: int | None = typer.Option(None, help="Override sessions per request"),
) -> None:
    """What fraction of the user's own specifics survive extraction?

    Needs no gold answers, so it cannot be fitted to the evaluation set — the
    reference is the source text. Run it before spending an ingest on a prompt
    change.
    """
    from chronomem.config import ExperimentConfig
    from chronomem.ingest import Extractor
    from chronomem.ingest.fidelity import score_sessions
    from chronomem.llm import Limits, QuotaManager, UsageTracker
    from chronomem.llm.client import GeminiClient
    from chronomem.store import Memory

    settings = Settings()
    cfg = ExperimentConfig.from_yaml(config)
    quota = QuotaManager(
        state_dir=settings.store_dir / "quota",
        default=Limits(rpm=cfg.quota.rpm, tpm=cfg.quota.tpm, rpd=cfg.quota.rpd),
    )
    quota.load_learned()
    usage = UsageTracker()
    client = GeminiClient(settings.require_api_key(), quota=quota, usage=usage)
    extractor = Extractor(client, cfg.models.extractor)

    with console.status("Loading corpus…"):
        every = lme.load(cfg.dataset_variant, settings.data_dir)
        dev, test = lme.split_dev_test(every)
        pool = test if holdout else dev
        picked: list = []
        for inst in pool:
            for sess in inst.sessions:
                picked.append(sess)
                if len(picked) >= sessions:
                    break
            if len(picked) >= sessions:
                break

    batch = batch or cfg.ingest.sessions_per_request
    console.print(
        f"[bold]fidelity[/bold] · {len(picked)} sessions from the "
        f"{'held-out' if holdout else 'dev'} split · {batch}/request · "
        f"{cfg.models.extractor}\n"
    )

    # Cache the extraction so a change to the *metric* can be re-scored for free.
    # The same sessions were re-extracted four times while the denominator was being
    # corrected; each pass cost requests and returned identical memories.
    import hashlib
    import json as _json

    from chronomem.ingest.extract import _PROMPT, EXTRACT_SYSTEM

    fingerprint = hashlib.sha1(
        (EXTRACT_SYSTEM + _PROMPT + cfg.models.extractor + str(batch)).encode()
    ).hexdigest()[:12]
    cache_path = settings.store_dir / "fidelity-cache" / f"{fingerprint}.json"
    cached = _json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if cached:
        console.print(
            f"[dim]reusing cached extraction {fingerprint} ({len(cached)} sessions)[/dim]"
        )

    pairs = []
    for i in range(0, len(picked), batch):
        chunk = picked[i : i + batch]
        if all(s.session_id in cached for s in chunk):
            for s in chunk:
                mems = [
                    Memory(**{**d, "event_time": None, "valid_from": None, "valid_to": None})
                    for d in cached[s.session_id]
                ]
                pairs.append((s, mems))
            continue
        outcome = extractor.extract(chunk)
        by_session: dict[str, list] = {s.session_id: [] for s in chunk}
        for m in outcome.memories:
            if m.source_session_id in by_session:
                by_session[m.source_session_id].append(m)
        pairs.extend((s, by_session[s.session_id]) for s in chunk)
        for s in chunk:
            cached[s.session_id] = [
                {
                    "id": m.id,
                    "user_id": m.user_id,
                    "type": m.type,
                    "content": m.content,
                    "token_count": m.token_count,
                    "subject": m.subject,
                    "predicate": m.predicate,
                    "object": m.object,
                }
                for m in by_session[s.session_id]
            ]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(_json.dumps(cached))
        console.print(f"  batch {i // batch + 1}: +{len(outcome.memories)} memories")

    report = score_sessions(pairs)

    t = Table(title="extraction fidelity", show_header=True)
    t.add_column("facet", style="cyan")
    t.add_column("stated", justify="right")
    t.add_column("retained", justify="right")
    t.add_column("recall", justify="right")
    for facet, score in sorted(report.per_facet.items(), key=lambda kv: kv[1].recall):
        style = "red" if score.recall < 0.4 else ("yellow" if score.recall < 0.7 else "green")
        t.add_row(
            facet, str(score.stated), str(score.retained), f"[{style}]{score.recall:.1%}[/{style}]"
        )
    console.print(t)
    console.print(
        f"[bold]overall {report.overall:.1%}[/bold] · "
        f"{report.memories_per_session:.1f} memories/session · {report.memories} memories"
    )

    for facet, misses in report.missed_examples.items():
        console.print(
            f"\n[red]dropped {facet}[/red]: " + ", ".join(repr(m) for m in misses[:show_misses])
        )

    usage.save(settings.results_dir / "raw" / "fidelity.usage.json")


influence_app = typer.Typer(help="P6: memory utility measurement and packing")
app.add_typer(influence_app, name="influence")


@influence_app.command("cost")
def influence_cost(
    questions: int = typer.Option(50),
    top_k: int = typer.Option(20),
    leave_one_in: bool = typer.Option(True),
) -> None:
    """What a full influence measurement costs, before committing to it."""
    from chronomem.influence import requests_needed

    answers = requests_needed(questions, top_k, leave_one_in)
    t = Table(show_header=False)
    t.add_column(style="cyan")
    t.add_column(justify="right")
    t.add_row("answer calls", f"{answers:,}")
    t.add_row("judge calls", f"{answers:,}")
    t.add_row("total requests", f"{answers * 2:,}")
    t.add_row("days at 500/day", f"{answers * 2 / 500:.1f}")
    console.print(t)
    console.print(
        "[dim]This is why the measurement is offline and its output is a training "
        "set: a packer running it per query would spend far more inference than the "
        "tokens it saves.[/dim]"
    )


@influence_app.command("fit")
def influence_fit(
    labels: str = typer.Option("results/raw/influence.jsonl"),
    alpha: float = typer.Option(1.0, help="Ridge penalty"),
) -> None:
    """Fit the utility predictor on measured labels, held out by question."""
    import numpy as np

    from chronomem.influence import InfluenceDataset, fit_grouped

    path = Path(labels)
    if not path.exists():
        console.print(f"[yellow]No labels at {path}.[/yellow] Run the measurement first.")
        raise typer.Exit(1)

    data = InfluenceDataset.load(path)
    features_path = path.with_suffix(".features.npy")
    if not features_path.exists():
        console.print(f"[yellow]No features at {features_path}.[/yellow]")
        raise typer.Exit(1)

    x = np.load(features_path)
    y = np.array([r.utility for r in data.rows])
    groups = [r.question_id for r in data.rows]

    model, report = fit_grouped(x, y, groups, alpha=alpha)
    model.save(Settings().store_dir / "utility-predictor.json")

    t = Table(title="utility predictor", show_header=False)
    t.add_column(style="cyan")
    t.add_column(justify="right")
    t.add_row("rows", f"{report.n_train:,}")
    t.add_row("held-out rows", f"{report.n_test:,}")
    t.add_row("RMSE", f"{report.rmse:.3f}")
    t.add_row("baseline (predict the mean)", f"{report.baseline_rmse:.3f}")
    t.add_row("beats baseline", "[green]yes[/green]" if report.beats_baseline else "[red]no[/red]")
    t.add_row("rank corr. with relevance", f"{report.spearman_vs_relevance:.2f}")
    console.print(t)

    if report.spearman_vs_relevance > 0.9:
        console.print(
            "\n[yellow]Predicted utility tracks the retrieval score almost exactly.[/yellow] "
            "The experiment's question — can downstream utility beat relevance for "
            "selection — is answered no at this scale."
        )

    ct = Table(title="coefficients")
    ct.add_column("feature", style="cyan")
    ct.add_column("weight", justify="right")
    for name, w in sorted(report.coefficients.items(), key=lambda kv: -abs(kv[1])):
        ct.add_row(name, f"{w:+.3f}")
    console.print(ct)


if __name__ == "__main__":
    app()
