"""LLTM command line.

P0 commands are the ones that need no API key: download the benchmark, measure it,
and plan the ingestion budget against the free-tier quota.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from llm_long_term_memory.config import Settings
from llm_long_term_memory.evaluation.datasets import longmemeval as lme
from llm_long_term_memory.locking import AlreadyRunning, exclusive

app = typer.Typer(add_completion=False, help="LLTM — long-term memory for LLM agents")
data_app = typer.Typer(help="Benchmark data: download, inspect, plan")
app.add_typer(data_app, name="data")

console = Console()


@contextmanager
def _cli_lock(path: Path, *, what: str):
    try:
        with exclusive(path, what=what):
            yield
    except AlreadyRunning as exc:
        console.print(f"[red]STOP[/red]: {exc}")
        raise typer.Exit(code=2) from None


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
        f"\n[dim]Token counts use the corpus-measured estimate of "
        f"{lme.CHARS_PER_TOKEN:g} chars/token. Exact counts depend on the model's "
        "provider tokenizer.[/dim]"
    )


@data_app.command("plan")
def data_plan(
    variant: str = typer.Option("s"),
    data_dir: str = typer.Option("data"),
    rpd: int = typer.Option(500, help="Requests/day allowed by your quota tier"),
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


def _default_store_name(variant: str) -> str:
    """Keep new extraction variants physically separate from frozen v1 data."""
    return "two-stage" if variant.startswith("two_stage") else "memories"


def _build(
    variant: str,
    cfg_path: str,
    store_name: str | None = None,
    top_k: int | None = None,
    rerank: bool | None = None,
):
    """Wire up client, judge, and runner for one variant.

    `top_k` and `rerank` override the config so that an accuracy-vs-context sweep is
    a loop over flags rather than six near-identical YAML files. Anything reported
    as a table row should still come from a config, not from a flag.
    """
    from llm_long_term_memory.config import ExperimentConfig
    from llm_long_term_memory.embed import Encoder
    from llm_long_term_memory.evaluation.judge import Judge
    from llm_long_term_memory.evaluation.runners.full_context import FullContextRunner
    from llm_long_term_memory.evaluation.runners.memory import MemoryRunner
    from llm_long_term_memory.evaluation.runners.naive_rag import NaiveRAGRunner
    from llm_long_term_memory.llm import Limits, QuotaManager, UsageTracker
    from llm_long_term_memory.llm.client import GeminiClient

    settings = Settings()
    cfg = ExperimentConfig.from_yaml(cfg_path)
    if top_k is not None:
        cfg.retrieval.top_k = top_k
    if rerank is not None:
        cfg.retrieval.rerank.enabled = rerank

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
    elif variant in (
        "chronomem",
        "chronomem_no_temporal",
        "two_stage",
        "two_stage_no_temporal",
        "two_stage_hydrated",
        "two_stage_hydrated_no_temporal",
        # A3. Same store and same pipeline; the only difference is
        # `retrieval.rerank.enabled` in the config, so the row is attributable.
        "two_stage_hydrated_rerank",
        # Memory first, raw source only when the answerer says it needs it.
        "two_stage_fallback",
        # P6. Same store, same retrieval, same fallback; the memories are grouped
        # into whole sessions in event order rather than handed over as a ranking.
        # The only difference from `two_stage_fallback` is the context's shape.
        "two_stage_coherent",
        # Validation-only ceiling: session ids come from benchmark gold labels.
        # It can explain a null but is never a product variant.
        "two_stage_coherent_oracle",
    ):
        from llm_long_term_memory.retrieve import SessionBudget
        from llm_long_term_memory.store import NumpyFlatIndex, SQLiteMemoryStore

        utility_model = None
        if cfg.pack.enabled and cfg.pack.utility_model_path:
            from llm_long_term_memory.influence import UtilityPredictor

            utility_model = UtilityPredictor.load(cfg.pack.utility_model_path)

        stem = store_name or _default_store_name(variant)
        store = SQLiteMemoryStore(settings.store_dir / f"{stem}.db")
        store.initialize()
        index = NumpyFlatIndex(settings.store_dir / f"{stem}-index", dim=cfg.models.embedding_dim)
        if not len(index):
            raise typer.BadParameter(
                f"the {stem!r} memory store is empty — run `lltm ingest run "
                f"--store-name {stem}` first"
            )
        reranker = None
        if cfg.retrieval.rerank.enabled:
            from llm_long_term_memory.retrieve import CrossEncoderReranker

            reranker = CrossEncoderReranker(
                model_name=cfg.retrieval.rerank.model,
                candidates=cfg.retrieval.rerank.candidates,
                batch_size=cfg.retrieval.rerank.batch_size,
            )
        runner = MemoryRunner(
            client,
            model=cfg.models.answerer,
            encoder=Encoder(),
            store=store,
            index=index,
            top_k=cfg.retrieval.top_k,
            temporal=not variant.endswith("_no_temporal"),
            retrieval_weights=cfg.retrieval.weights.model_dump(),
            candidate_limit=cfg.retrieval.candidate_limit,
            recency_halflife_days=cfg.retrieval.recency_halflife_days,
            decay_enabled=cfg.decay.enabled,
            decay_halflife_days=cfg.decay.halflife_days,
            reinforcement=cfg.decay.reinforcement,
            evidence_hydration="_hydrated" in variant,
            hydration_neighbouring_sentences=cfg.hydration.neighbouring_sentences,
            hydration_max_tokens=cfg.hydration.max_tokens,
            token_budget=cfg.pack.token_budget if cfg.pack.enabled else 0,
            utility_model=utility_model,
            type_floors=cfg.pack.type_floors if cfg.pack.enabled else None,
            reranker=reranker,
            raw_fallback=cfg.fallback.enabled,
            raw_fallback_max_turns=cfg.fallback.max_turns,
            raw_fallback_max_chars=cfg.fallback.max_chars,
            session_budget=SessionBudget(
                max_sessions=cfg.context.max_sessions,
                window_radius=cfg.context.window_radius,
                max_total_memories=cfg.context.max_total_memories,
                aggregate=cfg.context.aggregate,
                session_order=cfg.context.session_order,
                include_superseded=cfg.context.include_superseded,
            )
            if variant in {"two_stage_coherent", "two_stage_coherent_oracle"}
            else None,
            oracle_session_context=variant == "two_stage_coherent_oracle",
        )
        runner.name = variant
    else:
        raise typer.BadParameter(f"unknown variant {variant!r}")
    return cfg, settings, runner, judge, usage


@eval_app.command("run")
def eval_run(
    variant: str = typer.Argument(
        ...,
        help=(
            "full_context | naive_rag | two_stage | two_stage_no_temporal | "
            "two_stage_hydrated | two_stage_hydrated_no_temporal | two_stage_coherent | "
            "two_stage_coherent_oracle. "
            "The chronomem* names belong to the frozen v1 run and are kept so its "
            "rows are not overwritten by a re-measurement of a different pipeline."
        ),
    ),
    config: str = typer.Option("configs/baselines.yaml", "--config", "-c"),
    limit: int | None = typer.Option(None, help="Override dataset_limit from config"),
    questions: str | None = typer.Option(
        None,
        "--questions",
        help=(
            "Manifest of question ids (results/manifests/*.json, or a newline-delimited "
            "list). Evaluates exactly these. Use this, not --limit, for anything whose "
            "result gets reported: --limit draws a stratified sample scattered across "
            "the split, so --limit 31 is NOT the first 31 questions in dataset order."
        ),
    ),
    store_name: str | None = typer.Option(
        None, help="Store filename stem; two_stage defaults to an isolated two-stage store"
    ),
    fresh: bool = typer.Option(False, help="Ignore existing results and start over"),
    top_k: int | None = typer.Option(None, "--top-k", help="Override retrieval.top_k"),
    rerank: bool | None = typer.Option(
        None, "--rerank/--no-rerank", help="Override retrieval.rerank.enabled"
    ),
    label: str | None = typer.Option(
        None,
        help=(
            "Write to <variant>.<label>.jsonl instead of <variant>.jsonl. Sweep arms "
            "need this: they share a variant but are not the same run, and a labelled "
            "file is excluded from the default results table."
        ),
    ),
) -> None:
    """Evaluate one variant on LongMemEval. Resumes automatically if interrupted."""
    from llm_long_term_memory.evaluation.harness import run_eval
    from llm_long_term_memory.evaluation.report import render_summary

    cfg, settings, runner, judge, usage = _build(variant, config, store_name, top_k, rerank)
    n = limit if limit is not None else cfg.dataset_limit

    with console.status("Loading dataset…"):
        if questions:
            from llm_long_term_memory.evaluation.manifest import load_manifest

            # Load the full split and filter, rather than loading a stratified
            # sample of size len(manifest): stratification would return a different
            # set of questions entirely. See the --questions help text.
            manifest = load_manifest(questions)
            every = lme.load(manifest.variant, settings.data_dir)
            by_id = {inst.question_id: inst for inst in every}
            unknown = [qid for qid in manifest.question_ids if qid not in by_id]
            if unknown:
                raise typer.BadParameter(
                    f"{len(unknown)} question id(s) not in the {manifest.variant!r} "
                    f"split, first: {unknown[0]!r}"
                )
            instances = [by_id[qid] for qid in manifest.question_ids]
        else:
            instances = lme.load(cfg.dataset_variant, settings.data_dir, limit=n)

    stem = f"{variant}.{label}" if label else variant
    out = settings.results_dir / "raw" / f"{stem}.jsonl"
    runner.name = stem
    console.print(
        f"[bold]{stem}[/bold] · {len(instances)} questions · "
        f"answerer={cfg.models.answerer} judge={cfg.models.judge} · "
        f"top_k={cfg.retrieval.top_k} rerank={cfg.retrieval.rerank.enabled}\n"
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


@eval_app.command("budget-sweep")
def eval_budget_sweep(
    variant: str = typer.Option("two_stage", help="Memory-backed evaluation variant"),
    config: str = typer.Option("configs/baselines.yaml", "--config", "-c"),
    budgets: str = typer.Option("1000,2000,4000,8000", help="Comma-separated token budgets"),
    store_name: str | None = typer.Option(None, help="Memory store filename stem"),
    utility_model: str | None = typer.Option(None, help="Override utility-predictor JSON path"),
    limit: int | None = typer.Option(None, help="Override dataset_limit from config"),
    fresh: bool = typer.Option(False, help="Discard existing checkpoints for every budget"),
) -> None:
    """Evaluate relevance or utility packing over token budgets and draw a Pareto curve."""
    import json
    from dataclasses import asdict

    from llm_long_term_memory.evaluation.budget import (
        BudgetPoint,
        pareto_frontier,
        render_pareto_svg,
        sweep_artifact_stem,
    )
    from llm_long_term_memory.evaluation.harness import run_eval
    from llm_long_term_memory.evaluation.report import render_summary
    from llm_long_term_memory.evaluation.runners.memory import MemoryRunner
    from llm_long_term_memory.influence import UtilityPredictor

    try:
        parsed_budgets = sorted(
            {int(value.strip()) for value in budgets.split(",") if value.strip()}
        )
    except ValueError as exc:
        raise typer.BadParameter("budgets must be comma-separated positive integers") from exc
    if not parsed_budgets or any(budget <= 0 for budget in parsed_budgets):
        raise typer.BadParameter("budgets must contain at least one positive integer")

    points: list[BudgetPoint] = []
    settings = None
    selector = "relevance"
    for budget in parsed_budgets:
        cfg, settings, runner, judge, usage = _build(variant, config, store_name)
        if not isinstance(runner, MemoryRunner):
            raise typer.BadParameter("budget sweep requires a memory-backed variant")

        model_path = utility_model or cfg.pack.utility_model_path
        if model_path:
            runner.utility_model = UtilityPredictor.load(model_path)
            selector = "utility"
        runner.token_budget = budget
        runner.type_floors = cfg.pack.type_floors
        runner.name = f"{variant}_{selector}_pack_{budget}"

        n = limit if limit is not None else cfg.dataset_limit
        with console.status(f"Loading dataset for {budget:,} tokens…"):
            instances = lme.load(cfg.dataset_variant, settings.data_dir, limit=n)
        path = settings.results_dir / "raw" / f"{runner.name}.jsonl"
        report = run_eval(runner, judge, instances, path, usage=usage, resume=not fresh)
        points.append(
            BudgetPoint(
                budget=budget,
                accuracy=report.accuracy,
                median_context_tokens=report.median_context_tokens,
                p95_latency_ms=report.p95_latency_ms,
                n=report.n,
                completed=report.completed,
            )
        )
        console.print(render_summary(report))
        if not report.completed:
            console.print(f"[yellow]Paused at {budget:,} tokens; rerun to resume.[/yellow]")
            break

    assert settings is not None
    payload = {
        "variant": variant,
        "selector": selector,
        "points": [asdict(point) for point in points],
        "pareto_budgets": [point.budget for point in pareto_frontier(points)],
    }
    stem = sweep_artifact_stem(variant, selector)
    json_path = settings.results_dir / f"{stem}.json"
    svg_path = settings.results_dir / f"{stem}.svg"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    svg_path.write_text(
        render_pareto_svg(points, f"{variant}: {selector} packing"), encoding="utf-8"
    )

    table = Table(title=f"P6 budget sweep — {selector} packing")
    table.add_column("budget", justify="right", style="cyan")
    table.add_column("n", justify="right")
    table.add_column("accuracy", justify="right")
    table.add_column("median context", justify="right")
    table.add_column("p95 latency", justify="right")
    for point in points:
        table.add_row(
            f"{point.budget:,}",
            str(point.n),
            f"{point.accuracy:.1%}",
            f"{point.median_context_tokens:,.0f}",
            f"{point.p95_latency_ms / 1000:.1f}s",
        )
    console.print(table)
    console.print(f"[green]✓[/green] {json_path}\n[green]✓[/green] {svg_path}")


@eval_app.command("report")
def eval_report(
    variants: list[str] = _ALL_VARIANTS,
    out: str = typer.Option("results/table.md", help="Where to write the markdown table"),
) -> None:
    """Regenerate the results table from run artifacts."""
    from llm_long_term_memory.evaluation.report import (
        default_report_variants,
        load_report,
        render_table,
    )

    settings = Settings()
    raw = settings.results_dir / "raw"
    if not raw.exists():
        console.print("[yellow]No runs found.[/yellow] Try `lltm eval run full_context`.")
        raise typer.Exit(1)

    order = variants or default_report_variants(raw)
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
    dest.write_text(table + "\n", encoding="utf-8")
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
    from llm_long_term_memory.config import ExperimentConfig
    from llm_long_term_memory.evaluation.agreement import write_worksheet
    from llm_long_term_memory.evaluation.report import load_report

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
        f"Then run: [cyan]lltm eval agreement {variant}[/cyan]"
    )


@eval_app.command("agreement")
def eval_agreement(
    variant: str = typer.Argument(..., help="Which run to score"),
) -> None:
    """Compare the judge's verdicts against hand labels."""
    from llm_long_term_memory.evaluation.agreement import save_agreement, score_worksheet
    from llm_long_term_memory.evaluation.report import load_report

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


@eval_app.command("failure-audit")
def eval_failure_audit(
    variant: str = typer.Argument(..., help="Which evaluated variant to audit"),
    config: str = typer.Option("configs/baselines.yaml", "--config", "-c"),
    out: str | None = typer.Option(None, help="CSV destination"),
) -> None:
    """Write one worksheet row for every judged-wrong answer."""
    from llm_long_term_memory.config import ExperimentConfig
    from llm_long_term_memory.evaluation.failures import FAILURE_CODES, write_failure_worksheet
    from llm_long_term_memory.evaluation.report import load_report

    settings = Settings()
    source = settings.results_dir / "raw" / f"{variant}.jsonl"
    if not source.exists():
        console.print(f"[red]No run at {source}[/red]")
        raise typer.Exit(1)
    cfg = ExperimentConfig.from_yaml(config)
    instances = lme.load(cfg.dataset_variant, settings.data_dir, limit=cfg.dataset_limit)
    destination = Path(out) if out else settings.results_dir / f"failure-audit-{variant}.csv"
    report = load_report(source, variant=variant)
    questions = {item.question_id: item.question for item in instances}
    write_failure_worksheet(report.results, questions, destination)
    console.print(f"[green]✓[/green] {destination}")
    console.print("Label each row with exactly one primary failure:")
    console.print(", ".join(f"{code}={description}" for code, description in FAILURE_CODES.items()))
    console.print("For E1, add a detail such as number, date, duration, or entity.")
    console.print("Use event, relation, or negation when applicable.")


@eval_app.command("failure-report")
def eval_failure_report(
    worksheet: str = typer.Argument(..., help="Completed failure-audit CSV"),
) -> None:
    """Summarize a completed single-cause failure taxonomy worksheet."""
    from llm_long_term_memory.evaluation.failures import FAILURE_CODES, summarize_failure_worksheet

    report = summarize_failure_worksheet(worksheet)
    table = Table(title="failure decomposition")
    table.add_column("cause", style="cyan")
    table.add_column("count", justify="right")
    for code, description in FAILURE_CODES.items():
        table.add_row(f"{code} — {description}", str(report.by_code.get(code, 0)))
    console.print(table)
    console.print(f"classified: {report.classified}/{report.total}")
    if report.extraction_details:
        details = ", ".join(f"{key}={value}" for key, value in report.extraction_details.items())
        console.print("E1 details: " + details)
    if report.unclassified_ids:
        console.print("[yellow]Unclassified:[/yellow] " + ", ".join(report.unclassified_ids))


ingest_app = typer.Typer(help="Build the memory store from a corpus")
app.add_typer(ingest_app, name="ingest")


@ingest_app.command("run")
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
    with _cli_lock(settings.store_dir / f"{store_name}.db", what="ingest"):
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
        )

        with console.status("Loading corpus…"):
            if questions:
                # Load the whole split and filter, exactly as `eval run --questions`
                # does. A stratified sample of the manifest's size would return a
                # different set of questions, and for a held-out split the haystacks
                # ingested have to be the ones its questions are asked about.
                from llm_long_term_memory.evaluation.manifest import load_manifest

                manifest = load_manifest(questions)
                by_id = {i.question_id: i for i in lme.load(manifest.variant, settings.data_dir)}
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
    from llm_long_term_memory.config import ExperimentConfig
    from llm_long_term_memory.ingest import Extractor
    from llm_long_term_memory.ingest.coverage import UNMEASURABLE_TYPES, evaluate_coverage
    from llm_long_term_memory.llm import Limits, QuotaManager, UsageTracker
    from llm_long_term_memory.llm.client import GeminiClient

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
    t = Table(title="literal coverage", show_header=False)
    t.add_column(style="cyan")
    t.add_column(justify="right")
    t.add_row("[bold]structured literal (measurable)[/bold]", f"[bold]{report.rate:.1%}[/bold]")
    t.add_row("source literal (measurable)", f"{report.source_literal_rate:.1%}")
    t.add_row("[dim]structured literal (all types)[/dim]", f"[dim]{report.rate_all:.1%}[/dim]")
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
                "source_literal_rate_measurable": report.source_literal_rate,
                "memories_per_question": report.memories_per_session,
                "cases": [_asdict(c) for c in report.cases],
            },
            indent=2,
        ),
        encoding="utf-8",
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
    from llm_long_term_memory.store import SQLiteMemoryStore
    from llm_long_term_memory.temporal import TemporalResolver

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


@app.command("mcp")
def mcp_serve(
    transport: str = typer.Option("stdio", help="stdio | http"),
    # Matching MemoryService's defaults: the product config and the clean store.
    config: str = typer.Option("configs/fallback.yaml", "--config", "-c"),
    store_name: str = typer.Option("two-stage-p10", help="Memory store filename stem"),
) -> None:
    """Serve the memory over MCP, for Claude Desktop, Cursor, or any MCP client.

    The same `MemoryService` the REST API uses, so the tools cannot drift from the
    behaviour that was measured.
    """
    from llm_long_term_memory.api.service import MemoryService
    from llm_long_term_memory.mcp_server import run

    # stdio is the protocol channel: anything written to stdout that is not a
    # protocol message corrupts the stream, so no banner is printed there.
    if transport != "stdio":
        console.print(f"[bold]mcp[/bold] · {transport} · store={store_name}")
    run(transport=transport, service=MemoryService(config_path=config, store_name=store_name))


@eval_app.command("freeze")
def eval_freeze(
    name: str = typer.Argument(..., help="Manifest name, e.g. dev50 or test100"),
    n: int = typer.Option(50, help="How many questions to draw"),
    seed: int = typer.Option(0, help="Stratification seed"),
    variant: str = typer.Option("s", help="LongMemEval split"),
    exclude: str | None = typer.Option(
        None, help="Manifest whose ids must not appear (for building a held-out set)"
    ),
    note: str = typer.Option("", help="Why this set exists; stored in the manifest"),
    force: bool = typer.Option(False, help="Overwrite an existing manifest"),
) -> None:
    """Freeze a stratified question set into results/manifests/<name>.json.

    Every reported run should name a manifest rather than a size. A size is not an
    experiment identity: it says how many questions, not which ones, and the answer
    to "which ones" has already changed once under a partial ingest.
    """
    from llm_long_term_memory.evaluation.datasets.longmemeval import stratify
    from llm_long_term_memory.evaluation.manifest import Manifest, load_manifest

    settings = Settings()
    dest = settings.results_dir / "manifests" / f"{name}.json"
    if dest.exists() and not force:
        existing = load_manifest(dest)
        raise typer.BadParameter(
            f"{dest} already exists with {len(existing)} questions. Refusing to "
            f"regenerate: a frozen set that moves is not frozen. Pass --force only "
            f"if no published result depends on it."
        )

    every = lme.load(variant, settings.data_dir)
    if exclude:
        banned = set(load_manifest(exclude).question_ids)
        every = [inst for inst in every if inst.question_id not in banned]
        console.print(f"[dim]excluded {len(banned)} ids; {len(every)} remain[/dim]")

    chosen = stratify(every, n, seed=seed)
    manifest = Manifest(
        name=name,
        variant=variant,
        seed=seed,
        question_ids=tuple(inst.question_id for inst in chosen),
        note=note,
    )
    manifest.save(dest)

    by_type: dict[str, int] = {}
    for inst in chosen:
        by_type[inst.question_type] = by_type.get(inst.question_type, 0) + 1
    t = Table(title=f"frozen: {dest}", show_header=False)
    t.add_column(style="cyan")
    t.add_column(justify="right")
    for key in sorted(by_type):
        t.add_row(key, str(by_type[key]))
    t.add_row("[bold]total", f"[bold]{len(manifest)}")
    console.print(t)


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
    from llm_long_term_memory.evaluation.compare import compare as paired
    from llm_long_term_memory.evaluation.report import load_report

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
    every reported difference has to clear; create repeats with `eval repeat`.
    """
    from llm_long_term_memory.evaluation.compare import Variability
    from llm_long_term_memory.evaluation.report import load_report

    settings = Settings()
    raw = settings.results_dir / "raw"
    paths = sorted(raw.glob(f"{variant}.rep*.jsonl"))
    base = raw / f"{variant}.jsonl"
    if base.exists():
        paths = [base, *paths]
    if len(paths) < 2:
        console.print(
            f"[yellow]Need at least two runs.[/yellow] Found {len(paths)}. "
            f"Run `lltm eval repeat {variant} --runs 3`."
        )
        raise typer.Exit(1)

    accuracies = [load_report(p).accuracy for p in paths]
    v = Variability(variant, accuracies)
    console.print(v.summary())


@eval_app.command("repeat")
def eval_repeat(
    variant: str = typer.Argument(..., help="Variant to run repeatedly"),
    runs: int = typer.Option(3, min=2, max=5, help="Independent repetitions"),
    config: str = typer.Option("configs/baselines.yaml", "--config", "-c"),
    limit: int | None = typer.Option(None, help="Override dataset_limit from config"),
    store_name: str | None = typer.Option(None, help="Memory store filename stem"),
    fresh: bool = typer.Option(False, help="Discard each repeat checkpoint"),
) -> None:
    """Run 2-5 independently checkpointed repetitions of one configuration."""
    from llm_long_term_memory.evaluation.compare import Variability
    from llm_long_term_memory.evaluation.harness import run_eval

    accuracies = []
    for number in range(1, runs + 1):
        cfg, settings, runner, judge, usage = _build(variant, config, store_name)
        runner.name = f"{variant}.rep{number}"
        n = limit if limit is not None else cfg.dataset_limit
        with console.status(f"Loading dataset for repeat {number}/{runs}…"):
            instances = lme.load(cfg.dataset_variant, settings.data_dir, limit=n)
        path = settings.results_dir / "raw" / f"{runner.name}.jsonl"
        report = run_eval(runner, judge, instances, path, usage=usage, resume=not fresh)
        accuracies.append(report.accuracy)
        console.print(f"repeat {number}/{runs}: {report.accuracy:.1%} ({report.n} questions)")
        if not report.completed:
            console.print(f"[yellow]Paused repeat {number}; rerun to resume.[/yellow]")
            return
    console.print(Variability(variant, accuracies).summary())


@ingest_app.command("fidelity")
def ingest_fidelity(
    config: str = typer.Option("configs/baselines.yaml", "--config", "-c"),
    sessions: int = typer.Option(30, help="How many sessions to extract and score"),
    show_misses: int = typer.Option(5, help="Missed values to print per facet"),
    holdout: bool = typer.Option(
        True, help="Score sessions from the held-out split, never the dev questions"
    ),
    batch: int | None = typer.Option(None, help="Override sessions per request"),
    two_stage: bool | None = typer.Option(None, help="Override the two-stage flag"),
    min_score: float | None = typer.Option(
        None,
        help="Exit non-zero if overall fidelity falls below this (0-1). Turns the "
        "report into a gate for unattended runs.",
    ),
) -> None:
    """What fraction of the user's own specifics survive extraction?

    Needs no gold answers, so it cannot be fitted to the evaluation set — the
    reference is the source text. Run it before spending an ingest on a prompt
    change.
    """
    from llm_long_term_memory.config import ExperimentConfig
    from llm_long_term_memory.ingest import Extractor, TwoStageExtractor
    from llm_long_term_memory.ingest.fidelity import score_sessions
    from llm_long_term_memory.llm import Limits, QuotaManager, UsageTracker
    from llm_long_term_memory.llm.client import GeminiClient
    from llm_long_term_memory.store import Memory

    settings = Settings()
    cfg = ExperimentConfig.from_yaml(config)
    quota = QuotaManager(
        state_dir=settings.store_dir / "quota",
        default=Limits(rpm=cfg.quota.rpm, tpm=cfg.quota.tpm, rpd=cfg.quota.rpd),
    )
    quota.load_learned()
    usage = UsageTracker()
    client = GeminiClient(settings.require_api_key(), quota=quota, usage=usage)
    use_two_stage = cfg.ingest.two_stage if two_stage is None else two_stage
    extractor = (
        TwoStageExtractor(client, cfg.models.extractor)
        if use_two_stage
        else Extractor(client, cfg.models.extractor)
    )

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
        f"{'two-stage' if use_two_stage else 'single-stage'} · {cfg.models.extractor}\n"
    )

    # Cache the extraction so a change to the *metric* can be re-scored for free.
    # The same sessions were re-extracted four times while the denominator was being
    # corrected; each pass cost requests and returned identical memories.
    import hashlib
    import json as _json

    from llm_long_term_memory.ingest.extract import _PROMPT, EXTRACT_SYSTEM
    from llm_long_term_memory.ingest.extract_facts import _PROMPT as FACTS_PROMPT
    from llm_long_term_memory.ingest.extract_facts import FACTS_SYSTEM
    from llm_long_term_memory.ingest.keying import _PROMPT as KEYING_PROMPT
    from llm_long_term_memory.ingest.keying import KEYING_SYSTEM

    active_prompt = (
        FACTS_SYSTEM + FACTS_PROMPT + KEYING_SYSTEM + KEYING_PROMPT
        if use_two_stage
        else EXTRACT_SYSTEM + _PROMPT
    )
    fingerprint = hashlib.sha1(
        (active_prompt + cfg.models.extractor + str(batch) + str(use_two_stage)).encode()
    ).hexdigest()[:12]
    cache_path = settings.store_dir / "fidelity-cache" / f"{fingerprint}.json"
    cached = _json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
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
        cache_path.write_text(_json.dumps(cached), encoding="utf-8")
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
    artifact = settings.results_dir / "raw" / "fidelity.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        _json.dumps(
            {
                "overall": report.overall,
                "memories": report.memories,
                "memories_per_session": report.memories_per_session,
                "sessions": sessions,
                "holdout": holdout,
                "per_facet": {f: s.recall for f, s in report.per_facet.items()},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    console.print(f"[dim]→ {artifact}[/dim]")

    # Without --min-score this stays a report, which is what it is when a human is
    # reading it. Automation needs a verdict it cannot misread as success: printing
    # a bad number and exiting 0 is how a closed gate gets walked through.
    if min_score is not None and report.overall < min_score:
        console.print(
            f"\n[red]Fidelity gate closed.[/red] "
            f"overall {report.overall:.1%} < required {min_score:.1%}"
        )
        raise typer.Exit(1)
    if min_score is not None:
        console.print(
            f"\n[green]Fidelity gate open.[/green] overall {report.overall:.1%} >= {min_score:.1%}"
        )


lifecycle_app = typer.Typer(help="P5: decay, eviction, and traceable consolidation")
app.add_typer(lifecycle_app, name="lifecycle")


@lifecycle_app.command("run")
def lifecycle_run(
    config: str = typer.Option("configs/baselines.yaml", "--config", "-c"),
    store_name: str = typer.Option("two-stage", help="Memory store filename stem"),
    user_id: str | None = typer.Option(None, help="Only process one namespace"),
) -> None:
    """Apply configured strength decay and optional capacity eviction."""
    from datetime import datetime

    from llm_long_term_memory.config import ExperimentConfig
    from llm_long_term_memory.lifecycle import apply_decay, evict_to_limit
    from llm_long_term_memory.store import SQLiteMemoryStore

    cfg = ExperimentConfig.from_yaml(config)
    settings = Settings()
    store = SQLiteMemoryStore(settings.store_dir / f"{store_name}.db")
    store.initialize()
    users = [user_id] if user_id else store.user_ids()
    now = datetime.now()
    try:
        for namespace in users:
            if cfg.decay.enabled:
                decay = apply_decay(
                    store,
                    namespace,
                    now=now,
                    halflife_days=cfg.decay.halflife_days,
                )
            else:
                decay = None
            eviction = (
                evict_to_limit(store, namespace, limit=cfg.decay.max_memories_per_user)
                if cfg.decay.max_memories_per_user
                else None
            )
            console.print(
                f"{namespace}: decayed={decay.updated if decay else 0} "
                f"evicted={len(eviction.evicted_ids) if eviction else 0}"
            )
    finally:
        store.close()


@lifecycle_app.command("consolidate")
def lifecycle_consolidate(
    config: str = typer.Option("configs/baselines.yaml", "--config", "-c"),
    store_name: str = typer.Option("two-stage", help="Memory store filename stem"),
    user_id: str | None = typer.Option(None, help="Only process one namespace"),
) -> None:
    """Synthesize related active memories while retaining their source evidence."""
    from llm_long_term_memory.config import ExperimentConfig
    from llm_long_term_memory.consolidate import Consolidator
    from llm_long_term_memory.embed import Encoder
    from llm_long_term_memory.llm import Limits, QuotaManager, UsageTracker
    from llm_long_term_memory.llm.client import GeminiClient
    from llm_long_term_memory.store import NumpyFlatIndex, SQLiteMemoryStore

    cfg = ExperimentConfig.from_yaml(config)
    settings = Settings()
    store = SQLiteMemoryStore(settings.store_dir / f"{store_name}.db")
    store.initialize()
    index = NumpyFlatIndex(settings.store_dir / f"{store_name}-index", dim=cfg.models.embedding_dim)
    quota = QuotaManager(
        state_dir=settings.store_dir / "quota",
        default=Limits(rpm=cfg.quota.rpm, tpm=cfg.quota.tpm, rpd=cfg.quota.rpd),
    )
    quota.load_learned()
    usage = UsageTracker()
    consolidator = Consolidator(
        GeminiClient(settings.require_api_key(), quota=quota, usage=usage),
        cfg.models.extractor,
        Encoder(),
        store,
        index,
        similarity_threshold=cfg.consolidation_config.similarity_threshold,
        min_cluster_size=cfg.consolidation_config.min_cluster_size,
        source_strength_multiplier=cfg.consolidation_config.source_strength_multiplier,
    )
    users = [user_id] if user_id else store.user_ids()
    try:
        for namespace in users:
            report = consolidator.consolidate(namespace)
            console.print(
                f"{namespace}: clusters={report.clusters_found} "
                f"created={len(report.memories_created)}"
            )
    finally:
        index.save()
        usage.save(settings.results_dir / "raw" / f"consolidate-{store_name}.usage.json")
        store.close()


influence_app = typer.Typer(help="P6: memory utility measurement and packing")
app.add_typer(influence_app, name="influence")


@influence_app.command("cost")
def influence_cost(
    questions: int = typer.Option(50),
    top_k: int = typer.Option(20),
    leave_one_in: bool = typer.Option(True),
) -> None:
    """What a full influence measurement costs, before committing to it."""
    from llm_long_term_memory.influence import requests_needed

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


@influence_app.command("measure")
def influence_measure(
    variant: str = typer.Option("two_stage", help="Memory-backed evaluation variant"),
    config: str = typer.Option("configs/baselines.yaml", "--config", "-c"),
    limit: int | None = typer.Option(None, help="Override dataset_limit from config"),
    store_name: str | None = typer.Option(None, help="Memory store filename stem"),
    out: str | None = typer.Option(None, help="Output JSONL; defaults by variant"),
    leave_one_in: bool = typer.Option(True, help="Also test each memory on its own"),
    fresh: bool = typer.Option(False, help="Discard the existing checkpoint"),
) -> None:
    """Measure retrieved memories' answer utility for a fixed evaluation variant."""
    from llm_long_term_memory.evaluation.runners.memory import MemoryRunner
    from llm_long_term_memory.influence import measure_influence

    cfg, settings, runner, judge, usage = _build(variant, config, store_name)
    if not isinstance(runner, MemoryRunner):
        raise typer.BadParameter("utility measurement requires a memory-backed variant")

    n = limit if limit is not None else cfg.dataset_limit
    with console.status("Loading dataset…"):
        instances = lme.load(cfg.dataset_variant, settings.data_dir, limit=n)

    path = Path(out) if out else settings.results_dir / "raw" / f"influence-{variant}.jsonl"
    console.print(
        f"[bold]{variant}[/bold] · {len(instances)} questions · top_k={runner.top_k} · "
        f"leave-one-in={leave_one_in}\n[dim]→ {path}[/dim]"
    )

    def retrieve(instance):
        return [(hit.memory, hit.score) for hit in runner.retrieve(instance)]

    def progress(instance, dataset) -> None:
        done = len({row.question_id for row in dataset.rows})
        console.print(f"[green]✓[/green] {done}/{len(instances)} {instance.question_id}")

    outcome = measure_influence(
        instances,
        retrieve,
        runner.answer_with_memories,
        judge,
        path,
        leave_one_in=leave_one_in,
        resume=not fresh,
        on_question=progress,
    )
    usage.save(path.with_suffix(".usage.json"), merge=not fresh)

    if outcome.completed:
        console.print(
            f"[green]✓[/green] Measured {outcome.questions_done} questions "
            f"({len(outcome.dataset.rows)} memory labels)."
        )
    else:
        console.print(
            f"[yellow]Paused after {outcome.questions_done} questions:[/yellow] "
            f"{outcome.stopped_reason}\nRerun the same command to resume."
        )


@influence_app.command("fit")
def influence_fit(
    labels: str = typer.Option("results/raw/influence.jsonl"),
    alpha: float = typer.Option(1.0, help="Ridge penalty"),
    out: str | None = typer.Option(None, help="Predictor path; defaults beside the store"),
) -> None:
    """Fit the utility predictor on measured labels, held out by question."""
    import json

    import numpy as np

    from llm_long_term_memory.influence import FEATURE_NAMES, InfluenceDataset, fit_grouped

    path = Path(labels)
    if not path.exists():
        console.print(f"[yellow]No labels at {path}.[/yellow] Run the measurement first.")
        raise typer.Exit(1)

    data = InfluenceDataset.load(path)
    features_path = path.with_suffix(".features.npy")
    if not features_path.exists():
        console.print(f"[yellow]No features at {features_path}.[/yellow]")
        raise typer.Exit(1)
    header_path = features_path.with_suffix(".json")
    if not header_path.exists():
        console.print(f"[yellow]No feature header at {header_path}.[/yellow] Rerun measurement.")
        raise typer.Exit(1)
    feature_names = tuple(json.loads(header_path.read_text(encoding="utf-8")).get("features", ()))
    if feature_names != FEATURE_NAMES:
        console.print(
            "[yellow]Feature header does not match this code. Rerun measurement.[/yellow]"
        )
        raise typer.Exit(1)

    x = np.load(features_path)
    y = np.array([r.utility for r in data.rows])
    groups = [r.question_id for r in data.rows]

    model, report = fit_grouped(x, y, groups, alpha=alpha, feature_names=feature_names)
    model_path = Path(out) if out else Settings().store_dir / f"{path.stem}-utility-predictor.json"
    model.save(model_path)

    t = Table(title="utility predictor", show_header=False)
    t.add_column(style="cyan")
    t.add_column(justify="right")
    t.add_row("rows", f"{report.n_train:,}")
    t.add_row("held-out rows", f"{report.n_test:,}")
    t.add_row("RMSE", f"{report.rmse:.3f}")
    t.add_row("baseline (predict the mean)", f"{report.baseline_rmse:.3f}")
    t.add_row("beats baseline", "[green]yes[/green]" if report.beats_baseline else "[red]no[/red]")
    t.add_row("rank corr. with relevance", f"{report.spearman_vs_relevance:.2f}")
    t.add_row("saved model", str(model_path))
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


@ingest_app.command("temporal-gate")
def ingest_temporal_gate(
    config: str = typer.Option("configs/baselines.yaml", "--config", "-c"),
    stability: bool = typer.Option(True, help="Run the keyer twice to measure drift"),
) -> None:
    """Can Stage B key facts well enough for temporal resolution to work?

    Four numbers against fixed thresholds. Costs 2-3 requests, against ~380 for an
    ingest plus two evaluation runs — the point is to decide before spending that.
    """
    import json

    from llm_long_term_memory.config import ExperimentConfig
    from llm_long_term_memory.ingest.keying import FactKeyer
    from llm_long_term_memory.ingest.temporal_gate import (
        THRESHOLDS,
        score,
        stability_between,
        unusable,
    )
    from llm_long_term_memory.ingest.temporal_pairs import ALL_PAIRS
    from llm_long_term_memory.llm import Limits, QuotaManager, UsageTracker
    from llm_long_term_memory.llm.client import DailyQuotaExhausted, GeminiClient

    settings = Settings()
    cfg = ExperimentConfig.from_yaml(config)
    quota = QuotaManager(
        state_dir=settings.store_dir / "quota",
        default=Limits(rpm=cfg.quota.rpm, tpm=cfg.quota.tpm, rpd=cfg.quota.rpd),
    )
    quota.load_learned()
    usage = UsageTracker()
    keyer = FactKeyer(
        GeminiClient(settings.require_api_key(), quota=quota, usage=usage), cfg.models.extractor
    )

    # Flattened and interleaved, so the keyer never sees which two facts form a
    # pair. Presenting them adjacently would hand it the answer.
    flat = [f for p in ALL_PAIRS for f in (p.before, p.after)]
    console.print(f"[bold]temporal gate[/bold] · {len(ALL_PAIRS)} pairs · {cfg.models.extractor}\n")

    usage_artifact = settings.results_dir / "raw" / "temporal-gate.usage.json"
    try:
        first = keyer.key(flat)
        drift = None
        if stability:
            drift = stability_between(first, keyer.key(flat))
    except DailyQuotaExhausted as exc:
        # Daily exhaustion is a normal pause in this project, not a crash. Preserve
        # any completed first call from a stability run without replacing an older
        # usage artifact when the local limiter stopped us before making a request.
        if usage.records:
            usage.save(usage_artifact, merge=True)
        console.print(f"\n[yellow]Stopped on quota:[/yellow] {exc}")
        console.print("[dim]Rerun after the reported reset; no gate report was written.[/dim]")
        raise typer.Exit(2) from None

    report = score(
        ALL_PAIRS, [(first[2 * i], first[2 * i + 1]) for i in range(len(ALL_PAIRS))], drift
    )

    t = Table(title="Stage B temporal gate")
    t.add_column("metric", style="cyan")
    t.add_column("measured", justify="right")
    t.add_column("threshold", justify="right")
    t.add_column("", justify="center")
    rows = [
        ("key consistency", report.key_consistency, THRESHOLDS["key_consistency"], ">="),
        ("replacement recall", report.replacement_recall, THRESHOLDS["replacement_recall"], ">="),
        ("false supersede", report.false_supersede, THRESHOLDS["false_supersede"], "<="),
    ]
    if drift is not None:
        rows.append(("cross-run stability", drift, THRESHOLDS["cross_run_stability"], ">="))
    passes = report.passes()
    for name, value, threshold, direction in rows:
        ok = passes.get(name.replace(" ", "_").replace("-", "_"))
        mark = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
        t.add_row(name, f"{value:.1%}", f"{direction}{threshold:.1%}", mark)
    console.print(t)
    console.print(
        f"removal recall {report.removal_recall:.1%} · unkeyable {unusable(first)}/{len(flat)}"
    )

    if report.gate_open:
        console.print("\n[green]Gate open.[/green] A full ingest is worth its quota.")
    else:
        console.print("\n[yellow]Gate closed.[/yellow] Fix Stage B before spending the ingest.")

    for o in report.failures()[:10]:
        why = []
        if not o.same_key:
            why.append(f"keys differ ({o.before.temporal_key} / {o.after.temporal_key})")
        if not o.correct:
            why.append(f"want {o.expected}, got {o.actual}")
        console.print(f"  [dim]{o.attribute}: {'; '.join(why)}[/dim]")

    artifact = settings.results_dir / "raw" / "temporal-gate.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    console.print(f"[dim]→ {artifact}[/dim]")
    usage.save(usage_artifact)

    # A closed gate has to be a non-zero exit. It printed "Gate closed" and exited 0,
    # which reads fine to a human and is invisible to anything checking a return
    # code — so an unattended sequence would have gone straight on to spend ~250
    # requests on the run this gate exists to prevent. Exit 2 is already taken by the
    # quota stop above, which is a pause rather than a verdict.
    if not report.gate_open:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
