"""Benchmark data commands that do not require an API key."""

from __future__ import annotations

import typer
from rich.table import Table

from llm_long_term_memory.commands.common import console
from llm_long_term_memory.evaluation.datasets import longmemeval as lme

data_app = typer.Typer(help="Benchmark data: download, inspect, plan")


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

    table = Table(title=f"longmemeval_{variant}", show_header=False)
    table.add_column(style="cyan")
    table.add_column(justify="right")
    table.add_row("questions", f"{stats.n_questions:,}")
    table.add_row("  of which abstention", f"{stats.n_abstention:,}")
    table.add_row("sessions (with repeats)", f"{stats.n_sessions:,}")
    table.add_row("sessions (unique)", f"{stats.n_unique_sessions:,}")
    table.add_row(
        "[bold]session sharing factor[/bold]", f"[bold]{stats.session_sharing_factor:.2f}x[/bold]"
    )
    table.add_row("turns", f"{stats.n_turns:,}")
    table.add_row("characters", f"{stats.total_chars:,}")
    table.add_row("est. tokens", f"{stats.est_total_tokens:,}")
    table.add_row("median sessions / question", f"{stats.median_sessions_per_q:,.0f}")
    table.add_row("median est. tokens / question", f"{stats.median_tokens_per_q:,.0f}")
    console.print(table)

    question_types = Table(title="question types")
    question_types.add_column("type", style="cyan")
    question_types.add_column("n", justify="right")
    for name, n in stats.question_types.items():
        question_types.add_row(name, str(n))
    console.print(question_types)

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
    """Estimate ingestion days for each batch size at the given request quota."""
    with console.status("Loading…"):
        instances = lme.load(variant, data_dir)
        stats = lme.compute_stats(instances, variant=variant)

    table = Table(title=f"Ingestion budget — longmemeval_{variant} @ {rpd:,} requests/day")
    table.add_column("sessions/request", justify="right", style="cyan")
    table.add_column("requests", justify="right")
    table.add_column("est. tokens/request", justify="right")
    table.add_column("days", justify="right")

    for sessions_per_request in (1, 5, 10, 20, 40):
        plan = lme.plan_ingestion(stats, sessions_per_request, rpd=rpd, dedupe_sessions=True)
        days = f"{plan.days_at_quota:.1f}"
        style = (
            "green" if plan.fits_in_one_day else ("yellow" if plan.days_at_quota <= 3 else "red")
        )
        table.add_row(
            str(sessions_per_request),
            f"{plan.n_requests:,}",
            f"{plan.est_tokens_per_request:,}",
            f"[{style}]{days}[/{style}]",
        )
    console.print(table)
    console.print(
        "\n[dim]Assumes each unique session is extracted once and its memories reused "
        "across every question referencing it. Larger batches cost fewer requests but "
        "degrade extraction quality — pick the smallest batch that fits the schedule.[/dim]"
    )
