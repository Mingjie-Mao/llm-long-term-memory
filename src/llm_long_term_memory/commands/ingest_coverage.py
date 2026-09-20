"""Low-cost extraction coverage diagnostic command."""

from __future__ import annotations

import json
from dataclasses import asdict

import typer
from rich.table import Table

from llm_long_term_memory.commands.common import console
from llm_long_term_memory.config import Settings
from llm_long_term_memory.evaluation.datasets import longmemeval as lme


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
    from llm_long_term_memory.evaluation.extraction_coverage import (
        UNMEASURABLE_TYPES,
        evaluate_coverage,
    )
    from llm_long_term_memory.ingest import Extractor
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
    extractor = Extractor(
        GeminiClient(settings.require_api_key(), quota=quota, usage=usage),
        cfg.models.extractor,
    )

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
        instances,
        lambda sessions: extractor.extract(sessions).memories,
        on_case=progress,
    )

    console.print()
    table = Table(title="literal coverage", show_header=False)
    table.add_column(style="cyan")
    table.add_column(justify="right")
    table.add_row(
        "[bold]structured literal (measurable)[/bold]",
        f"[bold]{report.rate:.1%}[/bold]",
    )
    table.add_row("source literal (measurable)", f"{report.source_literal_rate:.1%}")
    table.add_row(
        "[dim]structured literal (all types)[/dim]",
        f"[dim]{report.rate_all:.1%}[/dim]",
    )
    table.add_row("measurable questions", str(len(report.measurable)))
    table.add_row("questions", str(report.n))
    table.add_row("memories / question", f"{report.memories_per_session:.1f}")
    console.print(table)

    by_type = Table(title="by question type")
    by_type.add_column("type", style="cyan")
    by_type.add_column("n", justify="right")
    by_type.add_column("covered", justify="right")
    for question_type, (total, covered) in report.by_type().items():
        note = " [dim](not string-matchable)[/dim]" if question_type in UNMEASURABLE_TYPES else ""
        by_type.add_row(question_type + note, str(total), f"{covered}/{total}")
    console.print(by_type)

    for case in report.misses()[:show_misses]:
        console.print(f"\n[red]MISS[/red] {case.question_type} · gold=[yellow]{case.gold}[/yellow]")
        console.print(f"  Q: {case.question[:150]}")
        for memory in case.memories[:6]:
            console.print(f"  [dim]- {memory[:130]}[/dim]")

    usage.save(settings.results_dir / "raw" / "coverage.usage.json")
    destination = settings.results_dir / "raw" / "coverage.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "rate_measurable": report.rate,
                "rate_all": report.rate_all,
                "source_literal_rate_measurable": report.source_literal_rate,
                "memories_per_question": report.memories_per_session,
                "cases": [asdict(case) for case in report.cases],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"\n[green]✓[/green] {destination}")
