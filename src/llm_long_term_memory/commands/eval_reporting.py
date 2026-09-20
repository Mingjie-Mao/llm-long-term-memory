"""Evaluation reporting, audit, comparison, and manifest commands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from llm_long_term_memory.commands.common import console
from llm_long_term_memory.config import Settings
from llm_long_term_memory.evaluation.datasets import longmemeval as lme

ALL_VARIANTS = typer.Argument(None, help="Defaults to every run found")


def eval_report(
    variants: list[str] = ALL_VARIANTS,
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
        load_report(raw / f"{variant}.jsonl", variant=variant)
        for variant in order
        if (raw / f"{variant}.jsonl").exists()
    ]
    if not reports:
        console.print("[yellow]No matching runs.[/yellow]")
        raise typer.Exit(1)

    table = render_table(reports)
    console.print(table)
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(table + "\n", encoding="utf-8")
    console.print(f"\n[green]✓[/green] {destination}")


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
    source = settings.results_dir / "raw" / f"{variant}.jsonl"
    if not source.exists():
        console.print(f"[red]No run at {source}[/red]")
        raise typer.Exit(1)

    report = load_report(source, variant=variant)
    instances = lme.load(cfg.dataset_variant, settings.data_dir, limit=cfg.dataset_limit)
    questions = {instance.question_id: instance.question for instance in instances}

    destination = write_worksheet(
        report.results,
        questions,
        settings.results_dir / f"labels-{variant}.csv",
        n=n,
    )
    console.print(
        f"[green]✓[/green] {destination}\n\n"
        "Fill in the [cyan]human[/cyan] column with 1 (correct) or 0 (wrong) for each "
        "row, judging the [cyan]hypothesis[/cyan] against the [cyan]gold[/cyan] answer.\n"
        f"Then run: [cyan]lltm eval agreement {variant}[/cyan]"
    )


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
        table = Table(title="disagreements")
        table.add_column("question_id", style="cyan")
        table.add_column("judge")
        table.add_column("human")
        for question_id, judge, human in agreement.disagreements:
            table.add_row(
                question_id,
                "correct" if judge else "wrong",
                "correct" if human else "wrong",
            )
        console.print(table)

    save_agreement(agreement, settings.results_dir / f"agreement-{variant}.json")


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
    destination = settings.results_dir / "manifests" / f"{name}.json"
    if destination.exists() and not force:
        existing = load_manifest(destination)
        raise typer.BadParameter(
            f"{destination} already exists with {len(existing)} questions. Refusing to "
            "regenerate: a frozen set that moves is not frozen. Pass --force only "
            "if no published result depends on it."
        )

    every = lme.load(variant, settings.data_dir)
    if exclude:
        banned = set(load_manifest(exclude).question_ids)
        every = [instance for instance in every if instance.question_id not in banned]
        console.print(f"[dim]excluded {len(banned)} ids; {len(every)} remain[/dim]")

    chosen = stratify(every, n, seed=seed)
    manifest = Manifest(
        name=name,
        variant=variant,
        seed=seed,
        question_ids=tuple(instance.question_id for instance in chosen),
        note=note,
    )
    manifest.save(destination)

    by_type: dict[str, int] = {}
    for instance in chosen:
        by_type[instance.question_type] = by_type.get(instance.question_type, 0) + 1
    table = Table(title=f"frozen: {destination}", show_header=False)
    table.add_column(style="cyan")
    table.add_column(justify="right")
    for key in sorted(by_type):
        table.add_row(key, str(by_type[key]))
    table.add_row("[bold]total", f"[bold]{len(manifest)}")
    console.print(table)


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
    for variant in (variant_a, variant_b):
        path = raw / f"{variant}.jsonl"
        if not path.exists():
            console.print(f"[red]No run at {path}[/red]")
            raise typer.Exit(1)
        reports[variant] = load_report(path, variant=variant)

    comparison = paired(reports[variant_a], reports[variant_b])
    table = Table(title=f"{variant_a} vs {variant_b}", show_header=False)
    table.add_column(style="cyan")
    table.add_column(justify="right")
    table.add_row("paired questions", str(comparison.n_paired))
    table.add_row("both right", str(comparison.both_right))
    table.add_row("both wrong", str(comparison.both_wrong))
    table.add_row(f"[green]{variant_b} wins[/green]", str(comparison.b_wins))
    table.add_row(f"[red]{variant_b} losses[/red]", str(comparison.b_losses))
    table.add_row("headline delta", f"{comparison.accuracy_delta:+.1%}")
    table.add_row("[bold]p-value[/bold]", f"[bold]{comparison.p_value:.4f}[/bold]")
    console.print(table)
    style = "green" if comparison.significant else "yellow"
    console.print(f"\n[{style}]{comparison.verdict()}[/{style}]")


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

    accuracies = [load_report(path).accuracy for path in paths]
    console.print(Variability(variant, accuracies).summary())
