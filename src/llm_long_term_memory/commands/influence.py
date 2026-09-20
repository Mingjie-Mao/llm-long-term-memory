"""Memory influence measurement and predictor commands."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import typer
from rich.table import Table

from llm_long_term_memory.commands.common import console
from llm_long_term_memory.config import Settings
from llm_long_term_memory.evaluation.datasets import longmemeval as lme

RunnerBuilder = Callable[..., tuple[Any, Any, Any, Any, Any]]


def create_influence_app(build_runner: RunnerBuilder) -> typer.Typer:
    """Create the command group with the CLI runner factory injected explicitly."""
    influence_app = typer.Typer(help="P6: memory utility measurement and packing")

    @influence_app.command("cost")
    def influence_cost(
        questions: int = typer.Option(50),
        top_k: int = typer.Option(20),
        leave_one_in: bool = typer.Option(True),
    ) -> None:
        """What a full influence measurement costs, before committing to it."""
        from llm_long_term_memory.influence import requests_needed

        answers = requests_needed(questions, top_k, leave_one_in)
        table = Table(show_header=False)
        table.add_column(style="cyan")
        table.add_column(justify="right")
        table.add_row("answer calls", f"{answers:,}")
        table.add_row("judge calls", f"{answers:,}")
        table.add_row("total requests", f"{answers * 2:,}")
        table.add_row("days at 500/day", f"{answers * 2 / 500:.1f}")
        console.print(table)
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

        cfg, settings, runner, judge, usage = build_runner(variant, config, store_name)
        if not isinstance(runner, MemoryRunner):
            raise typer.BadParameter("utility measurement requires a memory-backed variant")

        question_limit = limit if limit is not None else cfg.dataset_limit
        with console.status("Loading dataset…"):
            instances = lme.load(cfg.dataset_variant, settings.data_dir, limit=question_limit)

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
            console.print(
                f"[yellow]No feature header at {header_path}.[/yellow] Rerun measurement."
            )
            raise typer.Exit(1)
        feature_names = tuple(
            json.loads(header_path.read_text(encoding="utf-8")).get("features", ())
        )
        if feature_names != FEATURE_NAMES:
            console.print(
                "[yellow]Feature header does not match this code. Rerun measurement.[/yellow]"
            )
            raise typer.Exit(1)

        features = np.load(features_path)
        utilities = np.array([row.utility for row in data.rows])
        groups = [row.question_id for row in data.rows]

        model, report = fit_grouped(
            features,
            utilities,
            groups,
            alpha=alpha,
            feature_names=feature_names,
        )
        model_path = (
            Path(out) if out else Settings().store_dir / f"{path.stem}-utility-predictor.json"
        )
        model.save(model_path)

        table = Table(title="utility predictor", show_header=False)
        table.add_column(style="cyan")
        table.add_column(justify="right")
        table.add_row("rows", f"{report.n_train:,}")
        table.add_row("held-out rows", f"{report.n_test:,}")
        table.add_row("RMSE", f"{report.rmse:.3f}")
        table.add_row("baseline (predict the mean)", f"{report.baseline_rmse:.3f}")
        table.add_row(
            "beats baseline",
            "[green]yes[/green]" if report.beats_baseline else "[red]no[/red]",
        )
        table.add_row("rank corr. with relevance", f"{report.spearman_vs_relevance:.2f}")
        table.add_row("saved model", str(model_path))
        console.print(table)

        if report.spearman_vs_relevance > 0.9:
            console.print(
                "\n[yellow]Predicted utility tracks the retrieval score almost exactly.[/yellow] "
                "The experiment's question — can downstream utility beat relevance for "
                "selection — is answered no at this scale."
            )

        coefficients = Table(title="coefficients")
        coefficients.add_column("feature", style="cyan")
        coefficients.add_column("weight", justify="right")
        for name, weight in sorted(report.coefficients.items(), key=lambda item: -abs(item[1])):
            coefficients.add_row(name, f"{weight:+.3f}")
        console.print(coefficients)

    return influence_app
