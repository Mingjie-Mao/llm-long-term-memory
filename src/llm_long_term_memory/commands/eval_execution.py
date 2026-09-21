"""Model-calling evaluation commands with injected runner construction."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import typer
from rich.table import Table

from llm_long_term_memory.commands.common import console
from llm_long_term_memory.evaluation.datasets import longmemeval as lme

RunnerBuilder = Callable[..., tuple[Any, Any, Any, Any, Any]]
ManifestLoader = Callable[[Any, Any], list]


def register_run_commands(
    eval_app: typer.Typer,
    *,
    build_runner: RunnerBuilder,
    manifest_instances: ManifestLoader,
) -> None:
    """Register the primary run and budget-sweep commands in help order."""

    @eval_app.command("run")
    def eval_run(
        variant: str = typer.Argument(
            ...,
            help=(
                "full_context | naive_rag | two_stage | two_stage_no_temporal | "
                "two_stage_hydrated | two_stage_hydrated_no_temporal | two_stage_coherent | "
                "two_stage_coherent_oracle | two_stage_memory_only | two_stage_reasoned | "
                "two_stage_reasoned_evidence | two_stage_synthesis | "
                "two_stage_v2c_memory_only | two_stage_v2c | "
                "two_stage_v2d_memory_only | two_stage_v2d | "
                "two_stage_v5_fixed | two_stage_v5_planned. "
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
            None,
            help="Store filename stem; two_stage defaults to an isolated two-stage store",
        ),
        fresh: bool = typer.Option(False, help="Ignore existing results and start over"),
        top_k: int | None = typer.Option(None, "--top-k", help="Override retrieval.top_k"),
        rerank: bool | None = typer.Option(
            None,
            "--rerank/--no-rerank",
            help="Override retrieval.rerank.enabled",
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

        cfg, settings, runner, judge, usage = build_runner(
            variant, config, store_name, top_k, rerank
        )
        question_limit = limit if limit is not None else cfg.dataset_limit

        with console.status("Loading dataset…"):
            if questions:
                from llm_long_term_memory.evaluation.manifest import load_manifest

                manifest = load_manifest(questions)
                every = manifest_instances(manifest, settings)
                by_id = {instance.question_id: instance for instance in every}
                unknown = [
                    question_id for question_id in manifest.question_ids if question_id not in by_id
                ]
                if unknown:
                    raise typer.BadParameter(
                        f"{len(unknown)} question id(s) not in the {manifest.variant!r} "
                        f"split, first: {unknown[0]!r}"
                    )
                instances = [by_id[question_id] for question_id in manifest.question_ids]
            else:
                instances = lme.load(
                    cfg.dataset_variant,
                    settings.data_dir,
                    limit=question_limit,
                )

        stem = f"{variant}.{label}" if label else variant
        output = settings.results_dir / "raw" / f"{stem}.jsonl"
        runner.name = stem
        console.print(
            f"[bold]{stem}[/bold] · {len(instances)} questions · "
            f"answerer={cfg.models.answerer} judge={cfg.models.judge} · "
            f"top_k={cfg.retrieval.top_k} rerank={cfg.retrieval.rerank.enabled}\n"
            f"[dim]→ {output}[/dim]\n"
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
            runner,
            judge,
            instances,
            output,
            usage=usage,
            resume=not fresh,
            on_progress=progress,
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
            cfg, settings, runner, judge, usage = build_runner(variant, config, store_name)
            if not isinstance(runner, MemoryRunner):
                raise typer.BadParameter("budget sweep requires a memory-backed variant")

            model_path = utility_model or cfg.pack.utility_model_path
            if model_path:
                runner.utility_model = UtilityPredictor.load(model_path)
                selector = "utility"
            runner.token_budget = budget
            runner.type_floors = cfg.pack.type_floors
            runner.name = f"{variant}_{selector}_pack_{budget}"

            question_limit = limit if limit is not None else cfg.dataset_limit
            with console.status(f"Loading dataset for {budget:,} tokens…"):
                instances = lme.load(
                    cfg.dataset_variant,
                    settings.data_dir,
                    limit=question_limit,
                )
            path = settings.results_dir / "raw" / f"{runner.name}.jsonl"
            report = run_eval(
                runner,
                judge,
                instances,
                path,
                usage=usage,
                resume=not fresh,
            )
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
            render_pareto_svg(points, f"{variant}: {selector} packing"),
            encoding="utf-8",
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


def register_repeat_command(
    eval_app: typer.Typer,
    *,
    build_runner: RunnerBuilder,
) -> None:
    """Register repeat after comparison commands to preserve public help order."""

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
            cfg, settings, runner, judge, usage = build_runner(variant, config, store_name)
            runner.name = f"{variant}.rep{number}"
            question_limit = limit if limit is not None else cfg.dataset_limit
            with console.status(f"Loading dataset for repeat {number}/{runs}…"):
                instances = lme.load(
                    cfg.dataset_variant,
                    settings.data_dir,
                    limit=question_limit,
                )
            path = settings.results_dir / "raw" / f"{runner.name}.jsonl"
            report = run_eval(
                runner,
                judge,
                instances,
                path,
                usage=usage,
                resume=not fresh,
            )
            accuracies.append(report.accuracy)
            console.print(f"repeat {number}/{runs}: {report.accuracy:.1%} ({report.n} questions)")
            if not report.completed:
                console.print(f"[yellow]Paused repeat {number}; rerun to resume.[/yellow]")
                return
        console.print(Variability(variant, accuracies).summary())
