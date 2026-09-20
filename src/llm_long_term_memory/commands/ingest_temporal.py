"""Temporal-keying preflight command."""

from __future__ import annotations

import json

import typer
from rich.table import Table

from llm_long_term_memory.commands.common import console
from llm_long_term_memory.config import Settings


def ingest_temporal_gate(
    config: str = typer.Option("configs/baselines.yaml", "--config", "-c"),
    stability: bool = typer.Option(True, help="Run the keyer twice to measure drift"),
) -> None:
    """Can Stage B key facts well enough for temporal resolution to work?

    Four numbers against fixed thresholds. Costs 2-3 requests, against ~380 for an
    ingest plus two evaluation runs — the point is to decide before spending that.
    """
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
        GeminiClient(settings.require_api_key(), quota=quota, usage=usage),
        cfg.models.extractor,
    )

    facts = [fact for pair in ALL_PAIRS for fact in (pair.before, pair.after)]
    console.print(f"[bold]temporal gate[/bold] · {len(ALL_PAIRS)} pairs · {cfg.models.extractor}\n")

    usage_artifact = settings.results_dir / "raw" / "temporal-gate.usage.json"
    try:
        first = keyer.key(facts)
        drift = stability_between(first, keyer.key(facts)) if stability else None
    except DailyQuotaExhausted as exc:
        if usage.records:
            usage.save(usage_artifact, merge=True)
        console.print(f"\n[yellow]Stopped on quota:[/yellow] {exc}")
        console.print("[dim]Rerun after the reported reset; no gate report was written.[/dim]")
        raise typer.Exit(2) from None

    report = score(
        ALL_PAIRS,
        [(first[2 * index], first[2 * index + 1]) for index in range(len(ALL_PAIRS))],
        drift,
    )

    table = Table(title="Stage B temporal gate")
    table.add_column("metric", style="cyan")
    table.add_column("measured", justify="right")
    table.add_column("threshold", justify="right")
    table.add_column("", justify="center")
    rows = [
        ("key consistency", report.key_consistency, THRESHOLDS["key_consistency"], ">="),
        (
            "replacement recall",
            report.replacement_recall,
            THRESHOLDS["replacement_recall"],
            ">=",
        ),
        ("false supersede", report.false_supersede, THRESHOLDS["false_supersede"], "<="),
    ]
    if drift is not None:
        rows.append(("cross-run stability", drift, THRESHOLDS["cross_run_stability"], ">="))
    passes = report.passes()
    for name, value, threshold, direction in rows:
        ok = passes.get(name.replace(" ", "_").replace("-", "_"))
        mark = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
        table.add_row(name, f"{value:.1%}", f"{direction}{threshold:.1%}", mark)
    console.print(table)
    console.print(
        f"removal recall {report.removal_recall:.1%} · unkeyable {unusable(first)}/{len(facts)}"
    )

    if report.gate_open:
        console.print("\n[green]Gate open.[/green] A full ingest is worth its quota.")
    else:
        console.print("\n[yellow]Gate closed.[/yellow] Fix Stage B before spending the ingest.")

    for outcome in report.failures()[:10]:
        reasons = []
        if not outcome.same_key:
            reasons.append(
                f"keys differ ({outcome.before.temporal_key} / {outcome.after.temporal_key})"
            )
        if not outcome.correct:
            reasons.append(f"want {outcome.expected}, got {outcome.actual}")
        console.print(f"  [dim]{outcome.attribute}: {'; '.join(reasons)}[/dim]")

    artifact = settings.results_dir / "raw" / "temporal-gate.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    console.print(f"[dim]→ {artifact}[/dim]")
    usage.save(usage_artifact)

    if not report.gate_open:
        raise typer.Exit(1)
