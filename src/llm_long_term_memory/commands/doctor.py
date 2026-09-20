"""Credential and local-path diagnostics for the CLI."""

from __future__ import annotations

from rich.table import Table

from llm_long_term_memory.commands.common import console
from llm_long_term_memory.evaluation.datasets import longmemeval as lme


def mask_secret(secret: str) -> str:
    """Show enough to identify a key without exposing it."""
    value = secret.strip()
    if len(value) < 12:
        return "set (too short?)"
    return f"{value[:4]}…{value[-4:]} ({len(value)} chars)"


def render_doctor(settings) -> None:
    """Render diagnostics for an already-created settings object."""
    table = Table(show_header=False)
    table.add_column(style="cyan")
    table.add_column()

    if settings.has_api_key:
        table.add_row("GEMINI_API_KEY", f"[green]✓[/green] {mask_secret(settings.gemini_api_key)}")
    else:
        table.add_row("GEMINI_API_KEY", "[red]✗ not found[/red]")

    for label, path in (
        ("data dir", settings.data_dir),
        ("store dir", settings.store_dir),
        ("results dir", settings.results_dir),
    ):
        mark = "[green]✓[/green]" if path.exists() else "[dim]— (created on first use)[/dim]"
        table.add_row(label, f"{mark} {path}")

    for variant, filename in lme.VARIANTS.items():
        path = settings.data_dir / filename
        if path.exists():
            table.add_row(
                f"longmemeval_{variant}", f"[green]✓[/green] {path.stat().st_size / 1e6:.0f} MB"
            )
        else:
            table.add_row(f"longmemeval_{variant}", "[dim]— not downloaded[/dim]")

    console.print(table)

    if not settings.has_api_key:
        console.print(
            "\n[yellow]No API key.[/yellow] Get a free one (no card) at "
            "https://aistudio.google.com/apikey, then:\n"
            "  [cyan]cp .env.example .env[/cyan]  and put the key in it.\n"
            "[dim].env is gitignored. Never commit it or paste it into a chat.[/dim]"
        )
