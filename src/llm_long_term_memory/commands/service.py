"""Root-level maintenance and MCP serving commands."""

from __future__ import annotations

import typer
from rich.table import Table

from llm_long_term_memory.commands.common import console
from llm_long_term_memory.config import Settings


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

    del config  # Retained for CLI compatibility; resolution itself is pure SQL.
    settings = Settings()
    store = SQLiteMemoryStore(settings.store_dir / f"{store_name}.db")
    store.initialize()
    try:
        stats = TemporalResolver(store).resolve_everything()

        table = Table(title="temporal resolution", show_header=False)
        table.add_column(style="cyan")
        table.add_column(justify="right")
        table.add_row("keys examined", f"{stats.keys_examined:,}")
        table.add_row("keys resolved", f"{stats.keys_resolved:,}")
        table.add_row("superseded", f"{stats.superseded:,}")
        table.add_row("restatements", f"{stats.restatements:,}")
        table.add_row("promoted back", f"{stats.promoted:,}")
        table.add_row("undated (unresolvable)", f"{stats.skipped_undated:,}")
        table.add_row("active memories", f"{store.count(status='active'):,}")
        console.print(table)
    finally:
        store.close()


def mcp_serve(
    transport: str = typer.Option("stdio", help="stdio | http"),
    config: str = typer.Option("configs/fallback.yaml", "--config", "-c"),
    store_name: str = typer.Option("two-stage-p10", help="Memory store filename stem"),
) -> None:
    """Serve the memory over MCP, for Claude Desktop, Cursor, or any MCP client.

    The same `MemoryService` the REST API uses, so the tools cannot drift from the
    behaviour that was measured.
    """
    from llm_long_term_memory.api.service import MemoryService
    from llm_long_term_memory.mcp_server import run

    if transport != "stdio":
        console.print(f"[bold]mcp[/bold] · {transport} · store={store_name}")
    run(transport=transport, service=MemoryService(config_path=config, store_name=store_name))
