"""Shared CLI presentation objects and the helpers several command groups need."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import typer
from rich.console import Console

from llm_long_term_memory.evaluation.datasets import longmemeval as lme
from llm_long_term_memory.locking import AlreadyRunning, exclusive

console = Console()


@contextmanager
def cli_lock(path: Path, *, what: str):
    """Hold the store's exclusive lock, reporting a rival process as a plain message.

    A concurrent writer is an operator mistake, not a bug, so it exits 2 with one
    line instead of a traceback.
    """
    try:
        with exclusive(path, what=what):
            yield
    except AlreadyRunning as exc:
        console.print(f"[red]STOP[/red]: {exc}")
        raise typer.Exit(code=2) from None


def manifest_instances(manifest, settings) -> list:
    """Every instance of the dataset a manifest's question ids come from."""
    return lme.load(manifest.variant, settings.data_dir)
