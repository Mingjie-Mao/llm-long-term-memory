"""The `lltm` surface is a contract with scripts, cron jobs and the write-ups.

`cli.py` is being split into `commands/`, and a refactor that quietly drops a flag or
renames a command breaks reproduction of experiments whose logs record the exact
invocation. The golden file records every command and every option name, so a move
has to be deliberate to change it.
"""

from __future__ import annotations

from pathlib import Path

from typer.main import get_command

from llm_long_term_memory.cli import app

GOLDEN = Path(__file__).parent / "cli_surface.txt"


def _surface() -> list[str]:
    lines: list[str] = []

    def walk(cmd, path: str) -> None:
        subs = getattr(cmd, "commands", None)
        if subs:
            for name in sorted(subs):
                walk(subs[name], f"{path} {name}")
        else:
            opts = " ".join(sorted("|".join(p.opts) for p in cmd.params))
            lines.append(f"{path} :: {opts}".rstrip())

    walk(get_command(app), "lltm")
    return lines


def test_command_and_option_names_match_the_recorded_surface():
    recorded = GOLDEN.read_text(encoding="utf-8").splitlines()
    assert _surface() == recorded


def test_every_command_still_carries_its_help_text():
    def walk(cmd, path: str) -> None:
        subs = getattr(cmd, "commands", None)
        if subs:
            for name in sorted(subs):
                walk(subs[name], f"{path} {name}")
        else:
            assert (cmd.help or "").strip(), f"{path} lost its help text"

    walk(get_command(app), "lltm")
