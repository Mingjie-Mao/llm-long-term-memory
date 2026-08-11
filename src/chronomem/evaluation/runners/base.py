"""What every evaluated system must implement.

A runner turns one benchmark instance into one answer, and reports how many context
tokens it spent doing so. That token count is half the point of the project: the
results table is accuracy *per token*, and a system that wins on accuracy while
spending the whole context window has not won anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from chronomem.evaluation.datasets.longmemeval import Instance

ANSWER_SYSTEM = (
    "You answer questions about a user based on their chat history. Answer directly "
    "and concisely. If the history does not contain the information, say you do not "
    "know rather than guessing."
)


@dataclass(slots=True)
class Answer:
    text: str
    context_tokens: int
    """Tokens of retrieved/assembled context placed in the prompt. Excludes the
    question and system prompt so that variants are compared on the part they
    actually control."""

    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    retrieved_ids: list[str] = field(default_factory=list)
    notes: dict = field(default_factory=dict)


class Runner(Protocol):
    name: str

    def prepare(self, instance: Instance) -> None:
        """Build whatever index or store this system needs for one instance.

        Called before `answer`. Runners that ingest per-instance (every variant
        except `full_context`) do their work here.
        """
        ...

    def answer(self, instance: Instance) -> Answer: ...
