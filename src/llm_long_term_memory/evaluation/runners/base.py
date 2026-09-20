"""What every evaluated system must implement.

A runner turns one benchmark instance into one answer, and reports how many context
tokens it spent doing so. That token count is half the point of the project: the
results table is accuracy *per token*, and a system that wins on accuracy while
spending the whole context window has not won anything.
"""

from __future__ import annotations

from typing import Protocol

# Re-exported. The answering contract is the product's; a runner is the
# benchmark's way of driving it, which is why only this half stayed here.
from llm_long_term_memory.answering import (  # noqa: F401
    ANSWER_PROMPT_VERSION,
    ANSWER_SYSTEM,
    Answer,
    AnswerVerdict,
)
from llm_long_term_memory.evaluation.datasets.longmemeval import Instance


class Runner(Protocol):
    name: str

    def prepare(self, instance: Instance) -> None:
        """Build whatever index or store this system needs for one instance.

        Called before `answer`. Runners that ingest per-instance (every variant
        except `full_context`) do their work here.
        """
        ...

    def answer(self, instance: Instance) -> Answer: ...
