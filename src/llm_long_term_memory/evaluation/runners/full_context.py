"""Baseline: put the entire history in the prompt.

The accuracy ceiling a memory system is measured against, and the token-cost floor
it should beat by a wide margin. At ~122k median tokens per question this fits in a
1M-token window, so it is a real baseline here rather than a hypothetical one — but
it is also the row that makes the token column interesting.

Note this is not a trivially strong baseline: long-context models degrade on facts
buried in the middle of a very long input, which is the effect LongMemEval was
built to expose.
"""

from __future__ import annotations

from llm_long_term_memory.evaluation.datasets.longmemeval import Instance
from llm_long_term_memory.llm.client import GeminiClient

from .base import ANSWER_SYSTEM, Answer

_TEMPLATE = """\
Here is the complete chat history with the user, in chronological order.

{history}

Today's date is {date}.

Question: {question}
"""


def render_history(instance: Instance) -> str:
    blocks = []
    for sess in instance.sessions:
        turns = "\n".join(f"{t.role}: {t.content}" for t in sess.turns)
        blocks.append(f"[Session on {sess.date}]\n{turns}")
    return "\n\n".join(blocks)


class FullContextRunner:
    name = "full_context"

    def __init__(self, client: GeminiClient, model: str, max_output_tokens: int = 512) -> None:
        self.client = client
        self.model = model
        self.max_output_tokens = max_output_tokens
        self._history: str = ""

    def prepare(self, instance: Instance) -> None:
        self._history = render_history(instance)

    def answer(self, instance: Instance) -> Answer:
        prompt = _TEMPLATE.format(
            history=self._history, date=instance.question_date, question=instance.question
        )
        completion = self.client.generate(
            role="answerer",
            model=self.model,
            prompt=prompt,
            system=ANSWER_SYSTEM,
            temperature=0.0,
            max_output_tokens=self.max_output_tokens,
            est_input_tokens=int(len(prompt) / 4.6),
        )
        return Answer(
            text=completion.text.strip(),
            # The history is everything but the question and template scaffolding.
            context_tokens=completion.input_tokens,
            prompt_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            latency_ms=completion.api_latency_ms,
            notes={"sessions": len(instance.sessions)},
        )
