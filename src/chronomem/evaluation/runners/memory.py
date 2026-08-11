"""Answer from the extracted memory store rather than from raw sessions.

The first variant that is actually ChronoMem. Two knobs, so the ablation can
isolate what temporal resolution is worth:

    temporal=False   retrieve over every memory, superseded ones included. This is
                     the store's version of what naive RAG does — all three of
                     "uses TensorFlow", "learning PyTorch", "switched to PyTorch"
                     come back and the model picks.
    temporal=True    retrieve only what is currently in force, and tell the model
                     the validity window of what it is being given.

The retrieval itself is identical in both. Any difference in the results table is
attributable to the timeline, not to ranking.
"""

from __future__ import annotations

import numpy as np

from chronomem.embed import Encoder
from chronomem.evaluation.datasets.longmemeval import Instance
from chronomem.llm.client import GeminiClient
from chronomem.store import Memory, MemoryStore, NumpyFlatIndex

from .base import ANSWER_SYSTEM, Answer

_TEMPLATE = """\
Here is what is known about the user, drawn from their chat history.

{context}

Today's date is {date}.

Question: {question}
"""

TEMPORAL_NOTE = (
    "Each fact is shown with the period it was true for. `(since <date>)` means it "
    "is still true now; `(<date> to <date>)` means it was replaced and is no longer "
    "current. Answer with what is true now unless the question asks about the past."
)


def render_memory(memory: Memory, temporal: bool) -> str:
    if not temporal:
        return f"- {memory.content}"

    start = memory.valid_from or memory.event_time
    if start and memory.valid_to:
        window = f"({start:%Y-%m-%d} to {memory.valid_to:%Y-%m-%d}, no longer current)"
    elif start:
        window = f"(since {start:%Y-%m-%d})"
    else:
        window = "(date unknown)"
    return f"- {memory.content} {window}"


class MemoryRunner:
    def __init__(
        self,
        client: GeminiClient,
        model: str,
        encoder: Encoder,
        store: MemoryStore,
        index: NumpyFlatIndex,
        temporal: bool = False,
        top_k: int = 20,
        max_output_tokens: int = 512,
        chars_per_token: float = 4.6,
    ) -> None:
        self.client = client
        self.model = model
        self.encoder = encoder
        self.store = store
        self.index = index
        self.temporal = temporal
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens
        self.chars_per_token = chars_per_token
        # A plain attribute, not a property: the CLI overrides it so that the
        # results file is named after the config variant rather than after the flag.
        self.name = "chronomem" if temporal else "chronomem_no_temporal"

    def prepare(self, instance: Instance) -> None:
        """Nothing per-instance: the store is built once by `chronomem ingest`.

        This is the structural difference from the baselines, which rebuild an index
        for every question. It is also why this variant's cost is not visible in the
        per-question latency — ingestion is paid once, up front.
        """

    def _select(self, query: np.ndarray, namespace: str) -> list[Memory]:
        # Filtering after ranking, not before, so the two variants share an
        # identical ranking and the ablation isolates the timeline.
        # Over-fetch: the index is global, so a question's own memories are a small
        # slice of it and the namespace filter discards most hits. Without the
        # filter, retrieval for one question returns another question's evidence —
        # LongMemEval questions are independent users with no shared history.
        hits = self.index.search(query, limit=self.top_k * 40)
        memories = self.store.get_many([mid for mid, _ in hits])
        memories = [m for m in memories if m.user_id == namespace]
        if self.temporal:
            memories = [m for m in memories if m.status == "active"]
        return memories[: self.top_k]

    def answer(self, instance: Instance) -> Answer:
        query = self.encoder.encode_one(instance.question)
        selected = self._select(query, instance.question_id)

        body = "\n".join(render_memory(m, self.temporal) for m in selected)
        context = f"{TEMPORAL_NOTE}\n\n{body}" if self.temporal else body

        prompt = _TEMPLATE.format(
            context=context, date=instance.question_date, question=instance.question
        )
        completion = self.client.generate(
            role="answerer",
            model=self.model,
            prompt=prompt,
            system=ANSWER_SYSTEM,
            temperature=0.0,
            max_output_tokens=self.max_output_tokens,
            est_input_tokens=int(len(prompt) / self.chars_per_token),
        )

        evidence = set(instance.answer_session_ids)
        return Answer(
            text=completion.text.strip(),
            context_tokens=int(len(context) / self.chars_per_token),
            prompt_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            latency_ms=completion.api_latency_ms,
            retrieved_ids=[m.id for m in selected],
            notes={
                "top_k": self.top_k,
                "temporal": self.temporal,
                "superseded_shown": sum(1 for m in selected if m.status != "active"),
                "evidence_recalled": bool(
                    evidence & {m.source_session_id for m in selected if m.source_session_id}
                ),
            },
        )
