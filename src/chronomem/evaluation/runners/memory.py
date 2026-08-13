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
        token_budget: int = 0,
        utility_model=None,
        type_floors: dict[str, float] | None = None,
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
        # 0 disables packing and keeps the plain top-k truncation, so the P6 rows
        # are a change of selection policy against an otherwise identical pipeline.
        self.token_budget = token_budget
        self.utility_model = utility_model
        self.type_floors = type_floors
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
        #
        # The whole index is scanned rather than a top-N slice of it. The index is
        # global while a question's memories are one namespace of fifty, so any
        # fixed over-fetch returns mostly other questions' facts and the survivors
        # fall well short of top_k: at `top_k * 40` the packed context came to ~250
        # tokens against a ~40-memory namespace, and accuracy read 20% — a
        # measurement of the over-fetch factor, not of the memory system.
        # Scanning is affordable precisely because the store is small (2k vectors),
        # and it makes the namespace filter exact instead of best-effort.
        hits = self.index.search(query, limit=len(self.index))
        memories = self.store.get_many([mid for mid, _ in hits])
        memories = [m for m in memories if m.user_id == namespace]
        if self.temporal:
            memories = [m for m in memories if m.status == "active"]
        return memories[: self.top_k]

    def _pack(self, candidates: list[Memory], scores: dict[str, float], query: str):
        """Select under a token budget instead of truncating at top_k.

        Utilities come from the predictor when one is supplied and from the
        retrieval score otherwise. The fallback is not a placeholder: "pack by
        relevance" is the control the utility-aware packer has to beat, and running
        both through the same knapsack is what isolates the utility signal from the
        packing.
        """
        from chronomem.influence import build as build_features
        from chronomem.pack import pack

        if self.utility_model is None:
            utilities = [scores.get(m.id, 0.0) for m in candidates]
        else:
            import numpy as np

            rows = [
                build_features(
                    m,
                    query=query,
                    semantic_score=scores.get(m.id, 0.0),
                    rank=i,
                    neighbours=candidates,
                ).as_list()
                for i, m in enumerate(candidates)
            ]
            utilities = self.utility_model.predict(np.array(rows, dtype=float)).tolist()

        return pack(candidates, utilities, self.token_budget, type_floors=self.type_floors)

    def answer(self, instance: Instance) -> Answer:
        query = self.encoder.encode_one(instance.question)
        hits = dict(self.index.search(query, limit=len(self.index)))
        selected = self._select(query, instance.question_id)

        packed = None
        if self.token_budget:
            packed = self._pack(selected, hits, instance.question)
            selected = packed.selected

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
                "token_budget": self.token_budget,
                "packed_utilisation": packed.utilisation if packed else None,
                "dropped_negative": packed.dropped_negative if packed else None,
                "evidence_recalled": bool(
                    evidence & {m.source_session_id for m in selected if m.source_session_id}
                ),
            },
        )
