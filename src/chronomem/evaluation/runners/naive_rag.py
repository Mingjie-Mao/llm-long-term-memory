"""Baseline: chunk the history, embed, retrieve top-k, answer.

This is the "ChatGPT + a vector DB" system that most memory projects actually are,
and the row every later variant has to beat. It has no notion of time, no fact
extraction, and no contradiction handling: when the user changed their mind three
times, all three chunks come back and the model picks one.

Chunking is per session rather than per turn. A session is the natural unit here —
turns are short and lose their referents in isolation — and it keeps the vector
count at ~19k per full corpus instead of ~247k.
"""

from __future__ import annotations

from chronomem.embed import Encoder
from chronomem.evaluation.datasets.longmemeval import Instance
from chronomem.llm.client import GeminiClient
from chronomem.store import NumpyFlatIndex

from .base import ANSWER_SYSTEM, Answer

_TEMPLATE = """\
Here are the most relevant excerpts from the chat history with the user.

{context}

Today's date is {date}.

Question: {question}
"""


class NaiveRAGRunner:
    name = "naive_rag"

    def __init__(
        self,
        client: GeminiClient,
        model: str,
        encoder: Encoder,
        top_k: int = 5,
        max_output_tokens: int = 512,
        chars_per_token: float = 4.6,
    ) -> None:
        self.client = client
        self.model = model
        self.encoder = encoder
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens
        self.chars_per_token = chars_per_token
        self._chunks: dict[str, str] = {}
        self._index: NumpyFlatIndex | None = None

    def prepare(self, instance: Instance) -> None:
        """Index one instance's history in memory.

        Built per instance and discarded: LongMemEval questions have their own
        haystacks (measured sharing factor 1.24x), so a shared index would leak
        one question's evidence into another's retrieval.
        """
        self._chunks = {}
        texts, ids = [], []
        for sess in instance.sessions:
            body = "\n".join(f"{t.role}: {t.content}" for t in sess.turns)
            text = f"[Session on {sess.date}]\n{body}"
            self._chunks[sess.session_id] = text
            ids.append(sess.session_id)
            texts.append(text)

        vectors = self.encoder.encode(texts)
        # Ephemeral: an in-memory index that is never saved to disk.
        index = NumpyFlatIndex(path="/dev/null", dim=vectors.shape[1])
        index.add(ids, vectors)
        self._index = index

    def answer(self, instance: Instance) -> Answer:
        assert self._index is not None, "prepare() must run first"
        query = self.encoder.encode_one(instance.question)
        hits = self._index.search(query, limit=self.top_k)
        selected = [self._chunks[mid] for mid, _ in hits]
        context = "\n\n".join(selected)

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
        return Answer(
            text=completion.text.strip(),
            context_tokens=int(len(context) / self.chars_per_token),
            prompt_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            latency_ms=completion.api_latency_ms,
            retrieved_ids=[mid for mid, _ in hits],
            notes={
                "top_k": self.top_k,
                "indexed_sessions": len(self._chunks),
                # Whether the evidence session was retrieved at all. Separates
                # "retrieval missed it" from "the model had it and still failed".
                "evidence_recalled": bool(
                    set(instance.answer_session_ids) & {mid for mid, _ in hits}
                ),
            },
        )
