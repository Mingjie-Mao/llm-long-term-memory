"""Cross-encoder reranking, between hybrid recall and top-k truncation.

The failure the frozen v1 run localised is not a recall failure: 40 of 50 source
sessions had a selected memory, but only 12 of those answers were right. The
ranking put a plausible-but-wrong memory above the one carrying the answer. A
bi-encoder cannot fix that — it embeds query and memory independently, so it never
sees the two together and cannot tell that "17 vintage cameras" answers *how many*
while the question asked *how long*.

A cross-encoder does see both, and on that exact case it separates them cleanly
(the duration memory scores 7.2, the count memory 2.6, an unrelated record -11.3).

Two properties make this the right next experiment rather than one more knob:

  * **It spends no API quota.** The daily request cap is the binding constraint on
    everything else in this project (D5); this runs locally on MPS or CPU.
  * **It is a config flag, not a code branch**, so it produces an ablation row
    against an otherwise identical pipeline.

**Ordering is replaced, not blended.** The cross-encoder emits an uncalibrated
logit, while the five hybrid signals are all normalized to [0, 1]. Adding them
would reintroduce exactly the scale bug documented at the top of `hybrid.py`, where
an unbounded term silently dominates a bounded one. Blending is a defensible second
experiment, but it needs its own normalization and its own weight, so it is not
smuggled in here.

The hybrid score is retained on every result alongside the rerank score. That is
what makes the reranker inspectable: "ranked 7th by hybrid, 1st after reranking" is
the observation that either justifies the extra latency or condemns it.
"""

from __future__ import annotations

from dataclasses import replace

from .hybrid import RetrievedMemory

# A small, widely used MS MARCO cross-encoder: ~80MB, and fast enough that
# reranking 50 candidates stays well inside the latency budget the results table
# already reports. Pinned rather than floating, for the same reason the LLM model
# IDs are pinned (D3) — a silently changed reranker invalidates comparisons.
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class CrossEncoderReranker:
    """Rescore hybrid candidates by joint query-memory encoding.

    Loaded lazily, like `Encoder`, so that CLI commands which never rerank do not
    pay a multi-second torch import.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = 32,
        candidates: int = 50,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        # Bounds the work: the reranker sees the best `candidates` by hybrid score,
        # not every recalled memory. Raising it trades latency for the chance to
        # rescue a memory the hybrid ranking buried.
        self.candidates = max(1, candidates)
        self._device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - depends on the install
                raise ImportError(
                    "cross-encoder reranking needs the `rerank` extra:\n"
                    "    uv sync --extra rerank\n"
                    "It is off by default because it changed no answers and lowered "
                    "source recall on the measured sweep (results/rerank-pareto.md)."
                ) from exc

            self._model = CrossEncoder(self.model_name, device=self._device or _pick_device())
        return self._model

    def rerank(
        self, query_text: str, candidates: list[RetrievedMemory], limit: int
    ) -> list[RetrievedMemory]:
        """Return the top `limit` candidates ordered by cross-encoder score.

        `candidates` is expected in hybrid-score order; only the leading
        `self.candidates` of them are rescored. Anything beyond that is dropped
        rather than appended, because a memory the hybrid stage ranked below the
        cutoff has not been rescored and its hybrid score is not comparable to a
        rerank score.
        """
        if not candidates:
            return []

        pool = candidates[: self.candidates]
        scores = self.model.predict(
            [(query_text, hit.memory.content) for hit in pool],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        rescored = [
            replace(hit, rerank_score=float(score)) for hit, score in zip(pool, scores, strict=True)
        ]
        # Ties break on the hybrid score, then on id: two memories with identical
        # rerank scores must not reorder between runs, or the ablation table picks
        # up noise that has nothing to do with the configuration under test.
        rescored.sort(key=lambda hit: (-(hit.rerank_score or 0.0), -hit.score, hit.memory.id))
        return rescored[:limit]
