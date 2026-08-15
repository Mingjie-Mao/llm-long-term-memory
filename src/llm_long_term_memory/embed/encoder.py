"""Local sentence embeddings.

Runs on the machine rather than through an API because corpus-wide embedding is
~53M tokens and would otherwise consume the same daily request quota that is
already the binding constraint on the project (D5, D11). On Apple silicon this
uses the MPS backend.

The model is loaded lazily so that CLI commands which never embed anything do not
pay a multi-second import of torch.
"""

from __future__ import annotations

import numpy as np


def _pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Encoder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name, device=self._device or _pick_device()
            )
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
