"""Exact flat vector index backed by numpy.

Why not FAISS: `IndexFlatIP` is a brute-force inner-product scan, which is exactly
what the loop below does. At this corpus size (<1M vectors) the difference is a
constant factor on an operation that is not the bottleneck — the LLM calls are —
so the dependency buys nothing. Approximate indexes (HNSW/IVF) are ruled out for a
different reason: their recall noise is indistinguishable from a regression in the
memory algorithm, which would corrupt the ablation table. If profiling in P6 shows
this is hot, swapping in FAISS means implementing the same `VectorIndex` Protocol.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class NumpyFlatIndex:
    def __init__(self, path: str | Path, dim: int) -> None:
        self.path = Path(path)
        self.dim = dim
        self._ids: list[str] = []
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._load()

    @property
    def _vec_path(self) -> Path:
        return self.path.with_suffix(".npy")

    @property
    def _ids_path(self) -> Path:
        return self.path.with_suffix(".ids.json")

    def _load(self) -> None:
        if self._vec_path.exists() and self._ids_path.exists():
            self._vectors = np.load(self._vec_path)
            self._ids = json.loads(self._ids_path.read_text(encoding="utf-8"))

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if not ids:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected (n, {self.dim}) vectors, got {vectors.shape}")
        if len(ids) != vectors.shape[0]:
            raise ValueError(f"{len(ids)} ids but {vectors.shape[0]} vectors")
        # Normalize so inner product == cosine similarity. Done here rather than at
        # the encoder so every backend gets the same guarantee.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        self._vectors = np.vstack([self._vectors, vectors])
        self._ids.extend(ids)

    def search(self, vector: np.ndarray, limit: int) -> list[tuple[str, float]]:
        if len(self._ids) == 0:
            return []
        q = np.asarray(vector, dtype=np.float32).reshape(-1)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        scores = self._vectors @ q
        k = min(limit, len(self._ids))
        # argpartition is O(n) vs O(n log n) for a full sort; only the top-k is sorted.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self._ids[i], float(scores[i])) for i in top]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self._vec_path, self._vectors)
        self._ids_path.write_text(json.dumps(self._ids), encoding="utf-8")

    def __len__(self) -> int:
        return len(self._ids)
