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

import hashlib
import json
import os
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

    @property
    def _manifest_path(self) -> Path:
        return self.path.with_suffix(".manifest.json")

    def _load(self) -> None:
        if self._vec_path.exists() and self._ids_path.exists():
            if self._manifest_path.exists():
                manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
                for artifact in (self._vec_path, self._ids_path):
                    expected = manifest["sha256"][artifact.name]
                    actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
                    if actual != expected:
                        raise ValueError(
                            f"vector index generation is incomplete: {artifact.name} "
                            "does not match its commit manifest; rebuild the index"
                        )
            self._vectors = np.load(self._vec_path)
            self._ids = json.loads(self._ids_path.read_text(encoding="utf-8"))
            if self._vectors.ndim != 2 or self._vectors.shape[1] != self.dim:
                raise ValueError(
                    "vector index dimension mismatch: "
                    f"{self._vectors.shape}, expected (*, {self.dim})"
                )
            if self._vectors.shape[0] != len(self._ids):
                raise ValueError(
                    f"vector row/id mismatch: {self._vectors.shape[0]} rows, {len(self._ids)} ids"
                )
            if len(set(self._ids)) != len(self._ids):
                raise ValueError("vector index contains duplicate memory ids; rebuild the index")

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

    def remove(self, ids: list[str]) -> int:
        """Physically remove ids and their derived vectors, preserving order."""
        if not ids:
            return 0
        doomed = set(ids)
        keep = [i for i, memory_id in enumerate(self._ids) if memory_id not in doomed]
        removed = len(self._ids) - len(keep)
        if not removed:
            return 0
        self._vectors = self._vectors[keep] if keep else np.zeros((0, self.dim), dtype=np.float32)
        self._ids = [self._ids[i] for i in keep]
        return removed

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        vector_tmp = self._vec_path.with_suffix(self._vec_path.suffix + ".tmp")
        ids_tmp = self._ids_path.with_suffix(self._ids_path.suffix + ".tmp")
        manifest_tmp = self._manifest_path.with_suffix(self._manifest_path.suffix + ".tmp")
        with vector_tmp.open("wb") as handle:
            np.save(handle, self._vectors)
            handle.flush()
            os.fsync(handle.fileno())
        with ids_tmp.open("w", encoding="utf-8") as handle:
            json.dump(self._ids, handle)
            handle.flush()
            os.fsync(handle.fileno())
        vector_tmp.replace(self._vec_path)
        ids_tmp.replace(self._ids_path)
        manifest = {
            "schema_version": 1,
            "rows": len(self._ids),
            "dim": self.dim,
            "sha256": {
                self._vec_path.name: hashlib.sha256(self._vec_path.read_bytes()).hexdigest(),
                self._ids_path.name: hashlib.sha256(self._ids_path.read_bytes()).hexdigest(),
            },
        }
        with manifest_tmp.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        manifest_tmp.replace(self._manifest_path)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self._ids)

    def validate_ids(self, memory_ids: set[str]) -> None:
        indexed = set(self._ids)
        if len(indexed) != len(self._ids):
            raise ValueError("vector index contains duplicate memory ids; rebuild the index")
        if indexed != memory_ids:
            missing = sorted(memory_ids - indexed)
            orphaned = sorted(indexed - memory_ids)
            raise ValueError(
                "SQLite/vector index mismatch: "
                f"{len(missing)} memories lack vectors and {len(orphaned)} vectors lack memories; "
                "rebuild the index"
            )

    def __len__(self) -> int:
        return len(self._ids)
