from __future__ import annotations

import numpy as np
import pytest

from llm_long_term_memory.store import NumpyFlatIndex


def test_search_returns_nearest_first(tmp_path):
    idx = NumpyFlatIndex(tmp_path / "idx", dim=3)
    idx.add(["a", "b", "c"], np.array([[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]], dtype=np.float32))
    results = idx.search(np.array([1, 0, 0], dtype=np.float32), limit=2)
    assert [r[0] for r in results] == ["a", "c"]
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)


def test_vectors_are_normalized_so_scores_are_cosine(tmp_path):
    """Magnitude must not affect ranking — otherwise long memories would outrank
    short ones purely because of embedding norm."""
    idx = NumpyFlatIndex(tmp_path / "idx", dim=2)
    idx.add(["small", "huge"], np.array([[1, 0], [100, 0]], dtype=np.float32))
    scores = dict(idx.search(np.array([1, 0], dtype=np.float32), limit=2))
    assert scores["small"] == pytest.approx(scores["huge"], abs=1e-5)


def test_persists_across_instances(tmp_path):
    path = tmp_path / "idx"
    a = NumpyFlatIndex(path, dim=2)
    a.add(["x"], np.array([[0, 1]], dtype=np.float32))
    a.save()

    b = NumpyFlatIndex(path, dim=2)
    assert len(b) == 1
    assert b.search(np.array([0, 1], dtype=np.float32), limit=1)[0][0] == "x"


def test_empty_index_returns_nothing(tmp_path):
    idx = NumpyFlatIndex(tmp_path / "idx", dim=4)
    assert idx.search(np.zeros(4, dtype=np.float32), limit=5) == []


def test_limit_larger_than_corpus(tmp_path):
    idx = NumpyFlatIndex(tmp_path / "idx", dim=2)
    idx.add(["a"], np.array([[1, 0]], dtype=np.float32))
    assert len(idx.search(np.array([1, 0], dtype=np.float32), limit=99)) == 1


def test_shape_mismatch_is_rejected(tmp_path):
    idx = NumpyFlatIndex(tmp_path / "idx", dim=3)
    with pytest.raises(ValueError, match="expected"):
        idx.add(["a"], np.array([[1, 0]], dtype=np.float32))
    with pytest.raises(ValueError, match="ids but"):
        idx.add(["a", "b"], np.array([[1, 0, 0]], dtype=np.float32))
