from __future__ import annotations

import json

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


def test_remove_deletes_ids_and_vectors_and_persists(tmp_path):
    path = tmp_path / "idx"
    idx = NumpyFlatIndex(path, dim=2)
    idx.add(["keep", "drop"], np.array([[1, 0], [0, 1]], dtype=np.float32))

    assert idx.remove(["drop", "missing"]) == 1
    assert idx.search(np.array([0, 1], dtype=np.float32), limit=2) == [("keep", 0.0)]
    idx.save()

    loaded = NumpyFlatIndex(path, dim=2)
    assert len(loaded) == 1
    assert loaded.search(np.array([1, 0], dtype=np.float32), limit=2)[0][0] == "keep"


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


def test_committed_generation_detects_a_torn_pair(tmp_path):
    path = tmp_path / "idx"
    index = NumpyFlatIndex(path, dim=2)
    index.add(["a"], np.array([[1, 0]], dtype=np.float32))
    index.save()
    path.with_suffix(".ids.json").write_text(json.dumps(["other"]), encoding="utf-8")

    with pytest.raises(ValueError, match="generation is incomplete"):
        NumpyFlatIndex(path, dim=2)


def test_index_validates_exactly_against_database_ids(tmp_path):
    index = NumpyFlatIndex(tmp_path / "idx", dim=2)
    index.add(["a", "orphan"], np.array([[1, 0], [0, 1]], dtype=np.float32))

    with pytest.raises(ValueError, match="1 memories lack vectors and 1 vectors lack memories"):
        index.validate_ids({"a", "missing"})
