from __future__ import annotations

import sys
from types import ModuleType
from typing import ClassVar

import numpy as np

from llm_long_term_memory.embed import encoder as encoder_module
from llm_long_term_memory.embed.encoder import Encoder


class FakeSentenceTransformer:
    instances: ClassVar[list[FakeSentenceTransformer]] = []

    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self.calls = []
        self.__class__.instances.append(self)

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return [[1, 2, 3] for _ in texts]


def install_fake_model(monkeypatch):
    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    FakeSentenceTransformer.instances.clear()


def test_device_selection_prefers_mps_then_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert encoder_module._pick_device() == "mps"

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert encoder_module._pick_device() == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert encoder_module._pick_device() == "cpu"


def test_model_is_lazy_and_uses_the_selected_device(monkeypatch):
    install_fake_model(monkeypatch)
    monkeypatch.setattr(encoder_module, "_pick_device", lambda: "cpu-test")
    encoder = Encoder("fake-model", batch_size=7)

    assert encoder._model is None
    assert encoder.dim == 3
    assert encoder.model is encoder._model
    assert len(FakeSentenceTransformer.instances) == 1
    assert encoder.model.model_name == "fake-model"
    assert encoder.model.device == "cpu-test"


def test_encode_normalizes_the_runtime_contract_and_encode_one(monkeypatch):
    install_fake_model(monkeypatch)
    encoder = Encoder("fake-model", device="explicit", batch_size=7)

    vectors = encoder.encode(["one", "two"], show_progress=True)
    one = encoder.encode_one("single")

    assert vectors.dtype == np.float32
    assert vectors.shape == (2, 3)
    np.testing.assert_array_equal(one, np.array([1, 2, 3], dtype=np.float32))
    assert encoder.model.calls[0] == (
        ["one", "two"],
        {
            "batch_size": 7,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "show_progress_bar": True,
        },
    )


def test_empty_encode_has_the_model_dimension_and_float32_dtype(monkeypatch):
    install_fake_model(monkeypatch)
    encoder = Encoder("fake-model", device="cpu")

    result = encoder.encode([])

    assert result.shape == (0, 3)
    assert result.dtype == np.float32
    assert encoder.model.calls == []
