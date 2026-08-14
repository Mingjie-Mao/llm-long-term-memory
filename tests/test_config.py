from __future__ import annotations

import pytest

from chronomem.config import ExperimentConfig, RetrievalWeights, Settings


def test_defaults_are_the_naive_rag_baseline():
    """v1 must be semantic-only: the ablation's first row is 'vector search alone'."""
    cfg = ExperimentConfig()
    assert cfg.retrieval.weights.enabled() == ["semantic"]
    assert not cfg.temporal_resolution
    assert not cfg.consolidation
    assert not cfg.decay.enabled
    assert not cfg.pack.enabled


def test_enabled_lists_only_nonzero_signals():
    w = RetrievalWeights(semantic=1.0, bm25=0.5, recency=0.0, importance=0.2, entity=0.0)
    assert w.enabled() == ["semantic", "bm25", "importance"]


def test_yaml_roundtrip(tmp_path):
    cfg = ExperimentConfig(
        name="v4-temporal",
        description="hybrid retrieval + temporal resolution",
        temporal_resolution=True,
    )
    cfg.retrieval.weights = RetrievalWeights(semantic=1.0, bm25=0.6, recency=0.3)
    cfg.pack.token_budget = 4_000
    cfg.pack.enabled = True

    path = tmp_path / "v4.yaml"
    cfg.to_yaml(path)
    loaded = ExperimentConfig.from_yaml(path)

    assert loaded.name == "v4-temporal"
    assert loaded.temporal_resolution
    assert loaded.pack.token_budget == 4_000
    assert loaded.pack.enabled
    assert loaded.retrieval.weights.enabled() == ["semantic", "bm25", "recency"]


def test_settings_reads_key_from_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-key")
    s = Settings()
    assert s.has_api_key
    assert s.require_api_key() == "AIza-test-key"


def test_settings_reads_key_from_dotenv(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    (tmp_path / ".env").write_text("GEMINI_API_KEY=from-dotenv\n")
    monkeypatch.chdir(tmp_path)
    assert Settings().require_api_key() == "from-dotenv"


def test_missing_key_raises_actionable_error(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    s = Settings()
    assert not s.has_api_key
    with pytest.raises(RuntimeError, match=r"aistudio\.google\.com"):
        s.require_api_key()


def test_whitespace_only_key_is_not_a_key(monkeypatch, tmp_path):
    """A trailing newline from `echo >> .env` must not read as a configured key."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "   \n")
    assert not Settings().has_api_key


def test_path_overrides_are_namespaced(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHRONOMEM_DATA_DIR", "/tmp/lme")
    monkeypatch.setenv("DATA_DIR", "/should/be/ignored")
    assert str(Settings().data_dir) == "/tmp/lme"


def test_model_ids_are_not_guessed():
    """Model IDs must come from the provider's model-list endpoint, not from a
    constant somebody typed from memory."""
    cfg = ExperimentConfig()
    assert cfg.models.extractor == ""
    assert cfg.models.answerer == ""
    assert cfg.models.judge == ""
