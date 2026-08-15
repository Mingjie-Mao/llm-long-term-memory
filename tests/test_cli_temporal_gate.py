from __future__ import annotations

from typer.testing import CliRunner

from llm_long_term_memory.cli import app
from llm_long_term_memory.config import ExperimentConfig, ModelConfig
from llm_long_term_memory.ingest.keying import FactKeyer
from llm_long_term_memory.llm import Wait
from llm_long_term_memory.llm.client import DailyQuotaExhausted


def test_temporal_gate_treats_daily_quota_as_a_pause(monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    ExperimentConfig(models=ModelConfig(extractor="extractor-test")).to_yaml(config)
    results_dir = tmp_path / "results"
    usage_artifact = results_dir / "raw" / "temporal-gate.usage.json"
    usage_artifact.parent.mkdir(parents=True)
    usage_artifact.write_text("existing usage must survive", encoding="utf-8")

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-used")
    monkeypatch.setenv("CHRONOMEM_STORE_DIR", str(tmp_path / "stores"))
    monkeypatch.setenv("CHRONOMEM_RESULTS_DIR", str(results_dir))

    def quota_pause(self, facts):
        raise DailyQuotaExhausted("extractor-test", Wait(3600.0, "rpd"))

    monkeypatch.setattr(FactKeyer, "key", quota_pause)

    result = CliRunner().invoke(
        app,
        ["ingest", "temporal-gate", "--config", str(config)],
    )

    assert result.exit_code == 2
    assert "Stopped on quota" in result.stdout
    assert "Rerun after the reported reset" in result.stdout
    assert "Traceback" not in result.stdout
    assert usage_artifact.read_text(encoding="utf-8") == "existing usage must survive"
