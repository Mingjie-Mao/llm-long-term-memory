from __future__ import annotations

from contextlib import contextmanager

from typer.testing import CliRunner

from llm_long_term_memory.cli import app
from llm_long_term_memory.config import ExperimentConfig, ModelConfig
from llm_long_term_memory.evaluation.datasets.longmemeval import (
    HaystackSession,
    HaystackTurn,
    Instance,
)
from llm_long_term_memory.ingest import IngestOutcome, IngestProgress
from llm_long_term_memory.locking import AlreadyRunning


class StubEncoder:
    dim = 4

    def __init__(self, *args, **kwargs):
        pass


def _instance() -> Instance:
    session = HaystackSession("s1", "2026/01/05", [HaystackTurn("user", "hello")])
    return Instance("q1", "single-session-user", "q", "a", "2026/01/06", [session], ["s1"])


def _invoke_ingest(monkeypatch, tmp_path, outcome: IngestOutcome | BaseException):
    import llm_long_term_memory.embed as embed_module
    import llm_long_term_memory.ingest as ingest_module

    config = tmp_path / "config.yaml"
    ExperimentConfig(
        models=ModelConfig(extractor="extractor-test", embedding_dim=4),
        dataset_limit=1,
    ).to_yaml(config)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-used")
    monkeypatch.setenv("LLTM_STORE_DIR", str(tmp_path / "stores"))
    monkeypatch.setenv("LLTM_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(embed_module, "Encoder", StubEncoder)
    monkeypatch.setattr("llm_long_term_memory.cli.lme.load", lambda *args, **kwargs: [_instance()])

    class StubPipeline:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    monkeypatch.setattr(ingest_module, "IngestionPipeline", StubPipeline)
    return CliRunner().invoke(
        app,
        ["ingest", "run", "--config", str(config), "--store-name", "cli-test"],
    )


def test_ingest_cli_returns_nonzero_when_quota_left_work_pending(monkeypatch, tmp_path):
    result = _invoke_ingest(
        monkeypatch,
        tmp_path,
        IngestOutcome(IngestProgress(), completed=False, stopped_reason="daily quota exhausted"),
    )

    assert result.exit_code == 2
    assert "Stopped: daily quota exhausted" in result.stdout
    assert "sessions with terminal status" in result.stdout
    assert "raw-only pending sessions" in result.stdout


def test_ingest_cli_reports_content_blocked_sessions_as_terminal(monkeypatch, tmp_path):
    progress = IngestProgress(blocked_sessions={"q1:s1"}, blocked_batches=1)

    result = _invoke_ingest(
        monkeypatch,
        tmp_path,
        IngestOutcome(progress, completed=True),
    )

    assert result.exit_code == 0
    assert "content-blocked sessions" in result.stdout
    assert "sessions with terminal status" in result.stdout


def test_ingest_cli_reports_a_concurrent_writer_without_a_traceback(monkeypatch, tmp_path):
    @contextmanager
    def occupied(*args, **kwargs):
        raise AlreadyRunning("another ingest owns this store")
        yield

    monkeypatch.setattr("llm_long_term_memory.cli.exclusive", occupied)
    result = _invoke_ingest(
        monkeypatch,
        tmp_path,
        IngestOutcome(IngestProgress(), completed=True),
    )

    assert result.exit_code == 2
    assert "another ingest owns this store" in result.stdout
    assert "Traceback" not in result.stdout


def test_ingest_cli_checkpoints_usage_before_propagating_an_unexpected_failure(
    monkeypatch, tmp_path
):
    result = _invoke_ingest(monkeypatch, tmp_path, RuntimeError("unexpected"))

    assert result.exit_code == 1
    usage_path = tmp_path / "results" / "raw" / "cli-test.ingest.usage.json"
    assert usage_path.exists()
