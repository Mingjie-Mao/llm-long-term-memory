from __future__ import annotations

from types import SimpleNamespace

from typer.testing import CliRunner

from llm_long_term_memory import cli
from llm_long_term_memory.evaluation.datasets.longmemeval import Stats


def invoke(*args: str):
    return CliRunner().invoke(cli.app, list(args))


def test_secret_mask_never_echoes_a_short_key_and_only_shows_edges_of_a_long_one():
    assert cli._mask("short") == "set (too short?)"
    assert cli._mask(" 123456789012 ") == "1234…9012 (12 chars)"


def test_default_store_keeps_two_stage_variants_separate():
    assert cli._default_store_name("two_stage_coherent") == "two-stage"
    assert cli._default_store_name("naive_rag") == "memories"


def test_doctor_reports_key_paths_and_download_state(monkeypatch, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    first_name = next(iter(cli.lme.VARIANTS.values()))
    (data / first_name).write_text("[]", encoding="utf-8")
    settings = SimpleNamespace(
        has_api_key=True,
        gemini_api_key="123456789012",
        data_dir=data,
        store_dir=tmp_path / "stores",
        results_dir=tmp_path / "results",
    )
    monkeypatch.setattr(cli, "Settings", lambda: settings)

    result = invoke("doctor")

    assert result.exit_code == 0
    assert "1234…9012" in result.stdout
    assert "longmemeval_" in result.stdout
    assert "not downloaded" in result.stdout


def test_doctor_explains_how_to_configure_a_missing_key(monkeypatch, tmp_path):
    settings = SimpleNamespace(
        has_api_key=False,
        gemini_api_key="",
        data_dir=tmp_path / "data",
        store_dir=tmp_path / "stores",
        results_dir=tmp_path / "results",
    )
    monkeypatch.setattr(cli, "Settings", lambda: settings)

    result = invoke("doctor")

    assert result.exit_code == 0
    assert "No API key" in result.stdout
    assert "Never commit it" in result.stdout


def test_data_download_delegates_variant_directory_and_force(monkeypatch, tmp_path):
    downloaded = tmp_path / "dataset.json"
    downloaded.write_bytes(b"[]")
    calls = []

    def fake_download(variant, data_dir, force=False):
        calls.append((variant, data_dir, force))
        return downloaded

    monkeypatch.setattr(cli.lme, "download", fake_download)

    result = invoke("data", "download", "--variant", "m", "--data-dir", str(tmp_path), "--force")

    assert result.exit_code == 0
    assert calls == [("m", str(tmp_path), True)]
    assert downloaded.name in result.stdout


def stats() -> Stats:
    return Stats(
        variant="s",
        n_questions=2,
        n_abstention=1,
        question_types={"single-session-user": 1, "multi-session": 1},
        n_sessions=4,
        n_unique_sessions=2,
        n_turns=8,
        total_chars=1000,
        est_total_tokens=250,
        median_sessions_per_q=2,
        median_tokens_per_q=125,
    )


def test_data_stats_renders_measured_corpus_fields(monkeypatch):
    monkeypatch.setattr(cli.lme, "load", lambda *args, **kwargs: [object()])
    monkeypatch.setattr(cli.lme, "compute_stats", lambda *args, **kwargs: stats())

    result = invoke("data", "stats", "--variant", "s", "--limit", "2")

    assert result.exit_code == 0
    assert "session sharing factor" in result.stdout
    assert "2.00x" in result.stdout
    assert "multi-session" in result.stdout


def test_data_plan_renders_request_and_day_tradeoffs(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.lme, "load", lambda *args, **kwargs: [object()])
    monkeypatch.setattr(cli.lme, "compute_stats", lambda *args, **kwargs: stats())

    result = invoke("data", "plan", "--data-dir", str(tmp_path), "--rpd", "1")

    assert result.exit_code == 0
    assert "Ingestion budget" in result.stdout
    assert "sessions/request" in result.stdout
    assert "degrade extraction quality" in " ".join(result.stdout.split())
