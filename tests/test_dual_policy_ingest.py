"""One extraction pass, two dedup policies — and the ways the saving could be imaginary."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

from llm_long_term_memory.store import Memory

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

dual = pytest.importorskip("dual_policy_ingest")


class Session:
    def __init__(self, session_id):
        self.session_id = session_id


def memory(**kw):
    return Memory(
        id=kw.pop("id", "mem_1"),
        user_id=kw.pop("user_id", "beam-100K-5"),
        type=kw.pop("type", "semantic"),
        content=kw.pop("content", "The user owns 25 postcards."),
        token_count=kw.pop("token_count", 7),
        event_time=kw.pop("event_time", datetime(2024, 3, 15, 9, 0)),
        entities=kw.pop("entities", ["postcards"]),
        **kw,
    )


def test_a_batch_is_keyed_by_what_went_into_it_not_by_its_position():
    """The two passes build different stores, so a positional key would drift apart; the
    session list is identical in both."""
    sessions = [Session("s00c000"), Session("s00c001")]
    assert dual.batch_key("ns", sessions) == dual.batch_key("ns", list(sessions))
    assert dual.batch_key("ns", sessions) != dual.batch_key("other", sessions)
    assert dual.batch_key("ns", sessions) != dual.batch_key("ns", sessions[:1])


def test_a_recorded_memory_comes_back_the_same_memory():
    original = memory()
    restored = dual.from_record(json.loads(json.dumps(dual.to_record(original))))
    assert restored == original


def test_the_replay_costs_nothing(tmp_path):
    cache = tmp_path / "cache.jsonl"
    sessions = [Session("s00c000")]
    cache.write_text(
        json.dumps(
            {
                "key": dual.batch_key("beam-100K-5", sessions),
                "namespace": "beam-100K-5",
                "requests": 2,
                "dropped_bad_index": 0,
                "memories": [dual.to_record(memory())],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class Inner:
        version = "two-stage-p10-v2"
        model = "gemini-3.1-flash-lite"

        def prompt_texts(self):
            return ("a", "b")

    cached = dual.CachedExtractor(Inner(), cache)
    cached.user_id = "beam-100K-5"
    outcome = cached.extract(sessions)
    assert outcome.requests == 0
    assert [m.content for m in outcome.memories] == ["The user owns 25 postcards."]


def test_a_missing_batch_is_refused_rather_than_quietly_extracted(tmp_path):
    """Falling through to the real extractor would spend the quota this pass exists to
    avoid, and would do it silently."""
    cache = tmp_path / "cache.jsonl"
    cache.write_text("", encoding="utf-8")

    class Inner:
        version = "v"
        model = "m"

        def prompt_texts(self):
            return ()

    cached = dual.CachedExtractor(Inner(), cache)
    with pytest.raises(KeyError, match="not in the recorded pass"):
        cached.extract([Session("s00c000")])


@pytest.mark.parametrize("wrapper", ("RecordingExtractor", "CachedExtractor"))
def test_a_wrapper_fingerprints_as_the_extractor_it_wraps(tmp_path, wrapper):
    """A wrapper that fingerprinted as itself would make the store claim it was written by
    something that does not exist — the defect the ingest fingerprint exists to catch."""
    from llm_long_term_memory.ingest import fingerprint

    class Inner:
        version = "two-stage-p10-v2"
        model = "gemini-3.1-flash-lite"
        user_id = "user"

        def prompt_texts(self):
            return ("system", "prompt")

    cache = tmp_path / "cache.jsonl"
    cache.write_text("", encoding="utf-8")
    inner = Inner()
    wrapped = getattr(dual, wrapper)(inner, cache)
    assert fingerprint.from_runtime(wrapped, sessions_per_request=15).as_dict() == (
        fingerprint.from_runtime(inner, sessions_per_request=15).as_dict()
    )


def test_execute_refuses_without_the_registration(monkeypatch, capsys):
    monkeypatch.delenv("BEAM_PREREG", raising=False)
    monkeypatch.setattr(
        sys, "argv", ["dual_policy_ingest.py", "--questions", "m.json", "--execute"]
    )
    assert dual.main() == 2
    assert "BEAM_PREREG" in capsys.readouterr().out


def test_a_quota_stop_is_caught_rather_than_crashing_the_run():
    """`typer.Exit` derives from click's Exit, not from SystemExit. Catching SystemExit
    caught nothing: the tool died with a traceback 500 requests into a paid run instead of
    recording the state and saying "resume tomorrow". Found by drilling the stop against a
    fake provider."""
    import typer

    assert typer.Exit in dual._QUOTA_STOP
    assert not issubclass(typer.Exit, SystemExit)


def test_a_partial_cache_is_never_replayed_as_if_it_were_whole():
    """A cache file exists as soon as the first batch lands. Treating that as a finished
    extraction pass would build a second store missing whatever the quota stop interrupted,
    and nothing downstream would notice."""
    source = (REPO / "tools/dual_policy_ingest.py").read_text(encoding="utf-8")
    assert 'state.get("recording_complete")' in source
    assert "fresh=not started" in source


def test_a_transient_failure_is_not_reported_as_a_spent_day(tmp_path):
    """The pipeline returns the same incomplete outcome whether the day's budget ran out or
    the provider answered 503, and the two call for opposite responses. The first real run
    stopped on a 503 at 96 of 500 requests and was reported as a quota stop — which would
    have cost a day of doing nothing."""
    from types import SimpleNamespace

    quota = tmp_path / "quota"
    quota.mkdir()
    (quota / "observed-limits.json").write_text(
        json.dumps({"gemini-3.1-flash-lite": {"rpd": 500}}), encoding="utf-8"
    )
    settings = SimpleNamespace(store_dir=tmp_path)

    (quota / "quota-gemini-3.1-flash-lite.json").write_text(
        json.dumps({"day": "2026-09-15", "count": 96}), encoding="utf-8"
    )
    transient = dual.stop_reason(settings)
    assert transient["daily_budget_exhausted"] is False
    assert "Rerun now" in transient["advice"]

    (quota / "quota-gemini-3.1-flash-lite.json").write_text(
        json.dumps({"day": "2026-09-15", "count": 500}), encoding="utf-8"
    )
    spent = dual.stop_reason(settings)
    assert spent["daily_budget_exhausted"] is True
    assert "tomorrow" in spent["advice"]


def test_a_spent_day_is_not_retried(monkeypatch, tmp_path, capsys):
    """A transient 5xx is worth resuming through; the daily cap is not an obstacle to work
    around. Retrying past it is exactly what the cap exists to prevent."""

    calls = []
    monkeypatch.setattr(dual, "_run", lambda args: (calls.append(1), 2)[1])
    monkeypatch.setattr(dual, "stop_reason", lambda settings: {"daily_budget_exhausted": True})
    monkeypatch.setattr(dual.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["dual_policy_ingest.py", "--questions", "m.json", "--execute", "--retry-transient", "5"],
    )
    monkeypatch.setenv("BEAM_PREREG", dual.PREREG)
    assert dual.main() == 2
    assert len(calls) == 1
    assert "not retrying" in capsys.readouterr().out


def test_a_transient_stop_is_retried_up_to_the_bound(monkeypatch):
    calls = []
    monkeypatch.setattr(dual, "_run", lambda args: (calls.append(1), 2)[1])
    monkeypatch.setattr(dual, "stop_reason", lambda settings: {"daily_budget_exhausted": False})
    monkeypatch.setattr(dual.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["dual_policy_ingest.py", "--questions", "m.json", "--execute", "--retry-transient", "3"],
    )
    monkeypatch.setenv("BEAM_PREREG", dual.PREREG)
    assert dual.main() == 2
    assert len(calls) == 4  # the first try plus three retries
