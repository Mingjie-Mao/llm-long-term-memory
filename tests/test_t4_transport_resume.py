import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from t4_transport_resume import reserve, transport_failure


def test_only_unsaved_judge_transport_errors_are_retryable():
    progress = {"status": "RUNNING"}
    call = {"role": "judge", "ok": False, "error": "ConnectError: DNS unavailable"}
    assert transport_failure(progress, [call])
    assert not transport_failure({"status": "QUOTA_WAIT"}, [call])
    assert not transport_failure(progress, [{**call, "role": "answerer"}])
    assert not transport_failure(progress, [{**call, "error": "ValueError: bad schema"}])
    assert not transport_failure(progress, [{**call, "error": "ClientError: 429"}])
    assert not transport_failure(progress, [{**call, "ok": True}])


def test_budget_survives_resumption_and_answer_change_is_refused():
    entry = {"identity": "candidate:rep1:q", "reader_sha256": "same saved answer"}
    ledger = reserve([], entry)
    with pytest.raises(ValueError, match="answer changed"):
        reserve(ledger, {**entry, "reader_sha256": "different answer"})
    ledger = reserve(ledger, entry)
    with pytest.raises(ValueError, match="budget exhausted"):
        reserve(ledger, entry)
