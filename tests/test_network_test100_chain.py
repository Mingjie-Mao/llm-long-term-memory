import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import continue_dev_and_original_test100 as chain


def entry():
    return {"identity": "control:rep2:q", "reader_sha256": "saved", "failure_sha256": "error"}


def test_extra_budget_requires_old_allowances_and_survives_restarts():
    e = entry()
    with pytest.raises(ValueError, match="consumed first"):
        chain.extra_reserve([e], [], e)
    ledger = chain.extra_reserve([e, e], [], e)
    ledger = chain.extra_reserve([e, e], ledger, e)
    assert [r["attempt"] for r in ledger] == [1, 2]
    with pytest.raises(ValueError, match="exhausted"):
        chain.extra_reserve([e, e], ledger, e)


def test_saved_answer_binding_is_checked_in_both_ledgers():
    e = entry()
    changed = {**e, "reader_sha256": "changed"}
    with pytest.raises(ValueError, match="answer changed"):
        chain.extra_reserve([e, changed], [], e)
    with pytest.raises(ValueError, match="answer changed"):
        chain.extra_reserve([e, e], [changed], e)


def test_unavailable_network_cannot_consume_allowance(tmp_path, monkeypatch):
    monkeypatch.setattr(chain, "OLD_LEDGER", tmp_path / "old.json")
    monkeypatch.setattr(chain, "LEDGER", tmp_path / "extra.json")
    assert chain.charge(entry(), ready=False) is False
    assert list(tmp_path.iterdir()) == []


def test_original_and_extra_charges_use_separate_durable_ledgers(tmp_path, monkeypatch):
    old, extra = tmp_path / "old.json", tmp_path / "extra.json"
    monkeypatch.setattr(chain, "OLD_LEDGER", old)
    monkeypatch.setattr(chain, "LEDGER", extra)
    e = entry()
    assert chain.charge(e, ready=True)
    assert chain.charge(e, ready=True)
    before = old.read_bytes()
    assert chain.charge(e, ready=True)
    assert old.read_bytes() == before
    assert len(chain.read(extra)) == 1


def test_dns_probe_is_bounded_and_has_no_model_calls(monkeypatch):
    def timeout(cmd, **kwargs):
        assert kwargs["timeout"] == 15
        assert "socket.getaddrinfo" in cmd[-1]
        raise chain.subprocess.TimeoutExpired(cmd, 15)

    monkeypatch.setattr(chain.subprocess, "run", timeout)
    assert not chain.network_ready()
