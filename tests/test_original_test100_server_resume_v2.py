import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_long_term_memory.evaluation.recovery import RecoveringJudge, RecoveryExhausted
from llm_long_term_memory.llm.usage import UsageTracker

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from resume_original_test100_server_v2 import isolated_phase, phase_factory


def test_new_phase_preserves_old_state_and_cannot_renew_on_restart(tmp_path):
    old = tmp_path / "q.recovery-amendment-v1.json"
    old.write_text("immutable old recovery", encoding="utf-8")

    class ServerError(Exception):
        code = 500

    class Judge:
        prompt_version = "unchanged"
        calls = 0

        def grade(self, **kwargs):
            self.calls += 1
            raise ServerError("unavailable")

    judge = Judge()
    arguments = (judge, UsageTracker(), tmp_path / "usage.json", old, "identity")
    for _ in range(2):
        wrapped = phase_factory(RecoveringJudge, *arguments, max_recoveries=0, max_wait_seconds=0)
        with pytest.raises(RecoveryExhausted):
            wrapped.grade(hypothesis="exact original answer")
    assert judge.calls == 1
    assert old.read_text(encoding="utf-8") == "immutable old recovery"
    assert wrapped.state_path == tmp_path / "q.recovery-amendment-v2.json"
    with pytest.raises(ValueError, match="identity/policy"):
        phase_factory(RecoveringJudge, *arguments, max_recoveries=1, max_wait_seconds=0)


def test_foreign_recovery_cannot_enter_phase():
    with pytest.raises(ValueError, match="eligible"):
        phase_factory(lambda *a, **kw: None, None, None, None, Path("original.json"), "id")


def test_phase_restores_globals_and_original_factory_even_on_failure(tmp_path):
    profile = SimpleNamespace(
        STEM="original-test100",
        ANALYSIS=tmp_path,
        experiment=SimpleNamespace(instances=lambda: []),
    )
    amendment = SimpleNamespace(
        STEM="old",
        ANALYSIS="old",
        AMENDMENT="old",
        INVENTORY="old",
        LEDGER="old",
        instances_for=lambda split: [],
    )
    cli = SimpleNamespace(_build=lambda *a, **kw: None)
    recovery = SimpleNamespace(RecoveringJudge=RecoveringJudge)
    with pytest.raises(RuntimeError), isolated_phase(amendment, profile, cli, recovery):
        assert str(amendment.LEDGER).endswith(".judge-amendment-v2.json")
        assert recovery.RecoveringJudge is not RecoveringJudge
        raise RuntimeError("stop")
    assert amendment.LEDGER == "old"
    assert recovery.RecoveringJudge is RecoveringJudge
