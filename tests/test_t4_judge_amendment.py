import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_long_term_memory import cli
from llm_long_term_memory.evaluation.judge import JudgeResult
from llm_long_term_memory.llm.usage import UsageTracker

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import t4_judge_amendment as tool


def state(code=500):
    return {
        "failures": 4,
        "max_recoveries": 3,
        "events": [{"code": code, "error_type": "ServerError", "exhausted": True}],
    }


def test_only_transient_exhaustion_qualifies():
    assert tool.eligible(state(503))
    assert not tool.eligible(state(400))
    assert not tool.eligible(state(429))
    s = state()
    s["failures"] = 3
    assert not tool.eligible(s)


def test_missing_or_foreign_saved_rows_cannot_receive_extension():
    instances = [SimpleNamespace(question_id="a"), SimpleNamespace(question_id="b")]
    progress = {"status": "RECOVERY_EXHAUSTED", "arm": "candidate", "rep": 3}
    assert tool.target(progress, instances, {"a": {}, "b": {}}, {"a": {}})[2] == instances[1]
    with pytest.raises(ValueError, match="complete saved"):
        tool.target(progress, instances, {"a": {}}, {"a": {}})
    with pytest.raises(ValueError, match="complete saved"):
        tool.target(progress, instances, {"a": {}, "b": {}}, {"a": {}, "foreign": {}})
    with pytest.raises(ValueError, match="foreign arm"):
        tool.target({**progress, "arm": "foreign"}, instances, {}, {})


def test_extension_grades_exact_saved_answer_keeps_false_and_original_recovery(
    tmp_path, monkeypatch
):
    analysis, raw = tmp_path / "results/analysis", tmp_path / "results/raw"
    analysis.mkdir(parents=True)
    raw.mkdir(parents=True)
    monkeypatch.setattr(tool, "REPO", tmp_path)
    monkeypatch.setattr(tool, "ANALYSIS", analysis)
    monkeypatch.setattr(tool, "INVENTORY", analysis / "amendment-inventory.json")
    monkeypatch.setattr(tool, "LEDGER", analysis / "ledger.json")
    tool.INVENTORY.write_text("{}", encoding="utf-8")
    identity = "original-identity"
    monkeypatch.setattr(
        tool, "freeze", lambda: ({"arms": {"control": "original-prompt"}}, identity)
    )
    instance = SimpleNamespace(
        question_id="q",
        question="Where?",
        answer="gold",
        is_abstention=False,
        question_type="lookup",
    )
    monkeypatch.setattr(tool, "instances_for", lambda split: [instance])
    (analysis / f"{tool.STEM}.progress.json").write_text(
        json.dumps({"status": "RECOVERY_EXHAUSTED", "arm": "control", "rep": 1}),
        encoding="utf-8",
    )
    base = raw / f"{tool.STEM}-control-rep1"
    row = {
        "question_id": "q",
        "cohort": "control",
        "inventory_sha256": identity,
        "answer_prompt_version": "original-prompt",
        "answer": {
            "text": "original wrong answer",
            "context_tokens": 10,
            "prompt_tokens": 10,
            "output_tokens": 3,
            "latency_ms": 1,
            "notes": {
                "execution_identity": {
                    "inventory_sha256": identity,
                    "arm": "control",
                    "answerer": "gemini-3.5-flash-lite",
                }
            },
        },
    }
    rp, gp, up = [
        base.with_suffix("." + s) for s in ("readers.jsonl", "grades.jsonl", "usage.json")
    ]
    rp.write_text(json.dumps(row) + "\n", encoding="utf-8")
    gp.write_text("", encoding="utf-8")
    UsageTracker().save(up)
    original = base.with_suffix(".q.recovery.json")
    original.write_text(
        json.dumps(
            {**state(), "identity": hashlib.sha256(f"{identity}:control:1:q".encode()).hexdigest()}
        ),
        encoding="utf-8",
    )
    before = original.read_bytes(), rp.read_bytes()

    class Judge:
        prompt_version = "unchanged-judge"
        client = SimpleNamespace()

        def __init__(self):
            self.calls = []

        def grade(self, **kwargs):
            self.calls.append(kwargs)
            return JudgeResult(False, "original false verdict", 3, 2)

    judge = Judge()
    monkeypatch.setattr(
        cli,
        "_build",
        lambda *a, **kw: (
            None,
            None,
            SimpleNamespace(store=SimpleNamespace(close=lambda: None)),
            judge,
            UsageTracker(),
        ),
    )
    monkeypatch.setattr(sys, "argv", ["t4_judge_amendment.py", "--execute"])
    assert tool.main() == 0
    assert len(judge.calls) == 1
    assert judge.calls[0]["hypothesis"] == "original wrong answer"
    assert json.loads(gp.read_text(encoding="utf-8"))["verdict"]["correct"] is False
    assert before == (original.read_bytes(), rp.read_bytes())
    assert json.loads(tool.LEDGER.read_text(encoding="utf-8"))[0]["outcome"] == "GRADED"
    # A completed identity has no missing grade and cannot be rerun for a better verdict.
    with pytest.raises(ValueError, match="missing grade"):
        tool.main()
    assert len(judge.calls) == 1
