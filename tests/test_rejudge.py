"""The judge-only pass, and the three ways it could quietly stop measuring the judge."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

rejudge = pytest.importorskip("rejudge")


def record(answer_id, pass_label, score, correct, ability="information_extraction"):
    return {
        "answer_id": answer_id,
        "question_id": answer_id.split("@")[0],
        "question_type": ability,
        "pass": pass_label,
        "score": score,
        "correct": correct,
    }


def write(tmp_path, name, records):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_the_answer_not_the_question_is_the_identity():
    """The same question answered twice must not look like one answer graded twice — that
    would fold answerer variance into a measurement of the judge."""
    base = {"question_id": "beam-100K-1-information_extraction-0"}
    first = rejudge.answer_id({**base, "hypothesis": "twenty-five postcards"})
    second = rejudge.answer_id({**base, "hypothesis": "25 postcards"})
    assert first != second
    assert first.startswith(base["question_id"])


def test_a_pass_may_not_overwrite_the_answers_it_reads(tmp_path, capsys, monkeypatch):
    rows = write(tmp_path, "answers.jsonl", [])
    monkeypatch.setattr(
        sys, "argv", ["rejudge.py", "--rows", str(rows), "--pass-label", "b1", "--out", str(rows)]
    )
    assert rejudge.main() == 2
    assert "may not overwrite" in capsys.readouterr().out


def test_execute_refuses_without_the_registration(tmp_path, capsys, monkeypatch):
    """Quota is spent only against a named pre-registration."""
    monkeypatch.delenv("BEAM_PREREG", raising=False)
    rows = write(tmp_path, "answers.jsonl", [])
    monkeypatch.setattr(
        sys, "argv", ["rejudge.py", "--rows", str(rows), "--pass-label", "b1", "--execute"]
    )
    assert rejudge.main() == 2
    assert "BEAM_PREREG" in capsys.readouterr().out


def test_disagreement_is_counted_on_the_verdict_and_the_score(tmp_path):
    left = write(
        tmp_path,
        "b1.jsonl",
        [record("q1@a", "b1", 1.0, True), record("q2@b", "b1", 0.5, True)],
    )
    right = write(
        tmp_path,
        "b2.jsonl",
        [record("q1@a", "b2", 1.0, True), record("q2@b", "b2", 0.0, False)],
    )
    summary = rejudge.summarise([left, right])
    assert summary["answers_compared"] == 2
    assert summary["verdict_discordance"] == pytest.approx(0.5)
    assert summary["mean_absolute_score_difference"] == pytest.approx(0.25)
    assert summary["by_ability"]["information_extraction"]["flips"] == 1


def test_an_off_rubric_reply_is_recorded_rather_than_averaged_away(tmp_path):
    """Repairing it would hide exactly the variance this pass exists to measure."""
    left = write(tmp_path, "b1.jsonl", [record("q1@a", "b1", 1.0, True)])
    right = write(
        tmp_path,
        "b2.jsonl",
        [record("q1@a", "b2", 1.0, True), {"answer_id": "q2@b", "pass": "b2", "refused": "off"}],
    )
    summary = rejudge.summarise([left, right])
    assert summary["answers_compared"] == 1


def test_one_pass_alone_cannot_report_a_disagreement(tmp_path):
    only = write(tmp_path, "b1.jsonl", [record("q1@a", "b1", 1.0, True)])
    assert "two passes" in rejudge.summarise([only])["note"]


def test_a_rehearsed_second_pass_does_not_grade_identically():
    """A dry run whose repeat is byte-identical measures a variance of exactly zero, which
    is the one answer that cannot be right."""
    import beam_dry_run

    rejudge._PASS = "b1"
    first, _ = rejudge.build_judge("m", execute=False)
    rejudge._PASS = "b2"
    second, _ = rejudge.build_judge("m", execute=False)
    assert isinstance(first.client, beam_dry_run.FakeProvider)
    assert (first.client.full_credit_at, first.client.half_credit_at) != (
        second.client.full_credit_at,
        second.client.half_credit_at,
    )
