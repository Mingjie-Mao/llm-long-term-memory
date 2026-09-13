"""The grader must be readable without running the model.

Every case here is a hand-written answer string, so a failure points at the
grading rule rather than at a provider. The probe runner's value depends entirely
on these rules: a grader that silently scores an unparseable reply as wrong turns
a formatting difference into an accuracy claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from run_synthesis_probes import grade

COUNT = {"kind": "count", "answer": 3}
DURATION = {"kind": "duration", "answer": 21}
COMPARISON = {
    "kind": "comparison",
    "answer": "B",
    "question": (
        "Which of these happened first? "
        "A: The user purchased a Chanel handbag for $5,000. "
        "B: The user decided to use a hybrid approach for their content calendar."
    ),
}
STATE = {"kind": "current_state", "answer": "software engineering manager"}


def test_a_digit_and_a_written_number_grade_the_same():
    assert grade(COUNT, "You have 3 of them.", "answer", []).verdict == "correct"
    assert grade(COUNT, "Three items are on record.", "answer", []).verdict == "correct"
    assert grade(COUNT, "You have 2 of them.", "answer", []).verdict == "wrong"


def test_a_duration_prefers_the_number_attached_to_the_unit():
    # "about 3 weeks (21 days)" must not be read as 3: the unit decides.
    g = grade(DURATION, "About 3 weeks — 21 days between them.", "answer", [])
    assert g.verdict == "correct" and g.parsed == "21"


def test_abstention_is_its_own_verdict_not_a_wrong_answer():
    """Counting a refusal as wrong is what hid the abstention bucket in the first
    place; it must be separable from a wrong number."""
    g = grade(COUNT, "I do not know.", "no_evidence", [])
    assert g.verdict == "abstained"
    assert grade(COUNT, "I do not know.", "answer", []).verdict == "abstained"


def test_a_reply_with_no_number_at_all_is_unparseable_not_wrong():
    g = grade(COUNT, "That depends on how you group them.", "answer", [])
    assert g.verdict == "unparseable"


def test_a_comparison_reads_an_explicit_label():
    assert grade(COMPARISON, "B: the calendar came first.", "answer", []).verdict == "correct"
    assert grade(COMPARISON, "A: the handbag came first.", "answer", []).verdict == "wrong"


def test_current_state_separates_a_stale_answer_from_a_merely_wrong_one():
    """The failure this probe exists to catch is answering with the superseded
    value, so it is labelled rather than folded into `wrong`."""
    stale = ["Digital Marketing Specialist"]
    good = grade(STATE, "You are a software engineering manager.", "answer", stale)
    assert good.parsed == "active"
    g = grade(STATE, "You are a Digital Marketing Specialist.", "answer", stale)
    assert g.verdict == "wrong" and g.parsed == "superseded"
    assert grade(STATE, "You are a chef.", "answer", stale).parsed == "other"


def test_flags_are_derived_from_the_store_not_from_the_answer(tmp_path):
    """`anchor_conflict` must be computable before any model runs, or it becomes a
    post-hoc excuse for rows that scored badly."""
    from run_synthesis_probes import build_flags

    from llm_long_term_memory.store import Memory, SQLiteMemoryStore

    store_path = tmp_path / "flags.db"
    store = SQLiteMemoryStore(store_path)
    store.initialize()
    from datetime import datetime

    store.add_memories(
        [
            Memory(
                id="conflicting-date",
                user_id="alice",
                type="episodic",
                token_count=10,
                content="The user watched a film in March 2021.",
                event_time=datetime(2021, 12, 10),
            )
        ]
    )
    store.close()

    probe = {
        "probe_id": "x",
        "kind": "duration",
        "evidence_memory_ids": ["conflicting-date"],
    }
    flags = build_flags([probe], store_path)
    assert flags["x"]["anchor_conflict"] is True


# --- cases taken verbatim from the v3.3 run, which the first grader misread ---

CMP_Q = (
    "Which of these happened first? "
    "A: The user purchased a Chanel handbag for $5,000. "
    "B: The user decided to use a hybrid approach for their content calendar, "
    "with a master sheet and separate sheets for each platform."
)


def test_a_prose_comparison_is_graded_on_the_event_not_a_stray_letter():
    """The answerer names the events instead of replying `A`/`B`. Grading on a
    letter token scored 36 of 60 real replies unparseable."""
    probe = {"kind": "comparison", "answer": "B", "question": CMP_Q}
    text = (
        "The user decided to use a hybrid approach for their content calendar "
        "(July 21, 2023) happened before the user purchased a Chanel handbag "
        "for $5,000 (August 3, 2023)."
    )
    g = grade(probe, text, "answer", [])
    assert g.verdict == "correct" and g.parsed == "B"


def test_an_explicit_label_still_wins_when_the_reply_opens_with_one():
    probe = {"kind": "comparison", "answer": "B", "question": CMP_Q}
    g = grade(probe, "A: the handbag happened first.", "answer", [])
    assert g.verdict == "wrong" and g.parsed == "A"


def test_a_count_reads_the_stated_total_not_the_first_number_it_narrates():
    """A real reply mentioned "$200 spent last month" before its conclusion; the
    first grader returned 200."""
    probe = {"kind": "count", "answer": 3}
    text = (
        "Evidence summary:\n- Mentions lunch at The Apple Pan.\n"
        "- Mentions dining out generally ($200 spent last month).\n"
        "Conclusion: there are 3 distinct places."
    )
    assert grade(probe, text, "answer", []).parsed == "3"


def test_a_duration_reads_the_subtraction_not_the_dates_above_it():
    probe = {"kind": "duration", "answer": 8}
    text = "2023-05-29 minus 2023-05-21 = 8 days.\n\n8 days passed between the two events."
    assert grade(probe, text, "answer", []).verdict == "correct"


def test_a_zero_from_a_reply_that_reported_no_evidence_is_a_refusal():
    """`Count: 0` beside "no distinct ... are detailed" is an abstention wearing a
    number, and counting it as a wrong answer hides the abstention bucket again."""
    probe = {"kind": "count", "answer": 3}
    text = "No distinct completed music tracks or podcasts are detailed.\nCount: 0"
    assert grade(probe, text, "no_evidence", []).verdict == "abstained"


def test_resume_refuses_a_file_answered_for_a_different_probe_set(tmp_path, monkeypatch):
    """Probe ids are positional, so a regenerated set reuses `count_0000` for a
    different question. Resuming on the id alone would splice two incompatible sets
    into one file and report a number for it."""
    import json
    import subprocess
    import sys as _sys
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    raw = tmp_path / "raw"
    raw.mkdir()
    spec = json.loads((repo / "results/analysis/synthesis-probes.json").read_text(encoding="utf-8"))
    victim = spec["probes"][0]
    (raw / "probes.stale.jsonl").write_text(
        json.dumps(
            {
                "probe_id": victim["probe_id"],
                "kind": victim["kind"],
                "question": victim["question"] + " (from an older set)",
                "text": "3",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            _sys.executable,
            str(repo / "tools/run_synthesis_probes.py"),
            "--label",
            "stale",
            "--out",
            str(raw),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=repo,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "different probe set" in result.stderr
