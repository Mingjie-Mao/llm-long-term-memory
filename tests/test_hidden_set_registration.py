"""A final set that is not unseen must be refused, and the refusals must not be bypassable.

Written as the ways a "new" corpus is actually contaminated rather than as the ways
someone might cheat: nobody plans to reuse questions, they arrive renumbered, reworded, or
duplicated inside the candidate itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

register = pytest.importorskip("register_hidden_set")

SEEN_IDS = {"old-1", "old-2"}
SEEN_TEXTS = {
    "where did the user go on holiday last summer": "data/longmemeval_s_cleaned.json",
    "how many books has the user finished this year": "data/longmemeval_s_cleaned.json",
}


def row(qid: str, question: str) -> dict:
    return {"question_id": qid, "question": question}


def test_a_genuinely_new_set_passes():
    candidate = [
        row("new-1", "Which bicycle did the user service in March"),
        row("new-2", "What temperature does the user set the thermostat to"),
    ]
    assert register.check(candidate, SEEN_IDS, SEEN_TEXTS) == []


def test_a_reused_question_id_is_refused():
    problems = register.check([row("old-1", "something entirely new")], SEEN_IDS, SEEN_TEXTS)

    assert any("already in a registered set" in p for p in problems)


def test_new_ids_over_old_text_are_refused():
    """The failure an id check alone cannot see, and what the oracle haystack would have
    looked like had it been renumbered."""
    problems = register.check(
        [row("new-1", "Where did the user go on holiday last summer?")], SEEN_IDS, SEEN_TEXTS
    )

    assert any("verbatim repeats" in p for p in problems)


def test_a_paraphrase_is_refused_too():
    """A paraphrase is contamination for a memory benchmark in the same way a copy is."""
    problems = register.check(
        [row("new-1", "How many books has the user finished this year, roughly")],
        SEEN_IDS,
        SEEN_TEXTS,
    )

    assert any("near-duplicates" in p for p in problems)


def test_a_candidate_that_repeats_itself_is_refused():
    """n is the denominator of the reported result. A set that is 100 rows and 80
    questions reports a number over a sample it does not have."""
    problems = register.check(
        [row("new-1", "a question"), row("new-1", "a question")], SEEN_IDS, SEEN_TEXTS
    )

    assert any("repeated within the candidate" in p for p in problems)


def test_a_row_without_a_question_id_is_refused():
    problems = register.check([{"question": "a question"}], SEEN_IDS, SEEN_TEXTS)

    assert any("no question_id" in p for p in problems)


def test_every_registered_set_is_checked_against():
    """Dropping one from this tuple would silently let its questions through."""
    assert set(register.REGISTERED) == {
        "dev50",
        "heldout100",
        "train150",
        "dev100",
        "test100",
    }


def test_the_real_corpus_is_refused_as_its_own_final_set():
    """The end-to-end check: LongMemEval-S cannot be registered as a hidden set, because
    every one of its questions has been read."""
    import json

    source = REPO / "data/longmemeval_s_cleaned.json"
    if not source.is_file():
        pytest.skip("the dataset is downloaded on demand and not committed")
    ids, texts = register.seen_questions()
    candidate = json.loads(source.read_text(encoding="utf-8"))[:5]

    problems = register.check(candidate, ids, texts)

    assert len(problems) >= 3, "id, verbatim and near-duplicate checks should all fire"


@pytest.mark.parametrize(
    "candidate",
    [
        [],
        [None],
        [{}],
        [{"question_id": []}],
        [row(" ", "a question")],
        [row("new", "")],
        [row("new", None)],
        [row("new", "纯中文")],
    ],
)
def test_malformed_or_uncheckable_candidate_is_refused(candidate):
    assert register.check(candidate, SEEN_IDS, SEEN_TEXTS)


def test_missing_comparison_manifest_refuses_registration(tmp_path, monkeypatch):
    monkeypatch.setattr(register, "MANIFESTS", tmp_path)
    with pytest.raises(ValueError, match="missing comparison manifest"):
        register.seen_questions()


def test_missing_comparison_text_cannot_silently_skip_overlap_checks(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(register, "REPO", tmp_path)
    monkeypatch.setattr(register, "MANIFESTS", tmp_path)
    for name in register.REGISTERED:
        (tmp_path / f"{name}.json").write_text(
            json.dumps({"question_ids": ["old"]}), encoding="utf-8"
        )
    with pytest.raises(ValueError, match="comparison corpus is incomplete"):
        register.seen_questions()


def test_invalid_name_cannot_escape_manifest_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(register, "MANIFESTS", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "register_hidden_set",
            "absent.json",
            "--name",
            "../outside",
            "--source",
            "fixture",
            "--licence",
            "fixture",
        ],
    )
    assert register.main() == 2
    assert not (tmp_path.parent / "outside.json").exists()
