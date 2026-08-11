"""Dataset subsetting.

The bug these guard against was found by looking at a finished 50-question run and
noticing every row said `single-session-user`: the file is grouped by question type,
so a `[:50]` dev subset silently excludes temporal reasoning, knowledge updates and
abstention — the categories the whole project is about.
"""

from __future__ import annotations

from chronomem.evaluation.datasets.longmemeval import Instance, stratify


def make(qid: str, qtype: str) -> Instance:
    return Instance(
        question_id=qid,
        question_type=qtype,
        question="q",
        answer="a",
        question_date="2026-01-01",
        sessions=[],
        answer_session_ids=[],
    )


# Same shape as the real file: grouped by type, wildly uneven sizes.
CORPUS = (
    [make(f"ms{i}", "multi-session") for i in range(133)]
    + [make(f"tr{i}", "temporal-reasoning") for i in range(133)]
    + [make(f"ku{i}", "knowledge-update") for i in range(78)]
    + [make(f"su{i}", "single-session-user") for i in range(70)]
    + [make(f"sa{i}", "single-session-assistant") for i in range(56)]
    + [make(f"sp{i}", "single-session-preference") for i in range(30)]
)


def types_of(instances):
    counts: dict[str, int] = {}
    for i in instances:
        counts[i.question_type] = counts.get(i.question_type, 0) + 1
    return counts


def test_naive_head_slice_is_single_type():
    """Documents why stratify exists."""
    assert set(types_of(CORPUS[:50])) == {"multi-session"}


def test_every_type_survives_a_50_question_subset():
    picked = stratify(CORPUS, 50)
    assert len(picked) == 50
    assert set(types_of(picked)) == set(types_of(CORPUS))


def test_proportions_track_the_full_corpus():
    picked = types_of(stratify(CORPUS, 100))
    full = types_of(CORPUS)
    for qtype, n in full.items():
        expected = 100 * n / len(CORPUS)
        assert abs(picked[qtype] - expected) <= 1, qtype


def test_smallest_category_is_not_rounded_away():
    """single-session-preference is 6% of the corpus; naive int() truncation drops
    it entirely at small subset sizes."""
    picked = stratify(CORPUS, 20)
    assert picked and "single-session-preference" in types_of(picked)


def test_same_seed_gives_the_same_questions():
    """Two variants must be scored on identical questions or the comparison is
    meaningless."""
    a = [i.question_id for i in stratify(CORPUS, 40, seed=7)]
    b = [i.question_id for i in stratify(CORPUS, 40, seed=7)]
    assert a == b


def test_different_seeds_give_different_questions():
    a = {i.question_id for i in stratify(CORPUS, 40, seed=1)}
    b = {i.question_id for i in stratify(CORPUS, 40, seed=2)}
    assert a != b


def test_limit_at_or_above_corpus_returns_everything():
    assert len(stratify(CORPUS, len(CORPUS))) == len(CORPUS)
    assert len(stratify(CORPUS, 10_000)) == len(CORPUS)


def test_order_is_stable_across_runs():
    picked = stratify(CORPUS, 30)
    assert [i.question_id for i in picked] == sorted(i.question_id for i in picked)
