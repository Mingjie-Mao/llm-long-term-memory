"""Influence labelling and the utility predictor."""

from __future__ import annotations

import numpy as np
import pytest

from chronomem.influence import (
    FEATURE_NAMES,
    Influence,
    InfluenceDataset,
    MemoryInfluence,
    build,
    fit_grouped,
    requests_needed,
)
from chronomem.influence.predictor import UtilityPredictor
from chronomem.store import Memory


def row(full: bool, without: bool, alone: bool | None = None) -> MemoryInfluence:
    return MemoryInfluence(
        "q1", "m1", full_correct=full, without_correct=without, alone_correct=alone
    )


def test_removing_it_breaks_a_correct_answer_is_critical():
    assert row(full=True, without=False).label is Influence.CRITICAL
    assert row(full=True, without=False).utility == 1.0


def test_removing_it_fixes_a_wrong_answer_is_harmful():
    """The stale-memory case, and the reason questions answered *wrong* must be
    ablated too. Restricting the measurement to correct answers cannot see it."""
    r = row(full=False, without=True)
    assert r.label is Influence.HARMFUL
    assert r.utility < 0, "signed, so a packer learns to exclude rather than rank last"


def test_redundant_is_distinguished_from_inert():
    """Leave-one-out alone cannot tell them apart: both survive removal. Only the
    leave-one-in probe separates 'its evidence is elsewhere' from 'it has none'."""
    redundant = row(full=True, without=True, alone=True)
    inert = row(full=True, without=True, alone=False)

    assert redundant.label is Influence.REDUNDANT
    assert inert.label is Influence.INERT
    assert redundant.utility > inert.utility


def test_missing_leave_one_in_falls_back_to_inert_not_redundant():
    """A half-measured row must not be credited with evidence it was never tested
    for."""
    assert row(full=True, without=True, alone=None).label is Influence.INERT


def test_load_bearing_rate_counts_both_directions():
    data = InfluenceDataset(
        rows=[
            row(True, False),  # critical
            row(False, True),  # harmful
            row(True, True, False),  # inert
            row(True, True, False),  # inert
        ]
    )
    assert data.load_bearing_rate == 0.5


def test_dataset_roundtrips(tmp_path):
    data = InfluenceDataset(rows=[row(True, False), row(False, True), row(True, True, True)])
    path = tmp_path / "influence.jsonl"
    data.save(path)

    loaded = InfluenceDataset.load(path)
    assert [r.label for r in loaded.rows] == [r.label for r in data.rows]


def test_cost_is_reported_before_it_is_spent():
    """50 questions at k=20 with both probes is 2,050 answers — four days of a
    500/day budget, which is a decision, not a detail."""
    assert requests_needed(50, 20, leave_one_in=True) == 50 * 41
    assert requests_needed(50, 20, leave_one_in=False) == 50 * 21


# ----------------------------------------------------------------- features


def mem(mid: str, content: str, **kw) -> Memory:
    kw.setdefault("token_count", max(1, len(content) // 4))
    return Memory(id=mid, user_id="u", type=kw.pop("type", "semantic"), content=content, **kw)


def test_every_feature_is_computable_without_an_llm_or_a_gold_answer():
    f = build(
        mem("m1", "The user owns 25 postcards since March 2023"),
        query="how many postcards?",
        retrieval_score=0.8,
        rank=0,
        neighbours=[],
    )
    assert len(f.as_list()) == len(FEATURE_NAMES)
    assert f.has_number == 1.0
    assert f.has_date_words == 1.0
    assert f.query_term_overlap > 0


def test_duplicate_rate_sees_a_near_copy():
    """The one feature aimed at redundancy — invisible to leave-one-out, and the
    thing a packer must not pay for twice."""
    target = mem("m1", "The user lives in Canberra with two cats")
    twin = mem("m2", "The user lives in Canberra and has two cats")
    unrelated = mem("m3", "The user prefers oat milk")

    with_twin = build(target, query="q", retrieval_score=0.5, rank=0, neighbours=[twin])
    alone = build(target, query="q", retrieval_score=0.5, rank=0, neighbours=[unrelated])
    assert with_twin.duplicate_content_rate > alone.duplicate_content_rate


# ---------------------------------------------------------------- predictor


def test_splitting_is_by_question_so_a_question_never_spans_train_and_test():
    """Twenty memories retrieved for one question share its query terms and its
    answer. Splitting by row leaks, and the score would report memorisation."""
    rng = np.random.default_rng(0)
    groups = [f"q{i // 10}" for i in range(100)]
    x = rng.normal(size=(100, len(FEATURE_NAMES)))
    y = x[:, 0] * 2.0 + rng.normal(scale=0.1, size=100)

    _, report = fit_grouped(x, y, groups, folds=5)
    assert report.n_test > 0
    assert report.beats_baseline, "a signal this strong must be learnable out of fold"


def test_a_model_with_nothing_to_learn_does_not_beat_the_mean():
    rng = np.random.default_rng(1)
    groups = [f"q{i // 10}" for i in range(100)]
    x = rng.normal(size=(100, len(FEATURE_NAMES)))
    y = rng.normal(size=100)  # unrelated to x

    _, report = fit_grouped(x, y, groups, folds=5)
    assert not report.beats_baseline


def test_correlation_with_relevance_is_reported():
    """If predicted utility is just the retrieval score, the experiment has answered
    itself in the negative and the number has to say so."""
    rng = np.random.default_rng(2)
    groups = [f"q{i // 10}" for i in range(100)]
    x = rng.normal(size=(100, len(FEATURE_NAMES)))
    y = x[:, 0]  # utility *is* relevance

    _, report = fit_grouped(x, y, groups, folds=5)
    assert report.spearman_vs_relevance > 0.9


def test_a_constant_feature_does_not_produce_nan():
    x = np.ones((20, len(FEATURE_NAMES)))
    x[:, 1] = np.arange(20)
    y = np.arange(20, dtype=float)
    model = UtilityPredictor()
    model.fit(x, y)
    assert not np.isnan(model.predict(x)).any()


def test_predictor_roundtrips(tmp_path):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(30, len(FEATURE_NAMES)))
    y = x[:, 0]
    model = UtilityPredictor()
    model.fit(x, y)
    path = tmp_path / "m.json"
    model.save(path)

    assert np.allclose(UtilityPredictor.load(path).predict(x), model.predict(x))


def test_fitting_before_predicting_is_an_error():
    with pytest.raises(RuntimeError, match="fit"):
        UtilityPredictor().predict(np.zeros((1, len(FEATURE_NAMES))))


# ------------------------------------------------------------ regressions


def test_a_memory_that_answers_alone_but_loses_in_context_is_not_called_helpful():
    """Named for the symptom. `full_correct=False` with `alone_correct=True` means
    the evidence was present and something else in the prompt overrode it — the
    strongest signal available that the other memories interfere. Filing that under
    'helpful' invited exactly the wrong reading."""
    r = row(full=False, without=False, alone=True)
    assert r.label is Influence.DROWNED
    assert r.utility > 0


def test_saving_an_unfitted_model_says_so():
    with pytest.raises(RuntimeError, match="fit"):
        UtilityPredictor().save("/tmp/chronomem-never-fitted.json")


def test_loading_a_model_with_a_stale_feature_layout_is_refused(tmp_path):
    """Weights are positional. A feature added or reordered since fitting produces
    no shape error — every utility is computed against the wrong columns, and the
    packer, which only consumes the ranking, shows no symptom."""
    import json

    rng = np.random.default_rng(4)
    x = rng.normal(size=(20, len(FEATURE_NAMES)))
    model = UtilityPredictor()
    model.fit(x, x[:, 0])
    path = tmp_path / "m.json"
    model.save(path)

    d = json.loads(path.read_text())
    d["features"] = [*d["features"][:-1], "a_feature_that_no_longer_exists"]
    path.write_text(json.dumps(d))

    with pytest.raises(ValueError, match="Refit"):
        UtilityPredictor.load(path)


def test_a_feature_matrix_of_the_wrong_width_is_refused():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(20, len(FEATURE_NAMES) - 1))
    with pytest.raises(ValueError, match="columns"):
        fit_grouped(x, rng.normal(size=20), [f"q{i}" for i in range(20)])
