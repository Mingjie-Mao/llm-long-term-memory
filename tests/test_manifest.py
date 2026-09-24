"""Frozen question sets, and the re-stratification trap they exist to close."""

from __future__ import annotations

import json

import pytest

from llm_long_term_memory.evaluation.datasets.longmemeval import Instance, stratify
from llm_long_term_memory.evaluation.manifest import Manifest, load_manifest, require_claim


def instance(qid: str, qtype: str) -> Instance:
    return Instance(
        question_id=qid,
        question=f"q {qid}?",
        question_type=qtype,
        question_date="2026-01-01",
        answer="a",
        sessions=[],
        answer_session_ids=[],
    )


def test_a_stratified_subset_is_not_a_dataset_order_prefix():
    """The defect that invalidated the first A2 pilot, pinned as a test.

    Stratified sampling *is* nested — the per-category allocation is monotone in
    `limit`, so `stratify(pool, 31)` is a genuine subset of `stratify(pool, 50)`.
    What it is not is **the first 31 questions in dataset order**, because the
    sample is drawn per category and then re-sorted by id.

    That distinction is what cost a run. The ingest processes namespaces in dataset
    order, so after a partial ingest the questions with data are a *prefix*. Asking
    for `--limit 31` returns a stratified subset scattered across all 50 positions,
    a third of which had no ingested data, retrieved nothing, and were scored wrong.
    """
    pool = [
        instance(f"q{i:03d}", ["temporal", "multi", "update", "user"][i % 4]) for i in range(200)
    ]

    big = [inst.question_id for inst in stratify(pool, 50, seed=0)]
    small = [inst.question_id for inst in stratify(pool, 31, seed=0)]

    assert set(small) <= set(big), "stratified subsets should nest"
    assert small != big[:31], "but a stratified subset is not a dataset-order prefix"


def test_manifest_round_trips(tmp_path):
    m = Manifest(
        name="dev50",
        variant="s",
        seed=0,
        question_ids=("a", "b", "c"),
        note="why this set exists",
    )
    path = m.save(tmp_path / "dev50.json")

    loaded = load_manifest(path)
    assert loaded == m
    assert len(loaded) == 3
    assert json.loads(path.read_text(encoding="utf-8"))["n"] == 3


def test_a_plain_id_list_is_accepted(tmp_path):
    """So an ad-hoc set can be promoted without being rewritten first."""
    path = tmp_path / "ids.txt"
    path.write_text("# a comment\nq1\n\n  q2  \n", encoding="utf-8")

    m = load_manifest(path)

    assert m.question_ids == ("q1", "q2")


def test_duplicate_ids_are_rejected(tmp_path):
    """A duplicated id would be evaluated once and counted twice."""
    path = tmp_path / "dupes.json"
    path.write_text(json.dumps({"question_ids": ["q1", "q2", "q1"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="more than once"):
        load_manifest(path)


def test_legacy_manifests_are_regression_evidence(tmp_path):
    path = tmp_path / "spent.json"
    path.write_text(json.dumps({"question_ids": ["q1"]}), encoding="utf-8")

    manifest = load_manifest(path)

    assert manifest.exposure == "regression"
    with pytest.raises(ValueError, match="cannot be reported as unseen"):
        require_claim(manifest, "unseen")


def test_an_explicit_unseen_manifest_allows_weaker_claims():
    manifest = Manifest("new", "s", 0, ("q",), exposure="unseen")

    require_claim(manifest, "unseen")
    require_claim(manifest, "development")
    require_claim(manifest, "regression")
