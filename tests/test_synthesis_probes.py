"""A probe set is only useful if its ground truth is right and it cannot be curated.

These probes exist to replace a two-to-four sample measurement, so a silent defect in
the generator would not show up as noise — it would show up as a confident number
nobody could question. The tests below check the two properties that make the
instrument trustworthy: the answers follow from the evidence, and the set reproduces
from its own seed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import date, datetime
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PROBES = REPO / "results/analysis/synthesis-probes.json"


def _load_tool():
    path = REPO / "tools" / "synthesis_probes.py"
    spec = importlib.util.spec_from_file_location("synthesis_probes_for_tests", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


@pytest.fixture(scope="module")
def probe_set():
    if not PROBES.is_file():
        pytest.skip("no probe set generated in this checkout")
    return json.loads(PROBES.read_text(encoding="utf-8"))


@pytest.fixture
def recorded_store(probe_set):
    """Optional artifact replay; clean CI does not carry the private research store."""
    import sqlite3

    path = REPO / "stores" / probe_set["store"]
    if not path.is_file():
        pytest.skip("recorded research store is not distributed with the repository")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        yield connection
    finally:
        connection.close()


def test_every_probe_is_well_formed(probe_set):
    required = {
        "probe_id",
        "kind",
        "namespace",
        "question",
        "answer",
        "answer_kind",
        "evidence_memory_ids",
        "derivation",
    }
    ids = set()
    for probe in probe_set["probes"]:
        assert required <= set(probe), f"{probe.get('probe_id')} is missing fields"
        assert probe["probe_id"] not in ids, "duplicate probe id"
        ids.add(probe["probe_id"])
        assert probe["evidence_memory_ids"], "a probe with no evidence cannot be answered"


def test_count_answers_equal_their_own_evidence(probe_set):
    """The property that makes a count probe checkable: the answer *is* the evidence."""
    counts = [p for p in probe_set["probes"] if p["kind"] == "count"]
    assert counts, "no count probes generated"
    for probe in counts:
        assert probe["answer"] == len(probe["evidence_memory_ids"])
        assert probe["answer"] >= 3, "below three, a miscount is not distinguishable"


def test_duration_and_comparison_answers_follow_from_their_dates(probe_set):
    """Re-derive each answer from the dates in the derivation string.

    Deliberately not by calling the generator again — that would only prove it agrees
    with itself. This reads the recorded derivation and redoes the arithmetic.
    """
    import re

    stamp = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
    day = re.compile(r"\d{4}-\d{2}-\d{2}(?!T)")
    for probe in probe_set["probes"]:
        if probe["kind"] == "duration":
            # Dates, not timestamps: the answerer is shown `%Y-%m-%d` and nothing
            # finer, so a gold it cannot reach is a gold that is wrong.
            assert not stamp.findall(probe["derivation"]), probe["probe_id"]
            later, earlier = (date.fromisoformat(s) for s in day.findall(probe["derivation"]))
            assert (later - earlier).days == probe["answer"]
            assert probe["answer"] > 0, "a zero or negative duration is not a question"
        if probe["kind"] == "comparison":
            assert probe["answer"] in {"A", "B"}
            a_time, b_time = (datetime.fromisoformat(s) for s in stamp.findall(probe["derivation"]))
            assert a_time != b_time, "an unordered pair has no first"
            assert probe["answer"] == ("A" if a_time < b_time else "B")


def test_current_state_probes_have_a_superseded_predecessor(probe_set):
    """Without a superseded value there is no temporal question to get wrong."""
    states = [p for p in probe_set["probes"] if p["kind"] == "current_state"]
    assert states, "no current_state probes generated"
    for probe in states:
        assert len(probe["evidence_memory_ids"]) >= 2
        assert "superseded value(s) precede it" in probe["derivation"]
        assert str(probe["answer"]).strip(), "the current value cannot be empty"


def test_the_set_reproduces_from_its_own_seed(probe_set):
    """Curation after seeing a result is the failure this makes detectable."""
    store = REPO / "stores" / probe_set["store"]
    if not store.is_file():
        pytest.skip("store not present in this checkout")
    rebuilt = tool.generate(
        store,
        REPO / "results/analysis/predicate-map.csv",
        probe_set["seed"],
        probe_set["per_kind"],
    )
    assert rebuilt == probe_set


def test_a_different_seed_draws_a_different_set(probe_set):
    """If the seed did nothing, `--verify` would pass on a curated set too."""
    store = REPO / "stores" / probe_set["store"]
    if not store.is_file():
        pytest.skip("store not present in this checkout")
    other = tool.generate(
        store,
        REPO / "results/analysis/predicate-map.csv",
        probe_set["seed"] + 1,
        probe_set["per_kind"],
    )
    assert other["probes"] != probe_set["probes"]


def test_the_set_says_it_is_not_a_benchmark(probe_set):
    """The caveat travels with the data, not only in a document somebody may not read."""
    assert "not accuracy" in probe_set["not_a_benchmark"]
    assert probe_set["store_fingerprint"]


# --- rules added after the first set was read against v3.3 -------------------


def test_no_probe_is_anchored_on_a_date_its_own_text_contradicts():
    """`event_time` sometimes holds the session date rather than the event date.
    Arithmetic over it is then right and the question unanswerable: the answerer
    reads March from the text and is graded against December."""
    row = {
        "content": 'The user watched the film "Parasite" on Netflix on March 17th, 2021.',
        "event_time": "2021-12-10T11:56:00",
    }
    assert tool._anchor_disagrees(row) is True
    agrees = {**row, "event_time": "2021-03-17T11:56:00"}
    assert tool._anchor_disagrees(agrees) is False
    # A memory that states no date contradicts nothing.
    assert (
        tool._anchor_disagrees({"content": "The user owns a bike.", "event_time": "2021-01-01"})
        is False
    )


def test_dated_probes_carry_no_contradicting_anchor(probe_set, recorded_store):
    by_id = {}
    con = recorded_store
    for mid, content, event_time in con.execute("SELECT id, content, event_time FROM memories"):
        by_id[mid] = {"content": content or "", "event_time": event_time or ""}
    for probe in probe_set["probes"]:
        if probe["kind"] not in ("duration", "comparison"):
            continue
        for mid in probe["evidence_memory_ids"]:
            assert not tool._anchor_disagrees(by_id[mid]), probe["probe_id"]


def test_every_count_question_states_the_scope_it_is_counted_over(probe_set):
    """The truth counts one mapped relation; the store reaches it through several
    raw predicates. Without the scope in the question the answerer cannot know
    which of them the question meant, and over-counts."""
    for probe in probe_set["probes"]:
        if probe["kind"] != "count":
            continue
        assert probe["scope_predicates"], probe["probe_id"]
        for label in probe["scope_predicates"]:
            assert label in probe["question"], probe["probe_id"]


def test_count_probes_ask_about_the_user_and_count_the_user(probe_set):
    """Every phrasing asks what *the user* did. Six of the first sixty probes
    counted facts whose subject was the assistant."""
    for probe in probe_set["probes"]:
        if probe["kind"] == "count":
            assert "subject='user'" in probe["derivation"], probe["probe_id"]


def test_a_confusable_relation_disqualifies_a_count_group():
    """No phrasing separates `owns` from `acquired` in "how many possessions", so
    such a namespace is dropped rather than phrased around."""
    assert "acquired" in tool._rivals("owns")
    assert "owns" in tool._rivals("acquired")
    assert tool._rivals("grows") == set()


def test_a_duration_gold_is_reachable_from_the_rendered_dates(probe_set, recorded_store):
    """`render_memory` formats every anchor as `%Y-%m-%d`, so the answerer never
    sees a time of day. Flooring a timestamp difference put 45% of the first set's
    golds one day below what a correct reader can compute; subtracting dates cannot."""
    con = recorded_store
    stamps = dict(con.execute("SELECT id, event_time FROM memories"))
    for probe in probe_set["probes"]:
        if probe["kind"] != "duration":
            continue
        days = sorted(date.fromisoformat(stamps[m][:10]) for m in probe["evidence_memory_ids"])
        assert (days[1] - days[0]).days == probe["answer"], probe["probe_id"]


def test_the_relation_map_still_hashes_to_what_the_probe_set_recorded(probe_set):
    """The map's raw bytes are the probe set's identity, and the split's identity in turn.

    `test_the_set_reproduces_from_its_own_seed` already covers this, but only where
    `train150.db` exists — which is nowhere in CI. This needs no store, so the check
    survives in a clean checkout.

    The failure it exists for is not an edit. `core.autocrlf` rewriting CRLF to LF on a
    stash or checkout round-trip changes the hash while leaving every row identical, so
    a diff shows nothing and the held-out split quietly stops being provable. See
    `.gitattributes`.
    """
    relation_map = REPO / "results/analysis/predicate-map.csv"
    if not relation_map.is_file():
        pytest.skip("no relation map in this checkout")
    digest = hashlib.sha256(relation_map.read_bytes()).hexdigest()

    assert digest == probe_set["relation_map_sha256"], (
        "predicate-map.csv no longer hashes to the value the probe set was built from. "
        "If every row looks unchanged, compare line endings before anything else."
    )
