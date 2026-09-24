"""The batch-size curve has to survive the cache it was measured in.

Each arm cost real requests against a daily quota, and the CLI keeps its extractions
under `stores/`, which is ignored. One `rm -rf stores/` and the largest effect this
project measured would exist only as a table in a document — unfalsifiable, and
indistinguishable from a number someone typed.

So the archive is the artifact and these tests are what make it one: the recorded
curve must be *derived* from the archived extractions rather than asserted beside
them, and the derivation must still hold. Re-scoring needs the corpus, which is not
committed, so that half skips where the corpus is absent — but the half that checks
the archive against its own recorded counts runs everywhere, including CI.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CURVE = REPO / "results/analysis/extraction-batch-size.json"
ARCHIVE = REPO / "results/raw/extraction-batch-size.extractions.json"


def _tool():
    path = REPO / "tools" / "batch_size_curve.py"
    spec = importlib.util.spec_from_file_location("batch_size_curve_for_tests", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def curve() -> dict:
    return json.loads(CURVE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def archive() -> dict:
    return json.loads(ARCHIVE.read_text(encoding="utf-8"))


def test_every_arm_scored_the_same_sixty_sessions(archive):
    """One cohort, or the arms are not paired and the curve is six unrelated runs."""
    assert len(archive["session_ids"]) == archive["sessions"] == 60
    assert len(set(archive["session_ids"])) == 60
    for arm, held in archive["arms"].items():
        assert set(held["extractions"]) <= set(archive["session_ids"]), arm
        # A session absent from an arm is one that yielded nothing, which is a
        # result; a session the arm never saw would be a hole in the pairing.
        assert len(held["extractions"]) == 60, arm


def test_the_recorded_memory_counts_come_from_the_archive(curve, archive):
    """The cheap half of reproduction: counting does not need the source text."""
    for row in curve["arms"]:
        held = archive["arms"][row["arm"]]
        counted = sum(len(v) for v in held["extractions"].values())
        assert counted == row["memories"], row["arm"]
        assert counted / 60 == pytest.approx(row["memories_per_session"])


def test_the_request_counts_are_what_the_batching_implies(curve):
    """Cost is half the claim. Two requests per chunk, one per session when grounded."""
    for row in curve["arms"]:
        batch = row["sessions_per_request"]
        expected = 60 if row["grounded"] else 2 * -(-60 // batch)
        assert row["requests"] == expected, row["arm"]


def test_the_curve_is_monotone_in_batch_size_and_names_its_knee(curve):
    """The finding, stated as a property rather than as prose that can drift.

    Fidelity rises as the batch shrinks, and the single largest step is 15 → 8 —
    which is the whole reason 8 is the candidate rather than 1.
    """
    batched = sorted(
        (r for r in curve["arms"] if not r["grounded"]),
        key=lambda r: -r["sessions_per_request"],
    )
    overall = [r["overall"] for r in batched]
    assert overall == sorted(overall), "smaller batches should not score worse"
    steps = [b - a for a, b in itertools.pairwise(overall)]
    assert steps[0] == max(steps), "the 15 → 8 step should be the largest"
    assert batched[0]["sessions_per_request"] == 15
    assert batched[1]["sessions_per_request"] == 8
    assert batched[1]["overall"] - batched[0]["overall"] > 0.25


def test_the_knee_is_where_the_paired_test_stops_resolving(curve):
    """Why 8 and not 5, without appealing to the shape of a line.

    15 → 8 moves 41 specifics and loses 4. The very next step, 8 → 5, is +8/-6 —
    indistinguishable from noise on the same 134 items. An operating point chosen
    from a curve's visible bend is chosen from noise; this one is chosen from where
    the evidence runs out.
    """
    by_step = {(c["from"], c["to"]): c for c in curve["pairwise"]}
    first = by_step[("batch15", "batch8")]
    assert first["gained"] > 10 * first["lost"]
    assert first["p"] < 1e-6
    assert by_step[("batch8", "batch5")]["p"] > 0.05
    # And the whole move, end to end, costs nothing that it does not replace.
    assert by_step[("batch15", "batch1")]["lost"] == 0


def test_the_pairwise_counts_are_a_test_and_not_a_decoration(curve):
    """Recomputes each p from its own counts: the numbers must agree with the test
    they are labelled with, even where the corpus is unavailable to re-derive them."""
    from math import comb

    assert curve["specifics"] == 134
    for c in curve["pairwise"]:
        n = c["gained"] + c["lost"]
        tail = sum(comb(n, k) for k in range(min(c["gained"], c["lost"]) + 1))
        assert c["p"] == pytest.approx(min(1.0, 2 * tail / 2**n)), (c["from"], c["to"])


def test_grounding_cost_recall_against_its_matched_control(curve):
    """The negative result, pinned so it cannot quietly become a positive one.

    The grounded extractor is compared against `batch1`, not against the shipped
    batch of 15: it extracts one session per request by construction, so anything
    else would be two variables wearing one number.
    """
    by_arm = {r["arm"]: r for r in curve["arms"]}
    grounded, control = by_arm["grounded"], by_arm["batch1"]
    assert grounded["sessions_per_request"] == control["sessions_per_request"] == 1
    assert grounded["overall"] < control["overall"]
    assert grounded["memories"] < control["memories"] / 2
    step = next(c for c in curve["pairwise"] if c["from"] == "batch1" and c["to"] == "grounded")
    assert step["lost"] > step["gained"] and step["p"] < 0.05, "the loss is not noise"


@pytest.mark.skipif(
    not (REPO / "data" / "longmemeval_s_cleaned.json").exists(),
    reason="the corpus is downloaded on demand and never committed",
)
def test_rescoring_the_archive_reproduces_fidelity_and_labels_prompt_drift():
    """The expensive half: the ruler, run again, on the archived extractions.

    This is what makes the curve reproducible without the cache, without the API
    and without quota. The transition-target schema intentionally changed Stage B
    after these arms ran, so the old arms must now be labelled stale rather than
    attributed to the current extractor. The fidelity ruler itself must still
    reproduce the immutable historical result.
    """
    fresh = _tool().score(write=False)
    recorded = json.loads(CURVE.read_text(encoding="utf-8"))
    assert fresh["cohort_matches"], "the corpus is not the one the arms were run on"
    assert not fresh["prompts_match"]
    assert fresh["stale_arms"] == ["batch15", "batch8", "batch5", "batch3", "batch1"]
    for new, old in zip(fresh["arms"], recorded["arms"], strict=True):
        assert new["arm"] == old["arm"]
        assert new["overall"] == pytest.approx(old["overall"]), new["arm"]
        assert new["per_facet"] == pytest.approx(old["per_facet"]), new["arm"]
    assert fresh["pairwise"] == recorded["pairwise"]


def test_v2b_changes_exactly_one_thing_about_v2():
    """The config claims to be v2 with one parameter moved. That is checkable.

    A candidate that quietly carried a second change would make the curve above
    evidence for something other than what it measured.
    """
    from llm_long_term_memory.config import ExperimentConfig

    def flat(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from flat(v, f"{prefix}{k}.")
            else:
                yield f"{prefix}{k}", v

    base = dict(flat(ExperimentConfig.from_yaml(REPO / "configs/v2.yaml").model_dump()))
    cand = dict(flat(ExperimentConfig.from_yaml(REPO / "configs/v2b-batch8.yaml").model_dump()))
    moved = {k for k in base | cand if base.get(k) != cand.get(k)}
    assert moved == {"name", "description", "ingest.sessions_per_request"}
    assert base["ingest.sessions_per_request"] == 15
    assert cand["ingest.sessions_per_request"] == 8
