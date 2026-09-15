"""The registered declaration, and the artifacts that have to agree with it.

A declaration that names an ability BEAM does not have, or that leaves one unreported,
would be discovered when the result is read — which is after the quota is spent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from llm_long_term_memory.evaluation.datasets.beam import ABILITIES

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

DECLARATION = json.loads((REPO / "configs/beam-eval.json").read_text(encoding="utf-8"))
TAXONOMY = json.loads(
    (REPO / "results/analysis/beam-ability-taxonomy.json").read_text(encoding="utf-8")
)


def test_every_ability_is_either_primary_or_reported_separately():
    primary = set(DECLARATION["primary"]["abilities"])
    separate = set(DECLARATION["reported_separately"]["abilities"])
    assert primary & separate == set()
    assert primary | separate == set(ABILITIES)


def test_the_registration_it_names_exists():
    assert (REPO / DECLARATION["registration"]).is_file()


def test_no_gate_is_registered_against_a_noise_floor_nobody_measured():
    """`results/gate-protocol.md`: a threshold checked against an unmeasured instrument
    fails and passes the same configuration depending on the draw."""
    assert DECLARATION["gate"].startswith("none registered")
    assert "unmeasured" in DECLARATION["noise_floor"]


def test_the_power_table_was_built_from_this_declaration():
    import hashlib

    power = json.loads((REPO / "results/analysis/beam-power.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((REPO / "configs/beam-eval.json").read_bytes()).hexdigest()
    assert power["inputs"]["declaration_sha256"] == digest
    primary = len(DECLARATION["primary"]["abilities"])
    assert any(f"the {primary} primary abilities" == row["stratum"] for row in power["strata"])


@pytest.mark.parametrize("ability", ("instruction_following", "preference_following"))
def test_no_ability_is_separated_for_being_about_wording(ability):
    """The original proposal kept these out of the primary because "how the answer is
    written moves these as much as what memory kept". Over the development half that is 4
    rubric items of 68 and 3 of 73 — the reason does not hold, so whatever line is drawn
    between them, it cannot be that one."""
    row = TAXONOMY["abilities"][ability]
    assert row["manner_after_reading"] / row["rubric_items"] < 0.1


def test_the_line_between_the_two_directive_abilities_is_declared_not_implied():
    """They are not separable by any measurement this project has: the manner screen puts
    them at 4/68 and 3/73, and the coverage analyzer decides 1 of 68 and 3 of 73. Splitting
    them is a judgement, so both compositions have to be reported."""
    primary = set(DECLARATION["primary"]["abilities"])
    separate = set(DECLARATION["reported_separately"]["abilities"])
    directive = set(DECLARATION["ability_groups"]["standing_directive"])
    assert directive & primary and directive & separate
    assert "mean rubric score over the 7 fact abilities alone" in DECLARATION["secondary"]
    assert any("plus instruction_following" in row for row in DECLARATION["secondary"])


def test_summarization_is_separated_for_its_item_count_not_for_its_wording():
    row = TAXONOMY["abilities"]["summarization"]
    assert row["manner_after_reading"] == 0
    assert row["items_per_question"]["median"] >= 5
    assert "summarization" in DECLARATION["reported_separately"]["abilities"]


def test_every_screened_item_carries_a_hand_verdict():
    taxonomy = pytest.importorskip("beam_ability_taxonomy")
    unjudged = [key for key in taxonomy.REVIEWED if not key.startswith("beam-")]
    assert unjudged == []
    assert set(taxonomy.REVIEWED.values()) <= {"manner", "content"}
