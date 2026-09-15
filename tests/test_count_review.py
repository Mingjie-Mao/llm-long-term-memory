"""The human-review instrument, against the ways a gold set can quietly become fiction.

A review packet is only worth something if it cannot show sealed data, cannot be edited
after a reviewer signed it, cannot accept a member nobody can check in the supplied text,
and cannot be frozen against a store that has since changed. Each test below is one of
those failures. Passing here says the instrument refuses them; it says nothing about
whether the decisions a reviewer makes are correct.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

review = pytest.importorskip("count_review")

from llm_long_term_memory.store.session_keys import scoped_session_id  # noqa: E402


def boundary_store(rows: list[tuple[str, str]]) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, user_id TEXT)")
    connection.executemany(
        "INSERT INTO sessions VALUES (?,?)",
        [(scoped_session_id(user, external), user) for user, external in rows],
    )
    return connection


def manifest_repo(tmp_path: Path, question_ids: list[str]) -> Path:
    path = tmp_path / "results/manifests"
    path.mkdir(parents=True)
    (path / "v3-reasoning-dev60.json").write_text(
        json.dumps({"question_ids": question_ids}), encoding="utf-8"
    )
    return tmp_path


def packet_of(items: list[dict]) -> dict:
    packet = {
        "schema_version": 1,
        "status": "pending_human_review",
        "scope": "development_only_bounded_source_pool",
        "inputs": {"policy": {"path": "configs/count-policy-v1.json", "sha256": "a" * 64}},
        "store_fingerprint_kind": "sqlite-evidence-content-v1",
        "store_fingerprint": "f" * 64,
        "items": items,
        "quarantine": [],
    }
    packet["packet_sha256"] = review.digest(packet)
    return packet


def entity_item(item_id: str = "q1", *, flags: list[str] | None = None) -> dict:
    return {
        "id": item_id,
        "kind": "entity",
        "namespace": "ns1",
        "question": "How many distinct plants does the user grow?",
        "relation": "grows",
        "memories": [{"id": "m1", "predicate": "plants_grown"}],
        "sources": [
            {"id": "s1", "role": "user", "text": "I finally planted the basil this weekend."},
            {"id": "s2", "role": "assistant", "text": "Basil does well on a sunny sill."},
        ],
        "flags": flags or [],
        "proposed_entities": ["basil"],
        "prior_model_verdict": None,
        "cluster_id": "cluster_1",
    }


def decisions_of(packet: dict, rows: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "packet_sha256": packet["packet_sha256"],
        "reviewer_kind": "human",
        "reviewer": "Reviewer A",
        "policy_approved": True,
        "reviewed_at": "2026-09-14T01:00:00Z",
        "decisions": rows,
    }


def accept_row(item_id: str = "q1", **over) -> dict:
    row = {
        "id": item_id,
        "verdict": "accept",
        "reason": "one member, quoted from the user's own turn",
        "scope_complete": True,
        "members": [{"entity": "basil", "source_id": "s1", "quote": "planted the basil"}],
    }
    row.update(over)
    return row


# --- boundary: sealed development data must never enter the packet -------------------


def test_boundary_blocks_protected_namespace_and_every_session_it_shares(tmp_path):
    connection = boundary_store([("sealed_ns", "sess-a"), ("open_ns", "sess-a")])
    blocked, sessions = review.development_boundary(
        connection, manifest_repo(tmp_path, ["sealed_ns"])
    )
    assert blocked == {"sealed_ns"}
    # The same external session is also reachable from a readable namespace; it is still
    # protected, because quoting it would expose sealed turns through the open side.
    assert sessions == {"sess-a"}


def test_boundary_refuses_an_empty_protected_manifest(tmp_path):
    connection = boundary_store([("open_ns", "sess-a")])
    with pytest.raises(ValueError, match="empty protected manifest"):
        review.development_boundary(connection, manifest_repo(tmp_path, []))


def test_clusters_join_questions_through_a_shared_session_chain():
    items = [
        {"id": "q1", "namespace": "a", "sources": [{"session_key": "s1"}]},
        {"id": "q2", "namespace": "b", "sources": [{"session_key": "s1"}, {"session_key": "s2"}]},
        {"id": "q3", "namespace": "c", "sources": [{"session_key": "s2"}]},
        {"id": "q4", "namespace": "d", "sources": [{"session_key": "s9"}]},
    ]
    review.cluster_items(items)
    assert items[0]["cluster_id"] == items[1]["cluster_id"] == items[2]["cluster_id"]
    assert items[3]["cluster_id"] != items[0]["cluster_id"]


# --- the packet a reviewer signed is the packet that is scored -----------------------


def test_a_packet_edited_after_signing_is_rejected():
    packet = packet_of([entity_item()])
    packet["items"][0]["question"] = "How many plants did the user buy?"
    with pytest.raises(ValueError, match="content changed"):
        review.verify_packet(packet)


def test_decisions_from_another_packet_are_rejected():
    packet = packet_of([entity_item()])
    decisions = decisions_of(packet, [accept_row()])
    decisions["packet_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="different packet"):
        review.validate_decisions(packet, decisions)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reviewer_kind", "model", "human reviewer"),
        ("reviewer", "  ", "human reviewer"),
        ("policy_approved", False, "policy and review time"),
        ("reviewed_at", "", "policy and review time"),
    ],
)
def test_a_decision_file_without_a_named_human_and_approved_policy_is_rejected(
    field, value, message
):
    packet = packet_of([entity_item()])
    decisions = decisions_of(packet, [accept_row()])
    decisions[field] = value
    with pytest.raises(ValueError, match=message):
        review.validate_decisions(packet, decisions)


def test_every_reviewable_item_needs_exactly_one_decision():
    packet = packet_of([entity_item("q1"), entity_item("q2")])
    with pytest.raises(ValueError, match="every reviewable item"):
        review.validate_decisions(packet, decisions_of(packet, [accept_row("q1")]))


@pytest.mark.parametrize("over", [{"verdict": ""}, {"verdict": "maybe"}, {"reason": "  "}])
def test_each_decision_needs_a_verdict_and_a_stated_reason(over):
    packet = packet_of([entity_item()])
    with pytest.raises(ValueError, match="verdict and reason"):
        review.validate_decisions(packet, decisions_of(packet, [accept_row(**over)]))


def test_a_flagged_question_cannot_be_accepted_as_written():
    packet = packet_of([entity_item(flags=["unapproved_predicates: groceries"])])
    with pytest.raises(ValueError, match="excluded or rewritten"):
        review.validate_decisions(packet, decisions_of(packet, [accept_row()]))


def test_accepting_without_confirming_the_source_scope_is_rejected():
    packet = packet_of([entity_item()])
    decisions = decisions_of(packet, [accept_row(scope_complete=False)])
    with pytest.raises(ValueError, match="scope completeness"):
        review.validate_decisions(packet, decisions)


@pytest.mark.parametrize(
    "members",
    [
        # the same entity twice, only cased differently
        [
            {"entity": "basil", "source_id": "s1", "quote": "planted the basil"},
            {"entity": "Basil", "source_id": "s1", "quote": "planted the basil"},
        ],
        # a quote that is not in the supplied text
        [{"entity": "basil", "source_id": "s1", "quote": "I grow basil and mint"}],
        # a source id that was never supplied
        [{"entity": "basil", "source_id": "s7", "quote": "planted the basil"}],
        # an entity with no quote at all
        [{"entity": "basil", "source_id": "s1", "quote": "   "}],
    ],
)
def test_a_member_must_be_distinct_and_quoted_verbatim_from_a_supplied_source(members):
    packet = packet_of([entity_item()])
    decisions = decisions_of(packet, [accept_row(members=members)])
    with pytest.raises(ValueError, match="distinct and cite a verbatim"):
        review.validate_decisions(packet, decisions)


def test_an_empty_member_list_needs_explicit_evidence_that_the_answer_is_zero():
    packet = packet_of([entity_item()])
    decisions = decisions_of(packet, [accept_row(members=[])])
    with pytest.raises(ValueError, match="zero needs explicit source evidence"):
        review.validate_decisions(packet, decisions)


def test_zero_is_accepted_when_a_supplied_source_says_so():
    packet = packet_of([entity_item()])
    row = accept_row(
        members=[],
        zero_evidence={"source_id": "s1", "quote": "planted the basil"},
    )
    accepted = review.validate_decisions(packet, decisions_of(packet, [row]))
    assert accepted[0]["answer"] == 0


def test_an_accepted_question_carries_its_members_sources_and_cluster():
    packet = packet_of([entity_item()])
    accepted = review.validate_decisions(packet, decisions_of(packet, [accept_row()]))
    assert len(accepted) == 1
    assert accepted[0]["answer"] == 1
    assert accepted[0]["cluster_id"] == "cluster_1"
    assert accepted[0]["sources"] == packet["items"][0]["sources"]
    # the unreviewed generator output travels with it, as the control arm's input
    assert accepted[0]["proposed_entities"] == ["basil"]


# --- freezing binds the decisions to the data they quote -----------------------------


def test_freezing_binds_the_packet_decisions_and_store(monkeypatch):
    packet = packet_of([entity_item()])
    decisions = decisions_of(packet, [accept_row()])
    monkeypatch.setattr(review, "current_store_fingerprint", lambda repo=None: "f" * 64)
    gold = review.freeze_gold(packet, decisions)
    assert gold["bound_to"]["packet_sha256"] == packet["packet_sha256"]
    assert gold["bound_to"]["decisions_sha256"] == review.digest(decisions)
    assert gold["bound_to"]["store_fingerprint"] == packet["store_fingerprint"]
    assert gold["summary"] == {
        "questions": 1,
        "clusters": 1,
        "members": 1,
        "zero_answers": 0,
    }
    assert gold["gold_sha256"] == review.digest(
        {k: v for k, v in gold.items() if k != "gold_sha256"}
    )


def test_freezing_stops_when_the_evidence_store_changed_under_the_review(monkeypatch):
    packet = packet_of([entity_item()])
    decisions = decisions_of(packet, [accept_row()])
    monkeypatch.setattr(review, "current_store_fingerprint", lambda repo=None: "0" * 64)
    with pytest.raises(ValueError, match="store changed"):
        review.freeze_gold(packet, decisions)


def test_a_review_that_accepted_nothing_cannot_be_frozen(monkeypatch):
    packet = packet_of([entity_item()])
    row = {"id": "q1", "verdict": "rewrite", "reason": "the unit in the question is wrong"}
    monkeypatch.setattr(review, "current_store_fingerprint", lambda repo=None: "f" * 64)
    with pytest.raises(ValueError, match="nothing to freeze"):
        review.freeze_gold(packet, decisions_of(packet, [row]))


def test_the_committed_manifest_carries_the_hashes_but_not_the_raw_turns(tmp_path, monkeypatch):
    packet = packet_of([entity_item()])
    packet_file = tmp_path / "packet.json"
    packet_file.write_text(json.dumps(packet), encoding="utf-8")
    decisions_file = tmp_path / "decisions.json"
    decisions_file.write_text(json.dumps(decisions_of(packet, [accept_row()])), encoding="utf-8")
    gold_file, manifest_file = tmp_path / "gold.json", tmp_path / "gold-manifest.json"
    monkeypatch.setattr(review, "current_store_fingerprint", lambda repo=None: "f" * 64)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "count_review.py",
            "freeze",
            "--packet",
            str(packet_file),
            "--decisions",
            str(decisions_file),
            "--out",
            str(gold_file),
            "--manifest",
            str(manifest_file),
        ],
    )
    assert review.main() == 0
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert (
        manifest["gold_sha256"] == json.loads(gold_file.read_text(encoding="utf-8"))["gold_sha256"]
    )
    assert "questions" not in manifest
    # the reviewer's own words are in the gold file, never in the one that is committed
    assert "planted the basil" in gold_file.read_text(encoding="utf-8")
    assert "planted the basil" not in manifest_file.read_text(encoding="utf-8")


def test_the_review_page_puts_every_conclusion_after_the_evidence():
    """Anchoring is the one contamination the freeze checks cannot detect afterwards.

    The page used to render a verdict-shaped line ("现有模型建议") above the memories and
    the raw turns, with both of those collapsed. A reviewer who reads top to bottom then
    meets a conclusion before any evidence. Order is the whole mitigation, so it is
    pinned here rather than left to whoever next edits the template.
    """
    ui = pytest.importorskip("count_review_ui")
    body = ui.render(packet_of([entity_item()]))
    evidence = body.index("原文证据")
    assert evidence < body.index("旧生成器的成员")
    assert evidence < body.index("待复核的模型裁定")
    assert "现有模型建议" not in body


def test_an_entity_question_never_carries_a_prior_model_verdict(monkeypatch):
    """Gold members come from entity questions, and no model decided any part of one."""
    monkeypatch.setattr(review, "current_store_fingerprint", lambda repo=None: "f" * 64)
    item = entity_item()
    assert item["prior_model_verdict"] is None
    packet = packet_of([item])
    gold = review.freeze_gold(packet, decisions_of(packet, [accept_row()]))
    assert gold["prior_verdict_agreement"]["compared"] == 0


def test_freezing_records_how_often_the_human_matched_the_prior_model_verdict(monkeypatch):
    """A diagnostic, never a gate: it is recorded so a later reader can see the number."""
    monkeypatch.setattr(review, "current_store_fingerprint", lambda repo=None: "f" * 64)
    legacy = entity_item("q2")
    legacy["kind"] = "legacy"
    legacy["prior_model_verdict"] = "exclude"
    packet = packet_of([entity_item(), legacy])
    decisions = decisions_of(
        packet,
        [accept_row(), {"id": "q2", "verdict": "exclude", "reason": "agreed on review"}],
    )
    agreement = review.freeze_gold(packet, decisions)["prior_verdict_agreement"]
    assert agreement == {
        "scope": "legacy_re_reviews_only",
        "compared": 1,
        "agreed": 1,
        "rate": 1.0,
        "enforced": False,
    }


def test_questions_sharing_a_recorded_turn_must_share_a_cluster():
    """One recorded turn behind two questions is one observation, not two."""
    a, b = entity_item("q1"), entity_item("q2")
    a["cluster_id"], b["cluster_id"] = "cluster_a", "cluster_b"
    with pytest.raises(ValueError, match="share a recorded turn but not a cluster"):
        review.shared_evidence_audit([a, b])
    b["cluster_id"] = "cluster_a"
    audit = review.shared_evidence_audit([a, b])
    assert audit["question_pairs_sharing_a_turn"] == 1
    assert audit["all_in_one_cluster"] is True


def test_two_questions_naming_the_same_object_over_separate_turns_stay_separate():
    """`sapiens` read by two different users is two observations, not a duplicate.

    Eight object strings repeat across the probe set while only one pair of questions
    shares an actual turn, so merging on the label would have destroyed seven real
    observations. The criterion is the turn; this pins that it is not the name.
    """
    a, b = entity_item("q1"), entity_item("q2")
    b["namespace"] = "ns2"
    for source in a["sources"]:
        source["session_key"] = "session_a"
    b["sources"] = [
        {
            "id": "s9",
            "role": "user",
            "session_key": "session_b",
            "text": "I finished Sapiens this week.",
        }
    ]
    review.cluster_items([a, b])
    assert a["cluster_id"] != b["cluster_id"]
    assert review.shared_evidence_audit([a, b])["question_pairs_sharing_a_turn"] == 0
