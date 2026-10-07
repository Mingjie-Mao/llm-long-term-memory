import copy
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_long_term_memory.conversation import AnswerRequest
from llm_long_term_memory.llm.usage import UsageTracker
from llm_long_term_memory.runtime.evidence_pages import PageSelection
from llm_long_term_memory.runtime.grounded_answering import (
    EvidenceLedger,
    EvidenceSource,
    GroundedAnswererV15,
)
from llm_long_term_memory.store import Session, SQLiteMemoryStore, Turn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from recursive_summary_nav import (
    NavigationFailure,
    RegionSelection,
    SummaryNavigationAnswerer,
    build_tree,
    digest,
    navigate,
    validate_tree,
)


class Client:
    def __init__(self):
        self.calls = []
        self.foreign = False
        self.nav_first_only = False

    def generate(self, **kw):
        self.calls.append(kw)
        schema = kw.get("schema")
        if schema == RegionSelection:
            ids = json.loads(kw["prompt"].split("Required reviewed IDs: ")[1])
            text = RegionSelection(
                selected=["foreign"] if self.foreign else ids[:1] if self.nav_first_only else ids,
                reviewed=ids,
            ).model_dump_json()
        elif schema == PageSelection:
            ids = re.findall(r"^\[(E\d+)\]", kw["prompt"], re.M)
            text = PageSelection(
                selected_sources=ids, reviewed_sources=ids, scope_complete=True
            ).model_dump_json()
        else:
            text = "WRONG SUMMARY: team always has 999 members. Navigation only."
        return SimpleNamespace(text=text, input_tokens=10, output_tokens=5, api_latency_ms=1)


@pytest.fixture
def store(tmp_path):
    db = SQLiteMemoryStore(tmp_path / "s.db")
    db.initialize()
    for user, sid, body in [
        ("alice", "a", "In January team size5. In March size7. In June size6. " * 800),
        ("bob", "b", "BOB_PRIVATE_UNIQUE"),
    ]:
        turn = Turn(sid + ":0", sid, 0, "user", body, datetime(2026, 6, 1))
        db.add_session(Session(sid, user, datetime(2026, 6, 1), "test", [turn]))
    yield db
    db.close()


def build(store, tmp_path, client):
    path = tmp_path / (digest("alice") + ".tree.json")
    usage = UsageTracker()
    tree = build_tree(
        store, "alice", client, "same-extractor", path, usage, tmp_path / "usage.json"
    )
    return path, tree, usage


def test_recursive_tree_keeps_every_raw_span_and_excludes_foreign_user(store, tmp_path):
    path, tree, _ = build(store, tmp_path, Client())
    validate_tree(tree, "alice")
    assert tree["nodes"][tree["root"]]["children"]
    spans = [s for n in tree["nodes"].values() for s in n["sources"]]
    text = "".join(s["text"] for s in spans)
    assert text == store.turns_for_session("a")[0].content
    assert "BOB_PRIVATE_UNIQUE" not in path.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="foreign"):
        validate_tree(tree, "bob")


def test_completed_summary_checkpoint_is_immutable_and_uses_no_new_calls(store, tmp_path):
    client = Client()
    path, tree, usage = build(store, tmp_path, client)
    before, calls = path.read_bytes(), len(client.calls)
    resumed = build_tree(
        store, "alice", client, "same-extractor", path, usage, tmp_path / "usage.json"
    )
    assert resumed == tree and path.read_bytes() == before and len(client.calls) == calls


def test_failed_node_invocation_budget_survives_restart(store, tmp_path):
    class Broken(Client):
        def generate(self, **kw):
            self.calls.append(kw)
            raise RuntimeError("provider failure")

    client = Broken()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="provider failure"):
            build(store, tmp_path, client)
    with pytest.raises(ValueError, match="budget exhausted"):
        build(store, tmp_path, client)
    assert len(client.calls) == 2
    checkpoint = json.loads(
        (tmp_path / (digest("alice") + ".tree.json")).read_text(encoding="utf-8")
    )
    assert list(checkpoint["node_attempts"].values()) == [2]


@pytest.mark.parametrize("change", ["cycle", "raw", "user"])
def test_bad_tree_identity_or_raw_provenance_is_rejected(store, tmp_path, change):
    _, tree, _ = build(store, tmp_path, Client())
    tree = copy.deepcopy(tree)
    root = tree["nodes"][tree["root"]]
    if change == "cycle":
        root["children"] = [tree["root"]]
    elif change == "user":
        root["user_id"] = "bob"
    else:
        next(n for n in tree["nodes"].values() if n["sources"])["sources"][0]["text"] = "invented"
    with pytest.raises(ValueError):
        validate_tree(tree, "alice")


def test_navigation_cannot_select_foreign_child(store, tmp_path):
    client = Client()
    _, tree, _ = build(store, tmp_path, client)
    client.foreign = True
    with pytest.raises(NavigationFailure, match="foreign"):
        navigate(tree, AnswerRequest("March size?", "2026-06-01", "alice"), client, "same-answerer")


def test_answer_ledger_uses_raw_not_false_summary(store, tmp_path, monkeypatch):
    client = Client()
    client.nav_first_only = True
    path, _, _ = build(store, tmp_path, client)
    monkeypatch.setattr(GroundedAnswererV15, "extend_ledger", lambda *a: {})
    answerer = SummaryNavigationAnswerer(
        client,
        model="same-answerer",
        encoder=None,
        store=store,
        turn_index=[1],
        tree_dir=tmp_path,
        tree_hashes={"alice": hashlib.sha256(path.read_bytes()).hexdigest()},
    )
    ledger = EvidenceLedger()
    detail = answerer.extend_ledger(
        ledger, AnswerRequest("What was my team size in March?", "2026-06-01", "alice"), [], None
    )
    assert "WRONG SUMMARY" not in ledger.render()
    assert "In March size7" in ledger.render()
    assert detail["archive_review"]["globally_exhaustive"] is False


def test_incremental_update_is_cost_only_and_leaves_original_tree_unchanged(store, tmp_path):
    client = Client()
    path, tree, usage = build(store, tmp_path, client)
    before = path.read_bytes()
    extra = EvidenceSource(
        "E99",
        "raw",
        "synthetic-cost-update",
        "My project name changed today.",
        "user",
        "2026-10-04",
        session_id="cost-only",
    )
    update = build_tree(
        store,
        "alice",
        client,
        "same-extractor",
        tmp_path / "update.json",
        usage,
        tmp_path / "update-usage.json",
        inherited=tree,
        extra_source=extra,
    )
    assert update["header"]["update_cost_only"] and path.read_bytes() == before
    assert update["new_nodes"] < len(update["nodes"])


def test_quota_headroom_pause_does_not_charge_node_attempt(store, tmp_path):
    from llm_long_term_memory.llm.client import DailyQuotaExhausted

    client = Client()
    client.quota = SimpleNamespace(for_model=lambda model: SimpleNamespace(remaining_today=2))
    with pytest.raises(DailyQuotaExhausted):
        build(store, tmp_path, client)
    path = tmp_path / (digest("alice") + ".tree.json")
    assert not client.calls
    assert json.loads(path.read_text(encoding="utf-8"))["node_attempts"] == {}
