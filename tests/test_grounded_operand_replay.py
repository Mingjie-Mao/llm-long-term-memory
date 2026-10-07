from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_long_term_memory.runtime.grounded_answering import (
    EvidenceLedger,
    EvidenceSource,
    SessionEvidenceLedger,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from grounded_operand_replay import restore_ledger


@pytest.mark.parametrize("ledger_type", [EvidenceLedger, SessionEvidenceLedger])
def test_operand_replay_requires_unchanged_text_and_tenant_provenance(ledger_type):
    original = ledger_type(
        sources=[EvidenceSource("E1", "raw", "t1", "I paid $60.", "user", "2026-10-01", "s1", 0)]
    )
    turn = SimpleNamespace(id="t1", content="I paid $60.")
    session = SimpleNamespace(id="s1", user_id="alice")
    store = SimpleNamespace(
        iter_all=lambda user: [],
        get_session=lambda sid: session,
        turns_for_session=lambda sid: [turn],
    )
    assert (
        restore_ledger(store, "alice", original.audit(), ledger_type=ledger_type).render()
        == original.render()
    )
    turn.content = "I paid $80."
    with pytest.raises(ValueError, match="text changed"):
        restore_ledger(store, "alice", original.audit(), ledger_type=ledger_type)
    turn.content = "I paid $60."
    session.user_id = "bob"
    with pytest.raises(ValueError, match="another tenant"):
        restore_ledger(store, "alice", original.audit(), ledger_type=ledger_type)
