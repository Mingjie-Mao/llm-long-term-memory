import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import layer_audit

from llm_long_term_memory.store import Memory, Session, SQLiteMemoryStore, Turn
from llm_long_term_memory.store.session_keys import scoped_session_id


def _memory(mid, user, sid, predicate, op="coexists", status="active", day=1, scope=None):
    return Memory(
        id=mid,
        user_id=user,
        type="semantic",
        content=mid,
        token_count=1,
        status=status,
        subject="user",
        predicate=predicate,
        update_op=op,
        scope=scope,
        valid_from=datetime(2025, 1, day),
        ingested_at=datetime(2025, 1, day),
        source_session_id=scoped_session_id(user, sid),
        source_turn_index=0,
    )


def test_audit_counts_keys_predecessors_and_gold_coverage(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "s.db")
    store.initialize()
    for user, sid in (("ku", "old"), ("ku", "new"), ("pref", "p")):
        started = datetime(2025, 1, 1)
        scoped = scoped_session_id(user, sid)
        store.add_session(
            Session(
                scoped, user, started, turns=[Turn(f"{scoped}-0", scoped, 0, "user", "x", started)]
            )
        )
    store.add_memories(
        [
            _memory("a", "ku", "old", "employer", day=1, status="superseded"),
            _memory("b", "ku", "new", "employer", op="replaces", day=5),
            _memory("c", "ku", "new", "commute", op="replaces", day=5),  # nothing to replace
            _memory("d", "pref", "p", "likes", scope="preference"),
        ]
    )
    store.close()
    questions = {
        "ku": {
            "question_type": "knowledge-update",
            "answer_session_ids": ["old", "new"],
            "haystack_session_ids": ["old", "new"],
            "haystack_sessions": [[{"has_answer": True}], [{"has_answer": True}]],
        },
        "pref": {
            "question_type": "single-session-preference",
            "answer_session_ids": ["p"],
            "haystack_session_ids": ["p"],
            "haystack_sessions": [[{"has_answer": False}]],
        },
    }
    con = sqlite3.connect(tmp_path / "s.db")
    store_counts = layer_audit._store_counts(con)
    result = layer_audit.audit_split(questions, ["ku", "pref"], con)

    assert store_counts["replaces"] == 2
    assert store_counts["replaces_without_predecessor"] == 1
    assert result["all"]["ku_same_key"] == 1 and result["all"]["ku_superseded"] == 1
    pref = result["by_type"]["single-session-preference"]
    assert pref["pref_memory_on_gold_session"] == 1
    assert pref["memory_on_gold_turn"] == 0  # anchored, but not on a has_answer turn
