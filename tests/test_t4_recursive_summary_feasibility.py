import sys
from datetime import datetime
from pathlib import Path

from llm_long_term_memory.store import Session, SQLiteMemoryStore, Turn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from t4_recursive_summary_feasibility import size_histories


def test_sizing_keeps_assistant_cost_and_foreign_histories_separate(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "test.db")
    store.initialize()
    when = datetime(2026, 6, 1)
    try:
        for user in ("alice", "bob"):
            store.add_session(
                Session(
                    user,
                    user,
                    when,
                    "test",
                    [
                        Turn(user + ":0", user, 0, "user", "An original user statement.", when),
                        Turn(user + ":1", user, 1, "assistant", "A chorus. " * 2000, when),
                    ],
                )
            )
        users = size_histories(store, {"alice"}, ("user",))
        both = size_histories(store, {"alice"}, ("user", "assistant"))
        assert users["histories"] == both["histories"] == 1
        assert users["complete_build_sized"] and both["complete_build_sized"]
        assert both["approx_leaf_input_tokens"] > users["approx_leaf_input_tokens"]
        # Adding another tenant cannot alter Alice's resource estimate.
        store.add_session(
            Session(
                "bob-more",
                "bob",
                when,
                "test",
                [Turn("bob-more:0", "bob-more", 0, "user", "PRIVATE " * 2000, when)],
            )
        )
        assert size_histories(store, {"alice"}, ("user", "assistant")) == both
    finally:
        store.close()


def test_capped_sizing_does_not_claim_complete_build_cost(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "test.db")
    store.initialize()
    when = datetime(2026, 6, 1)
    try:
        store.add_session(
            Session(
                "long",
                "alice",
                when,
                "test",
                [Turn("long:0", "long", 0, "user", "Long original record. " * 40000, when)],
            )
        )
        sized = size_histories(store, {"alice"}, ("user",))
        assert sized["histories_with_omitted_raw_turns_at_32_page_cap"] == 1
        assert not sized["complete_build_sized"]
        assert sized["max_leaf_pages"] == 32
    finally:
        store.close()
