import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import anchor_v2_backfill as tool

from llm_long_term_memory.store import Memory, NumpyFlatIndex, Session, SQLiteMemoryStore, Turn


def test_backfill_moves_only_anchors_and_leaves_the_original_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(tool, "STORES", tmp_path)
    store = SQLiteMemoryStore(tmp_path / "s.db")
    store.initialize()
    day = datetime(2025, 1, 1)
    turns = [
        Turn("a-0", "a", 0, "user", "I need help planning a trip.", day),
        Turn("a-1", "a", 1, "assistant", "Sure. Where to?", day),
        Turn("a-2", "a", 2, "user", "Kyoto in April, staying at the Hoshinoya ryokan.", day),
    ]
    store.add_session(Session("a", "u", day, turns=turns))
    store.add_memories(
        [
            Memory(
                id="m1",
                user_id="u",
                type="semantic",
                content="The user is staying at the Hoshinoya ryokan in Kyoto.",
                token_count=9,
                source_session_id="a",
                source_turn_index=0,  # a v1-style anchor on the wrong turn
                source_char_start=0,
                source_char_end=10,
            )
        ]
    )
    store.close()
    index = NumpyFlatIndex(tmp_path / "s-index", dim=2)
    index.add(["m1"], np.ones((1, 2), dtype=np.float32))
    index.save()
    before = (tmp_path / "s.db").read_bytes()

    counts = tool.backfill("s")

    assert (tmp_path / "s.db").read_bytes() == before
    assert counts["moved_turn"] == 1
    con = sqlite3.connect(tmp_path / "s-anchor-v2.db")
    turn, content = con.execute(
        "select source_turn_index, content from memories where id='m1'"
    ).fetchone()
    assert turn == 2 and content.startswith("The user is staying")
    assert con.execute("select value from meta where key='anchor_backfill'").fetchone()
    assert (tmp_path / "s-anchor-v2-index.npy").exists()
