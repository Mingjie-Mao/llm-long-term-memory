"""Reach and size of raw-primary t1's date notes, before any call. No provider calls.

    python tools/time_notes_offline.py train150 train150.json --sample 40
    python tools/time_notes_offline.py heldout100 heldout100.json

Replays v1's 4,000-token excerpts from the read-only store and annotates them as t1
does. Reports how many questions gain a note, how many notes land on a gold turn
(LongMemEval `has_answer`), and the added size. `--sample` prints annotated
expressions with their conversation date for reading by hand — only use it on a set
whose per-item content may be read (train150).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    from llm_long_term_memory.evaluation.datasets import longmemeval as lme
    from llm_long_term_memory.retrieve.excerpts import archive_excerpts
    from llm_long_term_memory.retrieve.time_notes import _PATTERN, annotate
    from llm_long_term_memory.store import SQLiteMemoryStore, external_session_id

    parser = argparse.ArgumentParser()
    parser.add_argument("store")
    parser.add_argument("manifest")
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()
    ids = json.loads((REPO / "results/manifests" / args.manifest).read_text(encoding="utf-8"))[
        "question_ids"
    ]
    instances = {i.question_id: i for i in lme.load("s", REPO / "data")}
    store = SQLiteMemoryStore(REPO / "stores" / f"{args.store}.db", read_only=True)
    store.initialize()
    rows, samples = [], []
    try:
        for qid in ids:
            inst = instances[qid]
            gold = {
                (s.session_id, i)
                for s in inst.sessions
                for i, t in enumerate(s.turns)
                if t.has_answer
            }
            excerpts = archive_excerpts(store, qid, inst.question, 4000)
            notes = gold_notes = 0
            for turn in excerpts.turns:
                said = excerpts.session_dates.get(turn.session_id)
                n = annotate(turn.content, said)[1] if turn.role == "user" else 0
                notes += n
                if (external_session_id(turn.session_id), turn.turn_index) in gold:
                    gold_notes += n
                if n and args.sample:
                    for m in _PATTERN.finditer(turn.content):
                        text = annotate(m.group(0), said)[0]
                        if text != m.group(0):
                            start = max(0, m.start() - 60)
                            samples.append(
                                (said.date().isoformat(), turn.content[start : m.end() + 20], text)
                            )
            plain = excerpts.render()
            excerpts.time_notes = True
            excerpts.asked_on = lme_date(inst.question_date)
            noted = excerpts.render()
            rows.append(
                {
                    "question_id": qid,
                    "type": inst.question_type,
                    "notes": notes,
                    "gold_notes": gold_notes,
                    "added_tokens": (len(noted) - len(plain)) / 4.6,
                }
            )
    finally:
        store.close()
    temporal = [r for r in rows if r["type"] == "temporal-reasoning"]
    result = {
        "questions": len(rows),
        "questions_with_a_note": sum(r["notes"] > 0 for r in rows),
        "temporal_questions": len(temporal),
        "temporal_with_a_note_on_a_gold_turn": sum(r["gold_notes"] > 0 for r in temporal),
        "all_with_a_note_on_a_gold_turn": sum(r["gold_notes"] > 0 for r in rows),
        "added_tokens_median": statistics.median(r["added_tokens"] for r in rows),
        "added_tokens_max": max(r["added_tokens"] for r in rows),
    }
    print(json.dumps(result, indent=1))
    if args.sample:
        for said, context, text in random.Random(7).sample(samples, min(args.sample, len(samples))):
            print(f"[{said}] …{context}…  ->  {text}")
    return 0


def lme_date(value):
    from llm_long_term_memory.evaluation.runners.memory import _as_of

    return _as_of(value)


if __name__ == "__main__":
    raise SystemExit(main())
