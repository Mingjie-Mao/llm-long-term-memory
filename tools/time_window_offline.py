"""Reach of time-aware retrieval at 4,000 tokens, before any implementation. No calls.

    python tools/time_window_offline.py train150 train150.json
    python tools/time_window_offline.py dev100 dev100.json

For questions whose own words name a period ("last Saturday", "a week ago",
`time_window.question_window`, resolved from the question date), turns said in that
period — or whose own relative dates fall in it — are packed first, up to half the
budget, in BM25 order; the rest of the budget is v1's BM25 excerpts. A question counts
when every gold turn (LongMemEval `has_answer`) is in the excerpts. Every other question
is unchanged by construction and is not counted.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

BUDGET = 4000
WINDOW_SHARE = 0.5
CHARS_PER_TOKEN = 4.6


def overlaps(span, window) -> bool:
    return span[0] <= window[1] and window[0] <= span[1]


def pack(turns, budget, chosen, used):
    for turn in turns:
        if turn.id in chosen:
            continue
        cost = max(1, int(len(turn.content) / CHARS_PER_TOKEN))
        if used + cost > budget:
            continue
        chosen.add(turn.id)
        used += cost
    return used


def main() -> int:
    from llm_long_term_memory.evaluation.datasets import longmemeval as lme
    from llm_long_term_memory.evaluation.runners.memory import _as_of
    from llm_long_term_memory.retrieve.excerpts import archive_excerpts
    from llm_long_term_memory.retrieve.time_window import question_window, turn_dates
    from llm_long_term_memory.store import SQLiteMemoryStore, external_session_id

    parser = argparse.ArgumentParser()
    parser.add_argument("store")
    parser.add_argument("manifest")
    args = parser.parse_args()
    ids = json.loads((REPO / "results/manifests" / args.manifest).read_text(encoding="utf-8"))[
        "question_ids"
    ]
    instances = {i.question_id: i for i in lme.load("s", REPO / "data")}
    store = SQLiteMemoryStore(REPO / "stores" / f"{args.store}.db", read_only=True)
    store.initialize()
    counts = Counter()
    by_type = Counter()
    try:
        for qid in ids:
            inst = instances[qid]
            asked = _as_of(inst.question_date)
            window = question_window(inst.question, asked)
            gold = {
                (s.session_id, k)
                for s in inst.sessions
                for k, t in enumerate(s.turns)
                if t.has_answer
            }
            if window is None or not gold:
                continue
            counts["questions_with_window"] += 1
            by_type[inst.question_type] += 1
            own, dates = [], {}
            for sid in store.session_ids_for_user(qid):
                session = store.get_session(sid)
                dates[sid] = session.started_at if session else None
                own.extend(store.turns_for_session(sid))
            rank = {
                t.id: r for r, t in enumerate(store.search_turns(qid, inst.question, limit=500))
            }
            in_window = [
                t
                for t in own
                if (dates[t.session_id] and overlaps((dates[t.session_id].date(),) * 2, window))
                or (
                    t.role == "user"
                    and any(overlaps(s, window) for s in turn_dates(t.content, dates[t.session_id]))
                )
            ]
            in_window.sort(key=lambda t: (rank.get(t.id, 10**6), t.session_id, t.turn_index))
            v1 = archive_excerpts(store, qid, inst.question, BUDGET)
            key = {t.id: (external_session_id(t.session_id), t.turn_index) for t in own}
            got_v1 = {key[t.id] for t in v1.turns}
            chosen: set[str] = set()
            used = pack(in_window, int(BUDGET * WINDOW_SHARE), chosen, 0)
            ranked = sorted((t for t in own if t.id in rank), key=lambda t: rank[t.id])
            pack(ranked, BUDGET, chosen, used)
            got_w = {key[i] for i in chosen}
            counts["gold_in_window_turns"] += bool(gold & {key[t.id] for t in in_window})
            counts["all_gold_v1"] += gold <= got_v1
            counts["all_gold_window"] += gold <= got_w
            counts["window_gained"] += (gold <= got_w) and not (gold <= got_v1)
            counts["window_lost"] += (gold <= got_v1) and not (gold <= got_w)
    finally:
        store.close()
    print(json.dumps({"store": args.store, **counts, "by_type": dict(by_type)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def user_first_time_window(store, qid, question, asked_on, budget=BUDGET):
    """The candidate packing measured on 2026-10-07: the question's window (user turns,
    up to half the budget), then v1's BM25 order with the user's turns first
    (`archive_excerpts(prefer_user=True)`: 75% of the budget, then the rest)."""
    from llm_long_term_memory.retrieve.time_window import question_window, turn_dates

    own, dates = [], {}
    for sid in store.session_ids_for_user(qid):
        session = store.get_session(sid)
        dates[sid] = session.started_at if session else None
        own.extend(store.turns_for_session(sid))
    rank = {t.id: r for r, t in enumerate(store.search_turns(qid, question, limit=500))}
    ranked = sorted((t for t in own if t.id in rank), key=lambda t: rank[t.id])
    chosen: set[str] = set()
    used = 0
    window = question_window(question, asked_on)
    if window:
        in_window = [
            t
            for t in own
            if t.role == "user"
            and (
                (dates[t.session_id] and overlaps((dates[t.session_id].date(),) * 2, window))
                or any(overlaps(s, window) for s in turn_dates(t.content, dates[t.session_id]))
            )
        ]
        in_window.sort(key=lambda t: (rank.get(t.id, 10**6), t.session_id, t.turn_index))
        used = pack(in_window, int(budget * WINDOW_SHARE), chosen, used)
    used = pack([t for t in ranked if t.role == "user"], int(budget * 0.75), chosen, used)
    used = pack([t for t in ranked if t.role != "user"], budget, chosen, used)
    pack([t for t in ranked if t.role == "user"], budget, chosen, used)
    return [t for t in own if t.id in chosen]
