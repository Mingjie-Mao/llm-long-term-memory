"""Stage B of research/ROADMAP.md: v2 anchors for the existing stores, and whether the
evidence that reaches the answer gets better. Zero model calls.

    python tools/anchor_v2_backfill.py backfill   # writes stores/<split>-anchor-v2.*
    python tools/anchor_v2_backfill.py replay     # results/analysis/anchor-v2-backfill-v1.*

`backfill` copies `stores/<split>.db` (SQLite backup API, the original untouched) and its
vector index to `<split>-anchor-v2`, then recomputes each memory's anchor with
`source_span_for_v2` — the rule new ingests already use (`ANCHOR_VERSION`). Only
`source_turn_index`, `source_char_start` and `source_char_end` change; memory text,
ids, keys and vectors are identical, so retrieval is unchanged and any difference in the
answer context comes from the anchors alone. `event_time_source_expression` is left as
the v1 anchor found it.

`replay` re-runs t4's evidence hydration for every baseline row of the regression set
(dev100 + heldout100, 3 runs each) on both stores, from the ranked memory ids each run
recorded. On the original store it must reproduce the hydrated memory ids the run
recorded; it then reports how often the hydrated sentences land on a gold turn
(LongMemEval `has_answer`) before and after.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from t4_baseline_audit import BASELINE, gold_turns, load_runs  # noqa: E402

from llm_long_term_memory.conversation import ConversationSession, ConversationTurn  # noqa: E402
from llm_long_term_memory.ingest.provenance import source_span_for_v2  # noqa: E402
from llm_long_term_memory.retrieve import EvidenceHydrator  # noqa: E402
from llm_long_term_memory.store import SQLiteMemoryStore  # noqa: E402
from llm_long_term_memory.store.session_keys import external_session_id  # noqa: E402

STORES = REPO / "stores"
OUT = REPO / "results/analysis/anchor-v2-backfill-v1"
SUFFIX = "-anchor-v2"
# t4's hydration settings (configs/fallback.yaml defaults, runtime ANSWER_PROFILES).
HYDRATION = {"neighbouring_sentences": 1, "allocation": "ranked"}
HYDRATION_TOKENS = 800


def backfill(split: str) -> dict:
    source = STORES / f"{split}.db"
    target = STORES / f"{split}{SUFFIX}.db"
    if target.exists():
        raise FileExistsError(f"{target} exists; remove it to rebuild")
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as src, sqlite3.connect(target) as dst:
        src.backup(dst)
    for ext in ("-index.npy", "-index.ids.json"):
        shutil.copyfile(STORES / f"{split}{ext}", STORES / f"{split}{SUFFIX}{ext}")

    store = SQLiteMemoryStore(target)
    counts = {"memories": 0, "anchored": 0, "moved_turn": 0, "moved_span": 0, "unanchorable": 0}
    sessions: dict[str, ConversationSession] = {}
    updates = []
    try:
        rows = store._conn.execute(
            "select id, content, source_session_id, source_turn_index, source_char_start,"
            " source_char_end from memories where source_session_id is not null"
        ).fetchall()
        for mid, content, sid, turn, start, end in rows:
            counts["memories"] += 1
            if sid not in sessions:
                session = store.get_session(sid)
                turns = store.turns_for_session(sid)
                sessions[sid] = ConversationSession(
                    session_id=sid,
                    date=str(session.started_at) if session and session.started_at else "",
                    turns=[ConversationTurn(role=t.role, content=t.content) for t in turns],
                )
            span = source_span_for_v2(content, sessions[sid])
            if span is None:
                counts["unanchorable"] += 1
                continue
            counts["anchored"] += 1
            counts["moved_turn"] += span[0] != turn
            counts["moved_span"] += tuple(span) != (turn, start, end)
            updates.append((*span, mid))
        store._conn.executemany(
            "update memories set source_turn_index=?, source_char_start=?, source_char_end=?"
            " where id=?",
            updates,
        )
        store._conn.execute(
            "insert or replace into meta(key, value) values ('anchor_backfill', ?)",
            (json.dumps({"rule": "provenance-v2", "from": f"{split}.db"}),),
        )
        store._conn.commit()
    finally:
        store.close()
    return counts


def _hydrate(store, hydrator, ids):
    memories = {m.id: m for m in store.get_many(ids)}
    ranked = [memories[i] for i in ids if i in memories]
    return hydrator.hydrate(ranked, max_tokens=HYDRATION_TOKENS)


def replay() -> dict:
    questions = {
        q["question_id"]: q
        for q in json.loads((REPO / "data/longmemeval_s_cleaned.json").read_text(encoding="utf-8"))
    }
    report: dict = {"splits": {}}
    for split, spec in BASELINE.items():
        runs = load_runs(split)
        before = SQLiteMemoryStore(STORES / f"{spec['store']}.db", read_only=True)
        after = SQLiteMemoryStore(STORES / f"{spec['store']}{SUFFIX}.db", read_only=True)
        hydrate_before = EvidenceHydrator(before, **HYDRATION)
        hydrate_after = EvidenceHydrator(after, **HYDRATION)
        s = {
            "rows": 0,
            "replay_matches_recorded": 0,
            "rows_with_gold_in_hydration": {"before": 0, "after": 0},
            "hydrated_items": {"before": 0, "after": 0},
            "hydrated_items_on_gold_turn": {"before": 0, "after": 0},
            "hydrated_tokens": {"before": 0, "after": 0},
            "questions_gaining_gold": [],
            "questions_losing_gold": [],
        }
        try:
            for run in runs:
                for qid, row in run.items():
                    ids = row["notes"]["ranked_memory_ids"]
                    gold = gold_turns(questions[qid])
                    s["rows"] += 1
                    result = {}
                    for name, store, hydrator in (
                        ("before", before, hydrate_before),
                        ("after", after, hydrate_after),
                    ):
                        h = _hydrate(store, hydrator, ids)
                        on_gold = [
                            e
                            for e in h.evidence
                            if (external_session_id(e.session_id), e.turn_index) in gold
                        ]
                        result[name] = bool(on_gold)
                        s["hydrated_items"][name] += len(h.evidence)
                        s["hydrated_items_on_gold_turn"][name] += len(on_gold)
                        s["rows_with_gold_in_hydration"][name] += bool(on_gold)
                        s["hydrated_tokens"][name] += h.tokens
                        if name == "before":
                            recorded = row["notes"]["hydrated_memory_ids"]
                            hydrated = [e.memory_id for e in h.evidence]
                            s["replay_matches_recorded"] += hydrated == recorded
                    if result["after"] and not result["before"]:
                        s["questions_gaining_gold"].append(qid)
                    if result["before"] and not result["after"]:
                        s["questions_losing_gold"].append(qid)
        finally:
            before.close()
            after.close()
        for key in ("questions_gaining_gold", "questions_losing_gold"):
            s[key] = sorted(set(s[key]))
        report["splits"][split] = s
    return report


def render(report: dict) -> str:
    lines = [
        "# v2 anchor backfill: does the hydrated evidence improve? (stage B)",
        "",
        "> Generated by `tools/anchor_v2_backfill.py replay`; zero model calls. Exposed",
        "> development splits; the t4 baseline's own ranked memories, hydrated on the original",
        "> store and on the v2-anchored copy. Not an accuracy result.",
        "",
    ]
    if "backfill" in report:
        lines += [
            "## Backfill",
            "",
            "| split | memories | re-anchored | turn changed | unanchorable |",
        ]
        lines += ["|---|---:|---:|---:|---:|"]
        for split, c in report["backfill"].items():
            lines.append(
                f"| {split} | {c['memories']:,} | {c['anchored']:,} | {c['moved_turn']:,} |"
                f" {c['unanchorable']:,} |"
            )
        lines.append("")
    lines += [
        "## Hydration replay (3 runs x 100 questions per split)",
        "",
        "| split | replay = recorded | rows with a gold turn in hydration |"
        " hydrated items on a gold turn | hydrated tokens |",
        "|---|---:|---|---|---|",
    ]
    for split, s in report["splits"].items():
        b, a = s["rows_with_gold_in_hydration"], s["hydrated_items_on_gold_turn"]
        items = s["hydrated_items"]
        tok = s["hydrated_tokens"]
        lines.append(
            f"| {split} | {s['replay_matches_recorded']}/{s['rows']} |"
            f" {b['before']} → {b['after']} |"
            f" {a['before']}/{items['before']} → {a['after']}/{items['after']} |"
            f" {tok['before']:,} → {tok['after']:,} |"
        )
    lines.append("")
    for split, s in report["splits"].items():
        lines.append(
            f"- {split}: gold newly reached by hydration in "
            f"{', '.join(f'`{q}`' for q in s['questions_gaining_gold']) or 'none'}; "
            f"lost in {', '.join(f'`{q}`' for q in s['questions_losing_gold']) or 'none'}."
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> None:
    if argv[:1] == ["backfill"]:
        counts = {split: backfill(spec["store"]) for split, spec in BASELINE.items()}
        (OUT.parent / f"{OUT.name}.backfill.json").write_text(
            json.dumps(counts, indent=1) + "\n", encoding="utf-8"
        )
        print(json.dumps(counts, indent=1))
    elif argv[:1] == ["replay"]:
        report = replay()
        backfill_counts = OUT.parent / f"{OUT.name}.backfill.json"
        if backfill_counts.exists():
            report["backfill"] = json.loads(backfill_counts.read_text(encoding="utf-8"))
        OUT.with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        OUT.with_suffix(".md").write_text(render(report), encoding="utf-8")
        print(render(report))
    else:
        raise SystemExit("usage: anchor_v2_backfill.py backfill | replay")


if __name__ == "__main__":
    main(sys.argv[1:])
