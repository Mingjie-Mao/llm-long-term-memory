"""Memory as an index: does it put the gold turns in front of the answerer? No calls.

    python tools/index_offline_gate.py --json-out results/analysis/memory-as-index-offline-v1.json \
        --md-out results/analysis/memory-as-index-offline-v1.md

Registered in `results/prereg-memory-as-index-offline-v1.md`. v2 hands the answerer
extracted facts; this measures what it would receive if the facts were used only to
*locate* source turns, and compares that with searching the raw archive by the
question alone, at the same token budget. Gold evidence is LongMemEval's own
`has_answer` turn flags, so nothing is hand-labelled.

Local encoder, the real retriever, the real store. Reach, not use: a pass permits an
answer experiment and says nothing about its size.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import write_report  # noqa: E402

CHARS_PER_TOKEN = 4.6
BUDGETS = (1000, 2000, 4000, 8000)
ARMS = ("mem_turn", "mem_window", "mem_session_bm25", "bm25_turns", "mem_session_whole")
MEMORY_ARMS = ("mem_turn", "mem_window")
GATE_BUDGET = 4000
GATE_COVERAGE = 0.80

TurnKey = tuple[str, int]


def tokens(text: str) -> int:
    return max(1, int(len(text) / CHARS_PER_TOKEN)) if text else 0


def fill(
    units: Iterable[Sequence[TurnKey]], cost: dict[TurnKey, int], budget: int
) -> list[TurnKey]:
    """Take units in order until the budget is spent, as the hydrator does.

    A unit is a group of turns that arrive together (a turn and its neighbours, or a
    whole session). Turns already included cost nothing again, so overlapping windows
    are charged only for what they add. A unit that would overflow is skipped and
    filling continues: a later, smaller unit may still fit.
    """
    chosen: list[TurnKey] = []
    have: set[TurnKey] = set()
    spent = 0
    for unit in units:
        new = [key for key in dict.fromkeys(unit) if key not in have and key in cost]
        if not new:
            continue
        extra = sum(cost[key] for key in new)
        if spent + extra > budget:
            continue
        chosen.extend(new)
        have.update(new)
        spent += extra
    return chosen


def covers_all(chosen: Iterable[TurnKey], gold: set[TurnKey]) -> bool:
    return bool(gold) and gold <= set(chosen)


def covers_any(chosen: Iterable[TurnKey], gold: set[TurnKey]) -> bool:
    return bool(gold & set(chosen))


def paired(a: dict[str, bool], b: dict[str, bool]) -> tuple[int, int]:
    """(wins, losses) of `a` against `b` over the questions both scored."""
    shared = a.keys() & b.keys()
    return (
        sum(1 for q in shared if a[q] and not b[q]),
        sum(1 for q in shared if b[q] and not a[q]),
    )


def _units(arm, anchors, session_order, session_turns, bm25_local, bm25_global):
    if arm == "mem_turn":
        return [[key] for key in anchors]
    if arm == "mem_window":
        return [[(sid, index - 1), (sid, index), (sid, index + 1)] for sid, index in anchors]
    if arm == "mem_session_bm25":
        return [[key] for key in bm25_local]
    if arm == "bm25_turns":
        return [[key] for key in bm25_global]
    if arm == "mem_session_whole":
        return [session_turns[sid] for sid in session_order]
    raise ValueError(arm)


def analyse(store_name: str, manifest_name: str, config: str, limit: int = 50) -> dict:
    from llm_long_term_memory.api.service import MemoryService
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.evaluation.datasets import longmemeval as lme
    from llm_long_term_memory.evaluation.manifest import load_manifest
    from llm_long_term_memory.store import external_session_id

    settings = Settings()
    manifest = load_manifest(settings.results_dir / "manifests" / manifest_name)
    instances = {i.question_id: i for i in lme.load(manifest.variant, settings.data_dir)}
    svc = MemoryService(store_name=store_name, config_path=config)
    store = svc.store

    rows: list[dict] = []
    excluded_no_gold: list[str] = []
    content_mismatch = 0
    try:
        for qid in manifest.question_ids:
            inst = instances[qid]
            gold_text: dict[TurnKey, str] = {}
            for session in inst.sessions:
                for index, turn in enumerate(session.turns):
                    if turn.has_answer:
                        gold_text[(session.session_id, index)] = turn.content
            if not gold_text:
                excluded_no_gold.append(qid)
                continue

            # Every turn in the namespace, keyed the way the corpus keys it.
            cost: dict[TurnKey, int] = {}
            internal: dict[str, str] = {}
            session_turns: dict[str, list[TurnKey]] = defaultdict(list)
            for session_id in sorted(store.session_ids_for_user(qid)):
                raw_sid = external_session_id(session_id)
                internal[raw_sid] = session_id
                for turn in store.turns_for_session(session_id):
                    key = (raw_sid, turn.turn_index)
                    cost[key] = tokens(turn.content)
                    session_turns[raw_sid].append(key)
                    if key in gold_text and gold_text[key] != turn.content:
                        content_mismatch += 1
            gold = set(gold_text) & cost.keys()

            hits = svc.search(qid, inst.question, limit=limit).memories
            anchors: list[TurnKey] = []
            session_order: list[str] = []
            for hit in hits:
                memory = hit.memory
                if not memory.source_session_id:
                    continue
                sid = external_session_id(memory.source_session_id)
                if sid not in session_order:
                    session_order.append(sid)
                if memory.source_turn_index is not None:
                    anchors.append((sid, memory.source_turn_index))
            anchors = list(dict.fromkeys(anchors))

            def _keys(turns) -> list[TurnKey]:
                return [(external_session_id(t.session_id), t.turn_index) for t in turns]

            bm25_global = _keys(store.search_turns(qid, inst.question, limit=500))
            bm25_local = _keys(
                store.search_turns_in_sessions(
                    qid,
                    inst.question,
                    {internal[s] for s in session_order if s in internal},
                    limit=500,
                )
            )

            row = {
                "question_id": qid,
                "question_type": inst.question_type,
                "gold_turns": len(gold),
                # The anchors of the first 20 hits — what v2 itself ranks into its
                # context — not the first 20 distinct anchors.
                "anchor_hit_top20": covers_any(
                    [
                        (
                            external_session_id(h.memory.source_session_id),
                            h.memory.source_turn_index,
                        )
                        for h in hits[:20]
                        if h.memory.source_session_id and h.memory.source_turn_index is not None
                    ],
                    gold,
                ),
                "anchor_hit_top50": covers_any(anchors, gold),
                "arms": {},
            }

            for arm in ARMS:
                units = _units(arm, anchors, session_order, session_turns, bm25_local, bm25_global)
                for budget in BUDGETS:
                    chosen = fill(units, cost, budget)
                    row["arms"][f"{arm}@{budget}"] = {
                        "all": covers_all(chosen, gold),
                        "any": covers_any(chosen, gold),
                        "tokens": sum(cost[k] for k in chosen),
                    }
            rows.append(row)
    finally:
        svc.close()

    return summarise(rows, excluded_no_gold, content_mismatch, store_name, manifest_name, config)


def summarise(rows, excluded_no_gold, content_mismatch, store_name, manifest_name, config) -> dict:
    n = len(rows)
    table = {}
    for arm in ARMS:
        for budget in BUDGETS:
            label = f"{arm}@{budget}"
            cells = [row["arms"][label] for row in rows]
            table[label] = {
                "all": sum(c["all"] for c in cells) / max(1, n),
                "any": sum(c["any"] for c in cells) / max(1, n),
                "median_tokens": statistics.median(c["tokens"] for c in cells) if cells else 0,
            }

    def outcome(label):
        return {row["question_id"]: row["arms"][label]["all"] for row in rows}

    pairs = {}
    for arm in ARMS:
        if arm == "bm25_turns":
            continue
        for budget in BUDGETS:
            wins, losses = paired(outcome(f"{arm}@{budget}"), outcome(f"bm25_turns@{budget}"))
            pairs[f"{arm}@{budget}"] = {"wins": wins, "losses": losses}

    by_type: dict[str, dict] = {}
    types = Counter(row["question_type"] for row in rows)
    for qtype, count in sorted(types.items()):
        subset = [row for row in rows if row["question_type"] == qtype]
        by_type[qtype] = {"n": count}
        for arm in ARMS:
            label = f"{arm}@{GATE_BUDGET}"
            by_type[qtype][arm] = sum(r["arms"][label]["all"] for r in subset) / count

    best = max(MEMORY_ARMS, key=lambda arm: table[f"{arm}@{GATE_BUDGET}"]["all"])
    best_pair = pairs[f"{best}@{GATE_BUDGET}"]
    gate = {
        "best_memory_arm": best,
        "coverage_at_gate": table[f"{best}@{GATE_BUDGET}"]["all"],
        "coverage_ok": table[f"{best}@{GATE_BUDGET}"]["all"] >= GATE_COVERAGE,
        "not_worse_than_bm25": best_pair["losses"] <= best_pair["wins"],
    }
    gate["pass"] = gate["coverage_ok"] and gate["not_worse_than_bm25"]
    ceiling = table[f"{best}@{BUDGETS[-1]}"]["all"]
    gate["carried_budget"] = next(
        b for b in BUDGETS if table[f"{best}@{b}"]["all"] >= ceiling - 0.02
    )

    return {
        "store": store_name,
        "manifest": manifest_name,
        "config": config,
        "questions_scored": n,
        "excluded_no_gold": excluded_no_gold,
        "gold_content_mismatches": content_mismatch,
        "anchor_hit_top20": sum(r["anchor_hit_top20"] for r in rows) / max(1, n),
        "anchor_hit_top50": sum(r["anchor_hit_top50"] for r in rows) / max(1, n),
        "table": table,
        "paired_vs_bm25": pairs,
        "by_type_at_gate_budget": by_type,
        "gate": gate,
        "rows": rows,
    }


def render(result: dict) -> str:
    gate = result["gate"]
    lines = [
        "# Memory as an index — offline evidence reach v1",
        "",
        "> DEVELOPMENT, zero calls. Registered in `results/prereg-memory-as-index-offline-v1.md`.",
        "> Reach, not use: coverage means the gold turn is in the context, not that it is",
        "> answered correctly.",
        "",
        f"Store `{result['store']}`, manifest `{result['manifest']}`, retrieval "
        f"`{result['config']}`. {result['questions_scored']} questions with flagged gold turns; "
        f"{len(result['excluded_no_gold'])} without any were excluded. "
        f"Gold-turn text mismatches between corpus and store: {result['gold_content_mismatches']}.",
        "",
        f"Index hit rate — a top-20 memory is anchored to a gold turn: "
        f"**{result['anchor_hit_top20']:.1%}**; top 50: {result['anchor_hit_top50']:.1%}.",
        "",
        "## All-gold coverage by budget",
        "",
        "| arm | " + " | ".join(f"{b:,}" for b in BUDGETS) + " |",
        "|---|" + "---:|" * len(BUDGETS),
    ]
    for arm in ARMS:
        cells = []
        for b in BUDGETS:
            cell = result["table"][f"{arm}@{b}"]
            cells.append(f"{cell['all']:.1%} ({cell['median_tokens']:,.0f})")
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Cells: share of questions with every gold turn in context (median tokens used).",
        "",
        "## Paired against `bm25_turns` at the same budget (all-gold)",
        "",
        "| arm | " + " | ".join(f"{b:,}" for b in BUDGETS) + " |",
        "|---|" + "---:|" * len(BUDGETS),
    ]
    for arm in ARMS:
        if arm == "bm25_turns":
            continue
        cells = [
            f"+{result['paired_vs_bm25'][f'{arm}@{b}']['wins']} / "
            f"−{result['paired_vs_bm25'][f'{arm}@{b}']['losses']}"  # noqa: RUF001
            for b in BUDGETS
        ]
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"## By question type at {GATE_BUDGET:,} tokens (all-gold)",
        "",
        "| type | n | " + " | ".join(f"`{a}`" for a in ARMS) + " |",
        "|---|---:|" + "---:|" * len(ARMS),
    ]
    for qtype, cells in result["by_type_at_gate_budget"].items():
        lines.append(
            f"| {qtype} | {cells['n']} | " + " | ".join(f"{cells[a]:.0%}" for a in ARMS) + " |"
        )
    lines += [
        "",
        "## Registered gate",
        "",
        f"- best memory arm at {GATE_BUDGET:,}: `{gate['best_memory_arm']}`, "
        f"coverage {gate['coverage_at_gate']:.1%} (needs ≥ {GATE_COVERAGE:.0%}): "
        f"{'pass' if gate['coverage_ok'] else 'fail'}",
        f"- not worse than `bm25_turns` at {GATE_BUDGET:,}: "
        f"{'pass' if gate['not_worse_than_bm25'] else 'fail'}",
        f"- budget carried forward: {gate['carried_budget']:,}",
        "",
        f"**Decision: {'PASS — an answer experiment is justified' if gate['pass'] else 'STOP'}**",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default="train150")
    parser.add_argument("--questions", default="train150.json")
    parser.add_argument("--config", default="configs/v2.yaml")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args()
    if args.store != "train150":
        print("this gate decides on train150 only; other stores may check the code, not decide")
    result = analyse(args.store, args.questions, args.config)
    print(write_report(result, render, json_out=args.json_out, md_out=args.md_out), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
