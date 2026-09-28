"""Whole sessions found through their turns: does it reach more gold than turns? No calls.

    python tools/session_retrieval_offline.py                      # train150, decides
    python tools/session_retrieval_offline.py --store heldout100 \
        --questions heldout100.json --aggregate-only               # direction check

Registered in `results/prereg-session-retrieval-offline-v1.md`. Same fill rule, token
estimate and gold as `index_offline_gate.py`, whose helpers it reuses.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import write_report  # noqa: E402

BUDGETS = (2000, 4000, 8000, 16000)
DECIDING = (2000, 4000, 8000)
ARMS = ("bm25_turns", "bm25_turn_window", "bm25_sessions")
TOP_TURNS = 50
MARGIN = 0.05


def session_order(ranked_turns: list[tuple[str, int]], top: int = TOP_TURNS) -> list[str]:
    """Sessions by a DCG over their turns' ranks among the top matched turns."""
    score: dict[str, float] = defaultdict(float)
    first: dict[str, int] = {}
    for rank, (session_id, _) in enumerate(ranked_turns[:top], start=1):
        score[session_id] += 1 / math.log2(rank + 1)
        first.setdefault(session_id, rank)
    return sorted(score, key=lambda s: (-score[s], first[s]))


def analyse(store_name: str, manifest_name: str, config: str) -> dict:
    from index_offline_gate import covers_all, fill, paired, tokens

    from llm_long_term_memory.api.service import MemoryService
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.evaluation.datasets import longmemeval as lme
    from llm_long_term_memory.evaluation.manifest import load_manifest
    from llm_long_term_memory.store import external_session_id

    settings = Settings()
    manifest = load_manifest(settings.results_dir / "manifests" / manifest_name)
    instances = {i.question_id: i for i in lme.load(manifest.variant, settings.data_dir)}
    svc = MemoryService(store_name=store_name, config_path=config, read_only=True)
    store = svc.store
    rows = []
    try:
        for qid in manifest.question_ids:
            inst = instances[qid]
            gold = {
                (s.session_id, i)
                for s in inst.sessions
                for i, t in enumerate(s.turns)
                if t.has_answer
            }
            if not gold:
                continue
            cost: dict[tuple[str, int], int] = {}
            session_turns: dict[str, list[tuple[str, int]]] = defaultdict(list)
            for sid in sorted(store.session_ids_for_user(qid)):
                raw = external_session_id(sid)
                for turn in store.turns_for_session(sid):
                    key = (raw, turn.turn_index)
                    cost[key] = tokens(turn.content)
                    session_turns[raw].append(key)
            gold &= cost.keys()
            ranked = [
                (external_session_id(t.session_id), t.turn_index)
                for t in store.search_turns(qid, inst.question, limit=500)
            ]
            units = {
                "bm25_turns": [[key] for key in ranked],
                "bm25_turn_window": [[(s, i - 1), (s, i), (s, i + 1)] for s, i in ranked],
                "bm25_sessions": [session_turns[s] for s in session_order(ranked)],
            }
            row = {"question_id": qid, "question_type": inst.question_type, "arms": {}}
            for arm, arm_units in units.items():
                for budget in BUDGETS:
                    chosen = fill(arm_units, cost, budget)
                    row["arms"][f"{arm}@{budget}"] = {
                        "all": covers_all(chosen, gold),
                        "tokens": sum(cost[k] for k in chosen),
                    }
            rows.append(row)
    finally:
        svc.close()

    n = len(rows)
    table = {
        f"{arm}@{b}": sum(r["arms"][f"{arm}@{b}"]["all"] for r in rows) / max(1, n)
        for arm in ARMS
        for b in BUDGETS
    }

    def outcome(label):
        return {r["question_id"]: r["arms"][label]["all"] for r in rows}

    pairs = {
        f"{arm}@{b}": dict(
            zip(
                ("wins", "losses"),
                paired(outcome(f"{arm}@{b}"), outcome(f"bm25_turns@{b}")),
                strict=True,
            )
        )
        for arm in ARMS[1:]
        for b in BUDGETS
    }
    passing = [
        b
        for b in DECIDING
        if table[f"bm25_sessions@{b}"] - table[f"bm25_turns@{b}"] >= MARGIN
        and pairs[f"bm25_sessions@{b}"]["losses"] <= pairs[f"bm25_sessions@{b}"]["wins"]
    ]
    return {
        "store": store_name,
        "manifest": manifest_name,
        "questions": n,
        "table": table,
        "paired_vs_bm25_turns": pairs,
        "decision": {"sessions_pass_at": passing, "use_sessions": bool(passing)},
        "rows": rows,
    }


def render(result: dict) -> str:
    lines = [
        f"# Whole sessions through their turns — `{result['store']}`",
        "",
        "> Zero calls. Registered in `results/prereg-session-retrieval-offline-v1.md`.",
        "> Reach, not use. BM25 turn matching only (no turn embeddings in the store).",
        "",
        f"{result['questions']} questions with gold turns. All-gold coverage:",
        "",
        "| arm | " + " | ".join(f"{b:,}" for b in BUDGETS) + " |",
        "|---|" + "---:|" * len(BUDGETS),
    ]
    for arm in ARMS:
        lines.append(
            f"| `{arm}` | "
            + " | ".join(f"{result['table'][f'{arm}@{b}']:.1%}" for b in BUDGETS)
            + " |"
        )
    lines += [
        "",
        "Paired against `bm25_turns` at the same budget (wins / losses):",
        "",
        "| arm | " + " | ".join(f"{b:,}" for b in BUDGETS) + " |",
        "|---|" + "---:|" * len(BUDGETS),
    ]
    for arm in ARMS[1:]:
        cells = [
            f"+{result['paired_vs_bm25_turns'][f'{arm}@{b}']['wins']} / "
            f"-{result['paired_vs_bm25_turns'][f'{arm}@{b}']['losses']}"
            for b in BUDGETS
        ]
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
    decision = result["decision"]
    passes = ", ".join(f"{b:,}" for b in decision["sessions_pass_at"])
    verdict = (
        f"whole sessions (passes at {passes})" if decision["use_sessions"] else "stay with turns"
    )
    lines += ["", f"**Decision: {verdict}**"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default="train150")
    parser.add_argument("--questions", default="train150.json")
    parser.add_argument("--config", default="configs/v2.yaml")
    parser.add_argument("--aggregate-only", action="store_true", help="drop per-question rows")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args()
    result = analyse(args.store, args.questions, args.config)
    if args.aggregate_only:
        result["rows"] = []
    print(write_report(result, render, json_out=args.json_out, md_out=args.md_out), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
