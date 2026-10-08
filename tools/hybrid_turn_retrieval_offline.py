"""Dense and fused turn retrieval against BM25: evidence reach at 4,000 tokens. No calls.

    python tools/hybrid_turn_retrieval_offline.py                   # train150, decides
    python tools/hybrid_turn_retrieval_offline.py --store heldout100 \
        --questions heldout100.json --aggregate-only                 # direction check

Registered in `results/prereg-hybrid-turn-retrieval-offline-v1.md`. Turn embeddings come
from the store's own local encoder and are cached beside the scratch directory given by
`--cache`; nothing is written to the store.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import write_report  # noqa: E402

BUDGET = 4000
RRF_K = 60
ARMS = ("bm25", "dense", "hybrid")
MARGIN = 0.03


def rrf(*rankings: list, k: int = RRF_K) -> list:
    """Reciprocal-rank fusion; ties broken by first appearance."""
    score: dict = defaultdict(float)
    first: dict = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            score[key] += 1 / (k + rank)
            first.setdefault(key, len(first))
    return sorted(score, key=lambda key: (-score[key], first[key]))


def analyse(store_name: str, manifest_name: str, config: str, cache: Path | None) -> dict:
    from index_offline_gate import covers_all, fill, paired, tokens

    from llm_long_term_memory.config import ExperimentConfig, Settings
    from llm_long_term_memory.embed import Encoder
    from llm_long_term_memory.evaluation.datasets import longmemeval as lme
    from llm_long_term_memory.evaluation.manifest import load_manifest
    from llm_long_term_memory.store import SQLiteMemoryStore, external_session_id

    settings = Settings()
    cfg = ExperimentConfig.from_yaml(config)
    manifest = load_manifest(settings.results_dir / "manifests" / manifest_name)
    instances = {i.question_id: i for i in lme.load(manifest.variant, settings.data_dir)}
    store = SQLiteMemoryStore(settings.store_dir / f"{store_name}.db", read_only=True)
    store.initialize()
    encoder = Encoder(cfg.models.embedder)

    turns_of: dict[str, list] = {}
    all_ids: list[str] = []
    all_texts: list[str] = []
    for qid in manifest.question_ids:
        turns = [
            t
            for sid in sorted(store.session_ids_for_user(qid))
            for t in store.turns_for_session(sid)
        ]
        turns_of[qid] = turns
        all_ids.extend(t.id for t in turns)
        all_texts.extend(t.content for t in turns)
    cache_file = cache / f"{store_name}-turn-embeddings.npy" if cache else None
    if cache_file and cache_file.exists():
        vectors = np.load(cache_file)
    else:
        vectors = encoder.encode(all_texts, show_progress=False)
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, vectors)
    row_of = {turn_id: row for row, turn_id in enumerate(all_ids)}

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
            turns = turns_of[qid]
            key = {t.id: (external_session_id(t.session_id), t.turn_index) for t in turns}
            cost = {key[t.id]: tokens(t.content) for t in turns}
            bm25 = [
                key[t.id] for t in store.search_turns(qid, inst.question, limit=500) if t.id in key
            ]
            q = encoder.encode_one(inst.question)
            sims = vectors[[row_of[t.id] for t in turns]] @ q
            dense = [key[turns[i].id] for i in np.argsort(-sims)]
            ranked = {"bm25": bm25, "dense": dense, "hybrid": rrf(bm25, dense)}
            row = {"question_id": qid, "type": inst.question_type}
            for arm, order in ranked.items():
                row[arm] = covers_all(fill([[k] for k in order], cost, BUDGET), gold)
            rows.append(row)
    finally:
        store.close()

    n = len(rows)
    coverage = {arm: sum(r[arm] for r in rows) / n for arm in ARMS}
    by_type = {
        t: {
            arm: sum(r[arm] for r in rows if r["type"] == t)
            / sum(1 for r in rows if r["type"] == t)
            for arm in ARMS
        }
        for t in sorted({r["type"] for r in rows})
    }

    def outcome(arm):
        return {r["question_id"]: r[arm] for r in rows}

    pairs = {
        arm: dict(zip(("wins", "losses"), paired(outcome(arm), outcome("bm25")), strict=True))
        for arm in ("dense", "hybrid")
    }
    qualifying = [
        arm
        for arm in ("dense", "hybrid")
        if coverage[arm] - coverage["bm25"] >= MARGIN and pairs[arm]["losses"] <= pairs[arm]["wins"]
    ]
    return {
        "store": store_name,
        "questions": n,
        "turns_embedded": len(all_ids),
        "coverage": coverage,
        "by_type": by_type,
        "paired_vs_bm25": pairs,
        "qualifying_on_this_set": qualifying,
        "rows": rows,
    }


def render(result: dict) -> str:
    c = result["coverage"]
    lines = [
        f"# Dense + BM25 turn retrieval — `{result['store']}`",
        "",
        "> Zero calls (local encoder). Registered in",
        "> `results/prereg-hybrid-turn-retrieval-offline-v1.md`. Reach, not use.",
        "",
        f"{result['questions']} questions with gold turns; {result['turns_embedded']:,} turns "
        f"embedded. All-gold coverage at {BUDGET:,} tokens:",
        "",
        "| arm | coverage | paired vs `bm25` |",
        "|---|---:|---:|",
        f"| `bm25` | {c['bm25']:.1%} | — |",
    ]
    for arm in ("dense", "hybrid"):
        p = result["paired_vs_bm25"][arm]
        lines.append(f"| `{arm}` | {c[arm]:.1%} | +{p['wins']} / -{p['losses']} |")
    lines += ["", "| type | " + " | ".join(f"`{a}`" for a in ARMS) + " |", "|---|" + "---:|" * 3]
    for t, row in result["by_type"].items():
        lines.append(f"| {t} | " + " | ".join(f"{row[a]:.0%}" for a in ARMS) + " |")
    q = result["qualifying_on_this_set"]
    lines += ["", f"Qualifying on this set: {', '.join(f'`{a}`' for a in q) or 'none'}."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default="train150")
    parser.add_argument("--questions", default="train150.json")
    parser.add_argument("--config", default="configs/fallback.yaml")
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args()
    result = analyse(args.store, args.questions, args.config, args.cache)
    if args.aggregate_only:
        result["rows"] = []
    print(write_report(result, render, json_out=args.json_out, md_out=args.md_out), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
