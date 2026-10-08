"""Does letting memory-led turns vote hold turn retrieval up as history grows? No calls.

    python tools/growth_fusion_offline.py --cache <dir with train150-turn-embeddings.npy>

Registered in `results/prereg-growth-fusion-offline-v1.md`. Grows scratch copies of the
store with `history_growth_retention._grow`; dense vectors of copied turns are those of
the turns they copy, from the cache `hybrid_turn_retrieval_offline.py` wrote.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import write_report  # noqa: E402

FACTORS = (1, 4)
BUDGET = 4000
ARMS = ("bm25", "hybrid", "hybrid_memory")
MARGIN_4X = 0.02
TOLERANCE_1X = 0.01


def measure(store_name: str, manifest_name: str, config: str, cache: Path) -> dict:
    from history_growth_retention import _copy_store, _grow
    from index_offline_gate import covers_all, fill, tokens

    from llm_long_term_memory.api.service import MemoryService
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.evaluation.datasets import longmemeval as lme
    from llm_long_term_memory.evaluation.manifest import load_manifest
    from llm_long_term_memory.retrieve.excerpts import fuse
    from llm_long_term_memory.store import SQLiteMemoryStore, external_session_id

    base = Settings()
    manifest = load_manifest(base.results_dir / "manifests" / manifest_name)
    instances = {i.question_id: i for i in lme.load(manifest.variant, base.data_dir)}
    namespaces = list(manifest.question_ids)

    original = SQLiteMemoryStore(base.store_dir / f"{store_name}.db", read_only=True)
    original.initialize()
    order = [
        t.id
        for qid in namespaces
        for sid in sorted(original.session_ids_for_user(qid))
        for t in original.turns_for_session(sid)
    ]
    original.close()
    vectors = np.load(cache / f"{store_name}-turn-embeddings.npy")
    if len(vectors) != len(order):
        raise SystemExit("cached embeddings do not match this store's turn order")
    row_of = {tid: i for i, tid in enumerate(order)}

    results = {}
    for factor in FACTORS:
        with tempfile.TemporaryDirectory() as scratch:
            scratch_dir = Path(scratch)
            _copy_store(store_name, scratch_dir)
            settings = base.model_copy(update={"store_dir": scratch_dir})
            svc = MemoryService(store_name="growth", config_path=config, settings=settings)
            hits = dict.fromkeys(ARMS, 0)
            scored = 0
            try:
                _grow(svc.store, svc.index, namespaces, factor)
                for qid in namespaces:
                    inst = instances[qid]
                    turns = [
                        t
                        for sid in sorted(svc.store.session_ids_for_user(qid))
                        for t in svc.store.turns_for_session(sid)
                    ]
                    by_key = {(t.session_id, t.turn_index): t for t in turns}
                    gold_raw = {
                        (s.session_id, i)
                        for s in inst.sessions
                        for i, t in enumerate(s.turns)
                        if t.has_answer
                    }
                    gold = {
                        t.id
                        for t in turns
                        if "~" not in t.session_id
                        and (external_session_id(t.session_id), t.turn_index) in gold_raw
                    }
                    if not gold:
                        continue
                    scored += 1
                    cost = {t.id: tokens(t.content) for t in turns}
                    q = svc.encoder.encode_one(inst.question)
                    matrix = vectors[[row_of[t.id.split("~")[0]] for t in turns]]
                    dense = [turns[i].id for i in np.argsort(-(matrix @ q))]
                    bm25 = [t.id for t in svc.store.search_turns(qid, inst.question, limit=500)]
                    memory = []
                    for hit in svc.search(qid, inst.question, limit=20).memories:
                        m = hit.memory
                        turn = by_key.get((m.source_session_id, m.source_turn_index))
                        if turn is not None and turn.id not in memory:
                            memory.append(turn.id)
                    ranked = {
                        "bm25": bm25,
                        "hybrid": fuse(bm25, dense),
                        "hybrid_memory": fuse(bm25, dense, memory),
                    }
                    for arm, order_ids in ranked.items():
                        chosen = fill([[tid] for tid in order_ids], cost, BUDGET)
                        hits[arm] += covers_all(chosen, gold)
            finally:
                svc.close()
        results[str(factor)] = {arm: hits[arm] / max(1, scored) for arm in ARMS} | {
            "questions": scored
        }
    one, four = results["1"], results["4"]
    adopt = (
        four["hybrid_memory"] - four["hybrid"] >= MARGIN_4X
        and one["hybrid_memory"] >= one["hybrid"] - TOLERANCE_1X
    )
    return {"store": store_name, "by_factor": results, "add_memory_list": adopt}


def render(result: dict) -> str:
    lines = [
        f"# Memory-led turns in the fusion as history grows — `{result['store']}`",
        "",
        "> Zero calls. Registered in `results/prereg-growth-fusion-offline-v1.md`.",
        "> Scratch copies grown with other users' history. Reach, not use.",
        "",
        "| history | " + " | ".join(f"`{a}`" for a in ARMS) + " |",
        "|---|---:|---:|---:|",
    ]
    for factor, row in result["by_factor"].items():
        lines.append(f"| {factor}x | " + " | ".join(f"{row[a]:.1%}" for a in ARMS) + " |")
    verdict = (
        "add the memory list to the fusion" if result["add_memory_list"] else "keep BM25 + dense"
    )
    lines += ["", f"**Decision: {verdict}**"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default="train150")
    parser.add_argument("--questions", default="train150.json")
    parser.add_argument("--config", default="configs/v2.yaml")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args()
    result = measure(args.store, args.questions, args.config, args.cache)
    print(write_report(result, render, json_out=args.json_out, md_out=args.md_out), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
