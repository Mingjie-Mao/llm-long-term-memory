"""Does a fact stay findable as a user's history grows? Simulated growth, no calls.

    python tools/history_growth_retention.py

Retention in this system is not a clock. Decay is off and the recency weight is zero,
so a memory does not fade with time; the delayed-paraphrase check
(`results/analysis/retention-paraphrase-v1.md`) could not move retrieval by changing the
date, and said so. What does change as months pass is the **amount** of history: every
later conversation adds memories and turns that compete with the old ones for the same
top-k and the same token budget. That is the long-term failure this measures.

On a scratch copy of `train150`, each question's namespace is grown to 2x and 4x its
size by copying in the sessions, turns, memories and vectors of the next namespaces in
manifest order, re-owned by the target user, dated after its own conversations. Then,
per growth factor:

* **memory index** — is a gold-anchored memory still in the top 20?
* **raw turns** — at 4,000 tokens, does question-found BM25 still cover every gold turn?

What this cannot show: the donor history is other people's, so it competes lexically and
semantically but never contradicts the target user's facts; no temporal resolution is
run over it; and answers are not scored.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import tempfile
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import write_report  # noqa: E402

FACTORS = (1, 2, 4)
BUDGET = 4000


def _copy_store(name: str, into: Path) -> None:
    source = sqlite3.connect(f"file:{REPO / 'stores' / f'{name}.db'}?mode=ro", uri=True)
    target = sqlite3.connect(into / "growth.db")
    source.backup(target)
    source.close()
    target.close()
    for suffix in (".npy", ".ids.json", ".manifest.json"):
        src = REPO / "stores" / f"{name}-index{suffix}"
        if src.exists():
            shutil.copy(src, into / f"growth-index{suffix}")


def _grow(store, index, namespaces: list[str], factor: int) -> None:
    """Give every namespace copies of the next `factor - 1` namespaces' history."""
    from llm_long_term_memory.store import Session, Turn

    if factor == 1:
        return
    position = {memory_id: row for row, memory_id in enumerate(index.ids)}
    originals = {ns: store.iter_all(ns) for ns in namespaces}
    sessions = {
        ns: [store.get_session(sid) for sid in sorted(store.session_ids_for_user(ns))]
        for ns in namespaces
    }
    new_ids: list[str] = []
    new_rows: list[np.ndarray] = []
    for i, target in enumerate(namespaces):
        own = [s for s in sessions[target] if s is not None]
        latest = max(s.started_at for s in own)
        for step in range(1, factor):
            donor = namespaces[(i + step) % len(namespaces)]
            renamed: dict[str, str] = {}
            for offset, session in enumerate(s for s in sessions[donor] if s is not None):
                new_sid = f"{session.id}~{target}~{step}"
                renamed[session.id] = new_sid
                when = latest + timedelta(days=30 * step, hours=offset)
                store.add_session(
                    Session(
                        id=new_sid,
                        user_id=target,
                        started_at=when,
                        source="growth",
                        turns=[
                            Turn(
                                f"{t.id}~{target}~{step}",
                                new_sid,
                                t.turn_index,
                                t.role,
                                t.content,
                                when,
                            )
                            for t in store.turns_for_session(session.id)
                        ],
                    )
                )
            copies = []
            for memory in originals[donor]:
                if memory.id not in position:
                    continue
                copies.append(
                    replace(
                        memory,
                        id=f"{memory.id}~{target}~{step}",
                        user_id=target,
                        source_session_id=renamed.get(memory.source_session_id or ""),
                        superseded_by=None,
                    )
                )
                new_rows.append(index._vectors[position[memory.id]])
            store.add_memories(copies)
            new_ids.extend(m.id for m in copies)
    index.add(new_ids, np.vstack(new_rows).astype(np.float32))


def measure(store_name: str, manifest_name: str, config: str) -> dict:
    from index_offline_gate import covers_all, covers_any, fill, tokens

    from llm_long_term_memory.api.service import MemoryService
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.evaluation.datasets import longmemeval as lme
    from llm_long_term_memory.evaluation.manifest import load_manifest
    from llm_long_term_memory.store import external_session_id

    base = Settings()
    manifest = load_manifest(base.results_dir / "manifests" / manifest_name)
    instances = {i.question_id: i for i in lme.load(manifest.variant, base.data_dir)}
    namespaces = list(manifest.question_ids)

    results: dict[int, dict] = {}
    for factor in FACTORS:
        with tempfile.TemporaryDirectory() as scratch:
            scratch_dir = Path(scratch)
            _copy_store(store_name, scratch_dir)
            settings = base.model_copy(update={"store_dir": scratch_dir})
            svc = MemoryService(store_name="growth", config_path=config, settings=settings)
            try:
                _grow(svc.store, svc.index, namespaces, factor)
                memory_hits = raw_hits = scored = 0
                sizes = []
                for qid in namespaces:
                    inst = instances[qid]
                    own = {
                        external_session_id(sid): sid
                        for sid in svc.store.session_ids_for_user(qid)
                        if "~" not in sid
                    }
                    gold = {
                        (own[s.session_id], i)
                        for s in inst.sessions
                        if s.session_id in own
                        for i, t in enumerate(s.turns)
                        if t.has_answer
                    }
                    if not gold:
                        continue
                    scored += 1
                    sizes.append(len(svc.store.iter_all(qid)))
                    hits = svc.search(qid, inst.question, limit=20).memories
                    anchors = [
                        (h.memory.source_session_id, h.memory.source_turn_index)
                        for h in hits
                        if h.memory.source_session_id and h.memory.source_turn_index is not None
                    ]
                    memory_hits += covers_any(anchors, gold)
                    turns = svc.store.search_turns(qid, inst.question, limit=500)
                    cost = {(t.session_id, t.turn_index): tokens(t.content) for t in turns}
                    chosen = fill([[(t.session_id, t.turn_index)] for t in turns], cost, BUDGET)
                    raw_hits += covers_all(chosen, gold)
            finally:
                svc.close()
        results[factor] = {
            "questions": scored,
            "median_memories_per_user": float(np.median(sizes)) if sizes else 0,
            "memory_anchor_hit_top20": memory_hits / max(1, scored),
            "raw_turns_all_gold_at_4000": raw_hits / max(1, scored),
        }
    return {"store": store_name, "manifest": manifest_name, "config": config, "by_factor": results}


def render(result: dict) -> str:
    lines = [
        "# Retention as history grows — simulated, zero calls",
        "",
        f"> Scratch copies of `{result['store']}`; the store itself is not written. Growth",
        "> copies other users' history into each namespace, dated after the user's own. It",
        "> competes for rank and budget but never contradicts; answers are not scored.",
        "",
        "| history | median memories per user | gold-anchored memory in top 20 "
        "| raw turns cover all gold at 4,000 tokens |",
        "|---|---:|---:|---:|",
    ]
    for factor, row in result["by_factor"].items():
        lines.append(
            f"| {factor}x | {row['median_memories_per_user']:,.0f} | "
            f"{row['memory_anchor_hit_top20']:.1%} | {row['raw_turns_all_gold_at_4000']:.1%} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default="train150")
    parser.add_argument("--questions", default="train150.json")
    parser.add_argument("--config", default="configs/v2.yaml")
    parser.add_argument(
        "--json-out", type=Path, default=REPO / "results/analysis/history-growth-retention-v1.json"
    )
    parser.add_argument(
        "--md-out", type=Path, default=REPO / "results/analysis/history-growth-retention-v1.md"
    )
    args = parser.parse_args()
    result = measure(args.store, args.questions, args.config)
    result["by_factor"] = {str(k): v for k, v in result["by_factor"].items()}
    print(write_report(result, render, json_out=args.json_out, md_out=args.md_out), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
