"""Does deduplication compare facts across namespaces? Measured on a built store.

Retrieval is namespace-safe by an explicit filter: `HybridRetriever` searches the whole
index and then keeps only `memory.user_id == namespace`
(`src/llm_long_term_memory/retrieve/hybrid.py`). Deduplication is not. `Deduplicator._neighbours`
calls `index.search(vector, limit=max_neighbours)` on the same shared index and adjudicates
whatever comes back, with no namespace condition — so a fact extracted from one user's
history can be adjudicated against, and dropped as a DUPLICATE of, a fact belonging to a
different one.

This counts how reachable that is, on a store that already exists, with no provider call:
for a sample of memories, how many of their nearest neighbours above the dedup threshold
belong to another namespace.

It measures the *final* store, where every namespace is present. During ingestion the
store was smaller, so the counts here are not the counts that ran. The direction is what
this establishes: whether the cross-namespace path is rare or ordinary.

    python3 tools/dedup_namespace_probe.py --store stores/test100.db
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results/analysis/dedup-namespace-leak.json"

THRESHOLD = 0.92
"""`ingest.dedupe_similarity_threshold` in every committed config."""

MAX_NEIGHBOURS = 3
"""`Deduplicator.max_neighbours`: dedup only ever looks at this many index hits."""


def probe(store_path: Path, index_stem: Path, sample: int, seed: int, dim: int = 384) -> dict:
    import numpy as np

    from llm_long_term_memory.store import NumpyFlatIndex, SQLiteMemoryStore

    store = SQLiteMemoryStore(store_path, read_only=True)
    store.initialize()
    index = NumpyFlatIndex(index_stem, dim=dim)
    ids = list(index._ids)
    vectors = index._vectors
    owner = {
        memory.id: memory.user_id
        for namespace in store.user_ids()
        for memory in store.iter_all(namespace)
    }
    namespaces = len(store.user_ids())
    store.close()

    random.seed(seed)
    chosen = random.sample(range(len(ids)), min(sample, len(ids)))
    cross = same = unowned = 0
    crowded_out = 0
    for position in chosen:
        scores = vectors @ vectors[position]
        top = np.argsort(-scores)[: MAX_NEIGHBOURS + 1]
        foreign_in_window = 0
        for other in top:
            if ids[other] == ids[position] or scores[other] < THRESHOLD:
                continue
            mine, theirs = owner.get(ids[position]), owner.get(ids[other])
            if theirs is None or mine is None:
                unowned += 1
            elif theirs == mine:
                same += 1
            else:
                cross += 1
                foreign_in_window += 1
        if foreign_in_window >= MAX_NEIGHBOURS:
            crowded_out += 1

    return {
        "name": "dedup-namespace-leak",
        "store": str(store_path),
        "provider_calls": 0,
        "memories": len(ids),
        "namespaces": namespaces,
        "threshold": THRESHOLD,
        "max_neighbours": MAX_NEIGHBOURS,
        "sampled": len(chosen),
        "neighbours_above_threshold": {
            "same_namespace": same,
            "cross_namespace": cross,
            "owner_unknown": unowned,
        },
        "samples_whose_whole_window_was_foreign": crowded_out,
        "reading": (
            "Every cross-namespace pair here is a pair the deduplicator would have "
            "adjudicated had both memories been in the store at the time, and a DUPLICATE "
            "verdict on one drops a fact from a conversation that never contained its "
            "supposed duplicate. A window filled by foreign memories also crowds out the "
            "within-namespace duplicates dedup exists to catch."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default="stores/test100.db")
    parser.add_argument("--index", default=None, help="index stem; defaults to <store>-index")
    parser.add_argument("--sample", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    store_path = Path(args.store)
    default_index = store_path.with_name(store_path.stem + "-index")
    index_stem = Path(args.index) if args.index else default_index
    payload = probe(store_path, index_stem, args.sample, args.seed)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nwritten: {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
