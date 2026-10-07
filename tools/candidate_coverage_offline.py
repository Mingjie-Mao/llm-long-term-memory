"""All-gold reach at 4,000 tokens: v1's BM25 excerpts against candidate retrievals. No calls.

    python tools/candidate_coverage_offline.py train150 train150.json
    python tools/candidate_coverage_offline.py dev100 dev100.json

The v4 components were measured one at a time (`hybrid-turn-retrieval-offline-v1`,
`fact-key-expansion-offline-v1`); this measures BM25 + dense fusion over the fact-keyed
turn index together, as `archive_excerpts(turn_index=..., fact_keys=True)` runs it.
Memory-led turns are left out (they need the memory retrieval of each question). t1's
date notes change rendering only, so this reach is also t1-on-v4's. A question counts
when every gold turn (LongMemEval `has_answer`) is in the excerpts.

Added 2026-10-07: `v1_user_first` (`prefer_user=True`, same budget) and
`user_first_time_window` (`time_window_offline.user_first_time_window`). The v4 arm is
skipped for a store without a fact-keyed turn index.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))


def main() -> int:
    from llm_long_term_memory.config import ExperimentConfig
    from llm_long_term_memory.embed import Encoder
    from llm_long_term_memory.evaluation.datasets import longmemeval as lme
    from llm_long_term_memory.evaluation.runners.memory import _as_of
    from llm_long_term_memory.retrieve.excerpts import archive_excerpts
    from llm_long_term_memory.store import NumpyFlatIndex, SQLiteMemoryStore, external_session_id

    parser = argparse.ArgumentParser()
    parser.add_argument("store")
    parser.add_argument("manifest")
    args = parser.parse_args()
    ids = json.loads((REPO / "results/manifests" / args.manifest).read_text(encoding="utf-8"))[
        "question_ids"
    ]
    instances = {i.question_id: i for i in lme.load("s", REPO / "data")}
    encoder = Encoder(ExperimentConfig.from_yaml(REPO / "configs/fallback.yaml").models.embedder)
    from time_window_offline import user_first_time_window

    index_path = REPO / "stores" / f"{args.store}-turn-key-index"
    index = (
        NumpyFlatIndex(index_path, dim=384)
        if index_path.with_suffix(".manifest.json").exists()
        else None
    )
    store = SQLiteMemoryStore(REPO / "stores" / f"{args.store}.db", read_only=True)
    store.initialize()
    reach = defaultdict(lambda: defaultdict(list))
    try:
        for qid in ids:
            inst = instances[qid]
            gold = {
                (s.session_id, k)
                for s in inst.sessions
                for k, t in enumerate(s.turns)
                if t.has_answer
            }
            if not gold:
                continue
            arms = {
                "v1_bm25": archive_excerpts(store, qid, inst.question, 4000).turns,
                "v1_user_first": archive_excerpts(
                    store, qid, inst.question, 4000, prefer_user=True
                ).turns,
                "user_first_time_window": user_first_time_window(
                    store, qid, inst.question, _as_of(inst.question_date)
                ),
            }
            if index is not None:
                arms["v4_fused_fact_keys"] = archive_excerpts(
                    store,
                    qid,
                    inst.question,
                    4000,
                    turn_index=index,
                    query_vector=encoder.encode_one(inst.question),
                    fact_keys=True,
                ).turns
            for arm, turns in arms.items():
                got = {(external_session_id(t.session_id), t.turn_index) for t in turns}
                reach[arm][inst.question_type].append(gold <= got)
    finally:
        store.close()
    result = {
        arm: {
            "all": round(sum(sum(v) for v in by.values()) / sum(len(v) for v in by.values()), 3),
            **{t: f"{sum(v)}/{len(v)}" for t, v in sorted(by.items())},
        }
        for arm, by in reach.items()
    }
    print(json.dumps({"store": args.store, **result}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
