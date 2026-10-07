"""Facts as keys for raw turns: does indexing a turn under its facts find more gold? No calls.

    python tools/fact_key_expansion_offline.py --cache <dir with <store>-turn-embeddings.npy>
    python tools/fact_key_expansion_offline.py --store heldout100 --questions heldout100.json \
        --cache <dir> --aggregate-only

Registered in `results/prereg-fact-key-expansion-offline-v1.md`. The cached plain-turn
embeddings are the ones `hybrid_turn_retrieval_offline.py` wrote, in its order; only turns
whose key changes are re-embedded.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import write_report  # noqa: E402

BUDGET = 4000
ARMS = ("plain", "expanded", "expanded_stored", "expanded_dense_only")
MARGIN = 0.02
_TOKEN = re.compile(r"[a-z0-9]+")


def bm25_rank(query: str, keys: dict[str, str], k1: float = 1.2, b: float = 0.75) -> list[str]:
    """Okapi BM25 over one namespace's keys; turns with no shared term are left out."""
    docs = {tid: _TOKEN.findall(text.lower()) for tid, text in keys.items()}
    n = len(docs)
    avg = sum(len(d) for d in docs.values()) / max(1, n)
    df: Counter = Counter()
    for tokens in docs.values():
        df.update(set(tokens))
    terms = set(_TOKEN.findall(query.lower()))
    scores = {}
    for tid, tokens in docs.items():
        tf = Counter(tokens)
        score = 0.0
        for term in terms & tf.keys():
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * len(tokens) / avg))
        if score > 0:
            scores[tid] = score
    return sorted(scores, key=lambda tid: -scores[tid])


def analyse(store_name: str, manifest_name: str, config: str, cache: Path) -> dict:
    from index_offline_gate import covers_all, fill, paired, tokens

    from llm_long_term_memory.config import ExperimentConfig, Settings
    from llm_long_term_memory.embed import Encoder
    from llm_long_term_memory.evaluation.datasets import longmemeval as lme
    from llm_long_term_memory.evaluation.manifest import load_manifest
    from llm_long_term_memory.ingest.provenance import source_span_for_v2
    from llm_long_term_memory.retrieve.excerpts import fuse
    from llm_long_term_memory.store import SQLiteMemoryStore, external_session_id

    settings = Settings()
    cfg = ExperimentConfig.from_yaml(config)
    manifest = load_manifest(settings.results_dir / "manifests" / manifest_name)
    instances = {i.question_id: i for i in lme.load(manifest.variant, settings.data_dir)}
    store = SQLiteMemoryStore(settings.store_dir / f"{store_name}.db", read_only=True)
    store.initialize()
    encoder = Encoder(cfg.models.embedder)

    turns_of: dict[str, list] = {}
    order: list[str] = []
    for qid in manifest.question_ids:
        turns = [
            t
            for sid in sorted(store.session_ids_for_user(qid))
            for t in store.turns_for_session(sid)
        ]
        turns_of[qid] = turns
        order.extend(t.id for t in turns)
    plain_vectors = np.load(cache / f"{store_name}-turn-embeddings.npy")
    if len(plain_vectors) != len(order):
        raise SystemExit("cached embeddings do not match this store's turn order")
    row_of = {tid: i for i, tid in enumerate(order)}

    rows = []
    anchored_turns = Counter()
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
            by_key = {(t.session_id, t.turn_index): t for t in turns}
            sessions = {s.session_id: s for s in inst.sessions}
            facts = {"expanded": defaultdict(list), "expanded_stored": defaultdict(list)}
            facts["expanded_dense_only"] = facts["expanded"]
            for memory in store.iter_all(qid):
                if not memory.source_session_id:
                    continue
                session = sessions.get(external_session_id(memory.source_session_id))
                if memory.source_turn_index is not None:
                    turn = by_key.get((memory.source_session_id, memory.source_turn_index))
                    if turn:
                        facts["expanded_stored"][turn.id].append(memory.content)
                if session is not None:
                    span = source_span_for_v2(memory.content, session)
                    if span is not None:
                        turn = by_key.get((memory.source_session_id, span[0]))
                        if turn:
                            facts["expanded"][turn.id].append(memory.content)
            anchored_turns["expanded"] += len(facts["expanded"])
            key = {(t.id): (external_session_id(t.session_id), t.turn_index) for t in turns}
            cost = {key[t.id]: tokens(t.content) for t in turns}
            q = encoder.encode_one(inst.question)
            row = {"question_id": qid, "type": inst.question_type}
            for arm in ARMS:
                texts = {
                    t.id: t.content
                    if arm == "plain" or not facts[arm].get(t.id)
                    else t.content + "\n" + " ".join(facts[arm][t.id])
                    for t in turns
                }
                vectors = plain_vectors[[row_of[t.id] for t in turns]].copy()
                changed = [i for i, t in enumerate(turns) if texts[t.id] != t.content]
                if changed:
                    vectors[changed] = encoder.encode([texts[turns[i].id] for i in changed])
                dense = [turns[i].id for i in np.argsort(-(vectors @ q))]
                # The dense-only arm keys BM25 on the plain turn, which is what a store can
                # do without a second full-text index: only the turn vectors change.
                lexical = (
                    {t.id: t.content for t in turns} if arm == "expanded_dense_only" else texts
                )
                ranked = fuse(bm25_rank(inst.question, lexical), dense)
                row[arm] = covers_all(fill([[key[tid]] for tid in ranked], cost, BUDGET), gold)
            rows.append(row)
    finally:
        store.close()

    n = len(rows)
    coverage = {arm: sum(r[arm] for r in rows) / n for arm in ARMS}

    def outcome(arm):
        return {r["question_id"]: r[arm] for r in rows}

    pairs = {
        arm: dict(zip(("wins", "losses"), paired(outcome(arm), outcome("plain")), strict=True))
        for arm in ARMS[1:]
    }
    by_type = {
        t: {
            a: sum(r[a] for r in rows if r["type"] == t) / sum(1 for r in rows if r["type"] == t)
            for a in ARMS
        }
        for t in sorted({r["type"] for r in rows})
    }
    qualifies = (
        coverage["expanded"] - coverage["plain"] >= MARGIN
        and pairs["expanded"]["losses"] <= pairs["expanded"]["wins"]
    )
    return {
        "store": store_name,
        "questions": n,
        "coverage": coverage,
        "paired_vs_plain": pairs,
        "by_type": by_type,
        "expanded_qualifies_on_this_set": qualifies,
        "rows": rows,
    }


def render(result: dict) -> str:
    c, p = result["coverage"], result["paired_vs_plain"]
    lines = [
        f"# Facts as keys for raw turns — `{result['store']}`",
        "",
        "> Zero calls (local encoder). Registered in",
        "> `results/prereg-fact-key-expansion-offline-v1.md`. Reach, not use.",
        "",
        f"{result['questions']} questions with gold turns. All-gold coverage at {BUDGET:,} tokens:",
        "",
        "| arm | coverage | paired vs `plain` |",
        "|---|---:|---:|",
        f"| `plain` | {c['plain']:.1%} | — |",
    ]
    for arm in ARMS[1:]:
        lines.append(f"| `{arm}` | {c[arm]:.1%} | +{p[arm]['wins']} / -{p[arm]['losses']} |")
    lines += [
        "",
        "| type | " + " | ".join(f"`{a}`" for a in ARMS) + " |",
        "|---|" + "---:|" * len(ARMS),
    ]
    for t, row in result["by_type"].items():
        lines.append(f"| {t} | " + " | ".join(f"{row[a]:.0%}" for a in ARMS) + " |")
    lines += ["", f"`expanded` qualifies on this set: {result['expanded_qualifies_on_this_set']}."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default="train150")
    parser.add_argument("--questions", default="train150.json")
    parser.add_argument("--config", default="configs/fallback.yaml")
    parser.add_argument("--cache", type=Path, required=True)
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
