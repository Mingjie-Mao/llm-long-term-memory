"""Would an exhaustive scan close the counting ceiling, and what would it cost?

`count` questions ask for every member of a set. Top-k retrieval returns the *most
similar* memories, which is a different thing: measured on the probe set, only 58.3%
of count probes get every member they need into the context, so a perfect counter
still loses on the rest.

The alternative is not a better ranking. It is a different query — select every active
memory whose predicate maps to the relation being counted, and stop ranking. This tool
measures both arms offline: no provider call, no store migration, and the mapping is
applied at query time from `predicate-map.csv` so nothing is written that a later
decision would have to undo.

**This is a ceiling, not a working path.** The scan arm is told which relation to scan
by the probe's own ground truth. Routing a question to a relation is unbuilt, and it is
the half of the mechanism this measurement does not cover — a scan that fires on the
wrong relation is worse than a top-k that merely ranks badly. What is measured here is
whether the ceiling is worth building a router for.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

PROBES = REPO / "results/analysis/synthesis-probes.json"
MAP = REPO / "results/analysis/predicate-map.csv"
CONFIG = REPO / "configs/v3-phase5-compact.yaml"
# The runner's own divisor, so token counts here are comparable with a result row.
CHARS_PER_TOKEN = 4.6


def _relation_of(probe: dict) -> str:
    return probe["derivation"].split("relation_type is ")[1].split(";")[0].strip("'")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "results/analysis/exhaustive-scan.json")
    args = ap.parse_args()

    from llm_long_term_memory.config import ExperimentConfig
    from llm_long_term_memory.embed import Encoder
    from llm_long_term_memory.evaluation.runners.memory import render_memory
    from llm_long_term_memory.retrieve.hybrid import HybridRetriever
    from llm_long_term_memory.store import NumpyFlatIndex, SQLiteMemoryStore

    with open(MAP, encoding="utf-8") as handle:
        mapping = {
            row["predicate_raw"]: (row["relation_type"], row["arity"])
            for row in csv.DictReader(handle)
        }
    relation_of_predicate = {k: v[0] for k, v in mapping.items()}
    arity_of_relation = {v[0]: v[1] for v in mapping.values()}

    spec = json.loads(PROBES.read_text(encoding="utf-8"))
    probes = [p for p in spec["probes"] if p["kind"] == "count"]

    cfg = ExperimentConfig.from_yaml(CONFIG)
    store = SQLiteMemoryStore(REPO / "stores/train150.db")
    store.initialize()
    index = NumpyFlatIndex(REPO / "stores/train150-index", dim=cfg.models.embedding_dim)
    retriever = HybridRetriever(
        store,
        index,
        weights=cfg.retrieval.weights.model_dump(),
        candidate_limit=cfg.retrieval.candidate_limit,
        recency_halflife_days=cfg.retrieval.recency_halflife_days,
    )
    vectors = Encoder().encode([p["question"] for p in probes])

    def tokens(memories) -> int:
        rendered = "\n".join(render_memory(m, True) for m in memories)
        return int(len(rendered) / CHARS_PER_TOKEN)

    rows = []
    for probe, vector in zip(probes, vectors, strict=True):
        needed = set(probe["evidence_memory_ids"])
        relation = _relation_of(probe)

        ranked = [
            hit.memory
            for hit in retriever.retrieve(
                vector,
                probe["question"],
                probe["namespace"],
                temporal=True,
                limit=cfg.retrieval.top_k,
            )
        ]
        # The scan: every active memory in the namespace whose predicate maps to the
        # relation asked about. No ranking, no cut-off.
        scanned = [
            m
            for m in store.iter_active(probe["namespace"])
            if m.subject == "user" and relation_of_predicate.get(m.predicate or "") == relation
        ]

        got_ranked = {m.id for m in ranked}
        got_scan = {m.id for m in scanned}
        rows.append(
            {
                "probe_id": probe["probe_id"],
                "relation": relation,
                "members": len(needed),
                "top_k_found": len(needed & got_ranked),
                "top_k_complete": needed <= got_ranked,
                "top_k_tokens": tokens(ranked),
                "scan_found": len(needed & got_scan),
                "scan_complete": needed <= got_scan,
                "scan_returned": len(scanned),
                "scan_tokens": tokens(scanned),
            }
        )

    def block(prefix: str) -> dict:
        return {
            "complete": sum(1 for r in rows if r[f"{prefix}_complete"]),
            "complete_rate": round(sum(1 for r in rows if r[f"{prefix}_complete"]) / len(rows), 4),
            "mean_coverage": round(
                sum(r[f"{prefix}_found"] / r["members"] for r in rows) / len(rows), 4
            ),
            "median_context_tokens": statistics.median(r[f"{prefix}_tokens"] for r in rows),
        }

    # How far the mapping helps at all. Counted under the same definition as
    # `predicate-vocabulary.md` — `(user, subject, key)` over `set`-arity relations —
    # so the two numbers can be read against each other. Both statuses are reported:
    # that document counted every row, and only `active` rows can be counted in an
    # answer, which is a three-set difference and worth not having to rediscover.
    def set_sizes(active_only: bool, by_relation: bool) -> list[int]:
        clause = " WHERE status = 'active'" if active_only else ""
        keys: Counter[tuple[str, str, str]] = Counter()
        for user_id, subject, predicate in store._conn.execute(
            f"SELECT user_id, subject, predicate FROM memories{clause}"
        ):
            relation = relation_of_predicate.get(predicate or "", "other")
            if arity_of_relation.get(relation) != "set":
                continue
            keys[(user_id, subject or "", relation if by_relation else (predicate or ""))] += 1
        return list(keys.values())

    mapped_all = set_sizes(active_only=False, by_relation=True)
    mapped_active = set_sizes(active_only=True, by_relation=True)
    raw_all = set_sizes(active_only=False, by_relation=False)

    summary = {
        "schema_version": 1,
        "probes": len(rows),
        "arms": {"top_k": block("top_k"), "scan": block("scan")},
        "scan_is_a_ceiling": (
            "The scan arm is told which relation to scan by the probe's ground truth. "
            "Question-to-relation routing is unbuilt and is not measured here."
        ),
        "mapped_set_sizes": {
            "definition": "(user, subject, key) over set-arity relations",
            "sets": len(mapped_all),
            "median_members": statistics.median(mapped_all),
            "three_or_more_raw_predicate_keys": sum(1 for s in raw_all if s >= 3),
            "three_or_more_mapped_all_statuses": sum(1 for s in mapped_all if s >= 3),
            "three_or_more_mapped_active_only": sum(1 for s in mapped_active if s >= 3),
        },
        "rows": rows,
    }
    args.out.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")

    print(f"{'arm':22}{'all members in context':>24}{'mean coverage':>16}{'median tokens':>16}")
    for name, arm in summary["arms"].items():
        print(
            f"{name:22}{arm['complete']:>13} ({arm['complete_rate']:>6.1%})"
            f"{arm['mean_coverage']:>15.1%}{arm['median_context_tokens']:>16.0f}"
        )
    sets = summary["mapped_set_sizes"]
    print(
        f"\nset-arity keys with three or more members:"
        f"\n  raw predicate           {sets['three_or_more_raw_predicate_keys']:>5}"
        f"\n  mapped, all statuses    {sets['three_or_more_mapped_all_statuses']:>5}"
        f"\n  mapped, active only     {sets['three_or_more_mapped_active_only']:>5}"
        f"\n{len(mapped_all)} sets in total, median {sets['median_members']} member(s)"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
