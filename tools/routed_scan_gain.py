"""What does exhaustive scan buy, on the questions a router actually claims?

`exhaustive-scan.json` measured the mechanism against an oracle: it was told which
relation to scan. This measures it behind the real router, and reports the gain **only
on the slice the router routed**.

Pooling routed and abstained questions into one number would be the mistake the
`max_turns` probe caught in another form: it would dilute a ceiling with questions the
mechanism never claimed, and produce a figure that is neither the mechanism's nor the
system's. On the abstained slice the answer is by construction "top-k, unchanged", so
there is nothing to average in.

Development probes only. The held-out half is not read here.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
_SCOPE = re.compile(r"\s*Count only facts recorded under:.*$", re.IGNORECASE | re.DOTALL)


def _truth(probe: dict) -> str | None:
    found = re.search(r"relation_type is '([a-z_]+)'", probe["derivation"])
    return found.group(1) if found else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=REPO / "stores/train150.db")
    parser.add_argument("--index", type=Path, default=REPO / "stores/train150-index.npy")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out", type=Path, default=REPO / "results/analysis/routed-scan-gain.json")
    args = parser.parse_args()

    from llm_long_term_memory.embed import Encoder
    from llm_long_term_memory.retrieve.relation_router import ROUTING_CLASSES, RelationRouter
    from llm_long_term_memory.store import NumpyFlatIndex, SQLiteMemoryStore

    spec = json.loads((REPO / "results/analysis/synthesis-probes.json").read_text(encoding="utf-8"))
    split = json.loads((REPO / "results/manifests/v4-probe-split.json").read_text(encoding="utf-8"))
    development = set(split["development"])
    probes = [
        p
        for p in spec["probes"]
        if p["kind"] == "count" and _truth(p) and p["probe_id"] in development
    ]

    encoder = Encoder()
    router = RelationRouter(encoder)
    store = SQLiteMemoryStore(args.store)
    index = NumpyFlatIndex(args.index, dim=384)
    to_class = {r: c for c, rs in ROUTING_CLASSES.items() for r in rs}
    relation_of: dict[str, str] = {}
    import csv

    with open(REPO / "results/analysis/predicate-map.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            relation_of[row["predicate_raw"]] = row["relation_type"]

    rows = []
    for probe in probes:
        natural = _SCOPE.sub("", probe["question"]).strip()
        route = router.route(natural)
        needed = set(probe["evidence_memory_ids"])

        # top-k arm: exactly what the shipped retriever would surface.
        vector = np.asarray(encoder.encode([natural]), dtype=np.float32)[0]
        hits = index.search(vector, limit=len(index))
        found = []
        for memory_id, _ in hits:
            memory = store.get(memory_id)
            if memory and memory.user_id == probe["namespace"] and memory.status != "evicted":
                found.append(memory_id)
            if len(found) >= args.top_k:
                break
        top_k_complete = needed <= set(found)

        # scan arm: only when the router claimed the question.
        scan_complete = None
        scan_returned = None
        if route.routed:
            wanted = set(route.relations)
            scanned = [
                memory.id
                for memory in store.iter_active(probe["namespace"])
                if relation_of.get(memory.predicate) in wanted
            ]
            scan_returned = len(scanned)
            scan_complete = needed <= set(scanned)

        rows.append(
            {
                "probe_id": probe["probe_id"],
                "truth_class": to_class.get(_truth(probe), _truth(probe)),
                "routed": route.routed,
                "predicted_class": route.predicted_class,
                "margin": round(route.margin, 4),
                "route_correct": route.routed
                and route.predicted_class == to_class.get(_truth(probe), _truth(probe)),
                "members": len(needed),
                "top_k_complete": top_k_complete,
                "scan_complete": scan_complete,
                "scan_returned": scan_returned,
            }
        )

    store.close()
    routed = [r for r in rows if r["routed"]]
    abstained = [r for r in rows if not r["routed"]]

    def rate(subset, key):
        return sum(1 for r in subset if r[key]) / len(subset) if subset else None

    payload = {
        "schema_version": 1,
        "slice": "development count probes only; the held-out half is not read",
        "probes": len(rows),
        "routed": {
            "n": len(routed),
            "share": len(routed) / len(rows) if rows else None,
            "route_correct": rate(routed, "route_correct"),
            "top_k_complete": rate(routed, "top_k_complete"),
            "scan_complete": rate(routed, "scan_complete"),
        },
        "abstained": {
            "n": len(abstained),
            "share": len(abstained) / len(rows) if rows else None,
            "top_k_complete": rate(abstained, "top_k_complete"),
            "note": "By construction the answer here is today's retrieval, unchanged.",
        },
        "why_not_pooled": (
            "A pooled figure would dilute the mechanism's ceiling with questions it never "
            "claimed, and would be neither the mechanism's number nor the system's."
        ),
        "rows": rows,
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"development count probes: {len(rows)}")
    print(f"\n  routed slice   n={len(routed)} ({payload['routed']['share']:.0%})")
    if routed:
        print(f"    route correct        {payload['routed']['route_correct']:.0%}")
        print(f"    top-k complete       {payload['routed']['top_k_complete']:.0%}")
        print(f"    scan  complete       {payload['routed']['scan_complete']:.0%}")
    print(f"\n  abstained slice n={len(abstained)} ({payload['abstained']['share']:.0%})")
    if abstained:
        print(f"    top-k complete       {payload['abstained']['top_k_complete']:.0%}  (unchanged)")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
