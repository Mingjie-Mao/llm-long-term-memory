"""v4.0 changes the answerer only. This proves it, from the rows themselves.

The registered comparison is `v3.3 retrieval + v3.3 answerer` against `v3.3 retrieval +
v4.0 answerer`. If retrieval differs at all between the arms, the difference in accuracy
has two possible causes and the run cannot answer the question it was paid for.

Asserting that in a document is not the same as checking it. v3.2 was reported `STOP`
precisely because two things moved at once, and the way that was noticed was a number
nobody could attribute — after the quota was spent. This checks the same property before
anything is interpreted, and it is free: the rows already carry `retrieved_ids`.

Exit 2 on any divergence. The run is not to be read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Everything a change to the answerer must leave untouched. `context_tokens` is not on
# the list: v4 renders supersession chains differently, so the same memories can be
# spelled with a different number of tokens. What must not move is *which* memories.
INVARIANT = ("retrieved_ids", "evidence_found", "evidence_needed", "context_complete")
IDENTITY = ("question", "kind", "namespace", "gold", "store_fingerprint")


def _rows(path: Path) -> dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                probe_id = row.get("probe_id")
                if not isinstance(probe_id, str) or not probe_id or probe_id in out:
                    raise ValueError(f"{path}: missing or duplicate probe_id")
                if any(field not in row or row[field] is None for field in (*INVARIANT, *IDENTITY)):
                    raise ValueError(f"{path}: missing evidence or identity fields")
                ids = row["retrieved_ids"]
                if (
                    not isinstance(ids, list)
                    or not all(isinstance(mid, str) and mid for mid in ids)
                    or len(ids) != len(set(ids))
                    or type(row["context_complete"]) is not bool
                    or type(row["evidence_found"]) is not int
                    or type(row["evidence_needed"]) is not int
                    or not 0 <= row["evidence_found"] <= row["evidence_needed"]
                ):
                    raise ValueError(f"{path}: invalid retrieval evidence")
                out[probe_id] = row
    if not out:
        raise ValueError(f"{path}: no rows; an empty run proves no invariant")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--routed-scan", action="store_true", help="v4.1: permit changes only on routed probes"
    )
    args = parser.parse_args()

    for path in (args.baseline, args.candidate):
        if not path.is_file():
            print(f"STOP: missing {path}")
            return 2

    try:
        left, right = _rows(args.baseline), _rows(args.candidate)
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        print(f"STOP: {exc}")
        return 2
    only_left, only_right = sorted(set(left) - set(right)), sorted(set(right) - set(left))
    if only_left or only_right:
        print(
            f"STOP: the arms answered different probes "
            f"(+{len(only_right)} / -{len(only_left)}); a comparison over a shifting set "
            "is not a paired comparison"
        )
        return 2

    diverged: dict[str, list[str]] = {}
    for probe_id in sorted(left):
        fields = (*IDENTITY, *INVARIANT)
        if args.routed_scan:
            route = right[probe_id].get("scan_route")
            if not isinstance(route, dict) or type(route.get("routed")) is not bool:
                print("STOP: candidate is missing an observed scan_route")
                return 2
            if route["routed"]:
                fields = IDENTITY
        for field in fields:
            a, b = left[probe_id].get(field), right[probe_id].get(field)
            if a != b:
                diverged.setdefault(field, []).append(probe_id)

    if diverged:
        print(
            "STOP: retrieval is not identical across the arms, so any accuracy "
            "difference has two possible causes."
        )
        for field, ids in diverged.items():
            print(f"  {field}: {len(ids)} probes differ, e.g. {ids[:3]}")
        return 2

    scope = "abstained probes" if args.routed_scan else f"all {len(left)} probes"
    print(f"PASS: retrieval identical across both arms on {scope}")
    print(f"  fields checked: {', '.join(INVARIANT)}")
    print("  paired probe identities and retrieval order were checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
