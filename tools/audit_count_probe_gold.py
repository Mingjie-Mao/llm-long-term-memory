"""Is a `plan` or a `preference` a member of the set a count question asks about?

The count probes ask what the user has *done* or *has* — "how many distinct dishes has the
user cooked", "how many distinct books has the user read", "how many distinct plants does
the user grow". Their gold answer is a SQL `COUNT` over memories whose mapped relation type
matches the question's scope, and that count does not distinguish a completed act from an
intention or a taste.

So the gold answer for "how many books has the user read" includes

    The user is looking for thriller and mystery book recommendations.   (preference)

and the gold answer for "how many plants does the user grow" includes

    The user is hardening off seedlings 7-10 days before transplanting.  (plan)

A model that declines to count those is not under-enumerating. It is right, and the probe
is wrong. This matters because the whole of v4 was aimed at a reading failure diagnosed
from exactly this signal: count answers that name fewer members than the gold set.

This measures how much of the gold is affected. It does **not** prove each such memory
should be excluded — the question's wording decides that, and some plans may legitimately
count. It is the list a human has to read, with the size of the problem attached.

Zero provider calls.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Scopes that record an intention or a taste rather than a completed act. Chosen from the
# store's own vocabulary, not from the failures: `event` and `profile` are the scopes that
# assert something happened or holds, and they are excluded from suspicion here.
INTENTION_SCOPES = ("plan", "preference")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=REPO / "stores/train150.db")
    parser.add_argument(
        "--probes", type=Path, default=REPO / "results/analysis/synthesis-probes.json"
    )
    parser.add_argument(
        "--split", type=Path, default=REPO / "results/manifests/v4-probe-split.json"
    )
    parser.add_argument("--half", default="development", choices=("development", "held_out", "all"))
    parser.add_argument(
        "--out", type=Path, default=REPO / "results/analysis/count-probe-gold-audit.json"
    )
    args = parser.parse_args()

    spec = json.loads(args.probes.read_text(encoding="utf-8"))
    split = json.loads(args.split.read_text(encoding="utf-8"))
    wanted = (
        {p["probe_id"] for p in spec["probes"]} if args.half == "all" else set(split[args.half])
    )
    with sqlite3.connect(f"file:{args.store}?mode=ro", uri=True) as con:
        attributes = {
            row[0]: {"scope": row[1], "content": row[2]}
            for row in con.execute("select id, scope, content from memories")
        }

    probes = [p for p in spec["probes"] if p["kind"] == "count" and p["probe_id"] in wanted]
    if not probes:
        print("STOP: no count probes in that half")
        return 2

    affected, scopes, examples = [], Counter(), []
    for probe in probes:
        gold = probe["evidence_memory_ids"]
        soft = [g for g in gold if attributes.get(g, {}).get("scope") in INTENTION_SCOPES]
        for g in soft:
            scopes[attributes[g]["scope"]] += 1
            if len(examples) < 25:
                examples.append(
                    {
                        "probe_id": probe["probe_id"],
                        "question": probe["question"],
                        "gold_answer": probe["answer"],
                        "scope": attributes[g]["scope"],
                        "memory": attributes[g]["content"],
                    }
                )
        if soft:
            affected.append(
                {
                    "probe_id": probe["probe_id"],
                    "gold_answer": int(probe["answer"]),
                    "intention_members": len(soft),
                    "gold_without_them": int(probe["answer"]) - len(soft),
                }
            )

    total_members = sum(len(p["evidence_memory_ids"]) for p in probes)
    soft_members = sum(a["intention_members"] for a in affected)
    payload = {
        "schema_version": 1,
        "provider_calls": 0,
        "not_a_verdict": (
            "A memory in one of these scopes is not automatically excluded from the set a "
            "question asks about; the question's wording decides. This measures exposure "
            "and lists what a human has to read."
        ),
        "half": args.half,
        "intention_scopes": list(INTENTION_SCOPES),
        "count_probes": len(probes),
        "probes_affected": len(affected),
        "share_affected": round(len(affected) / len(probes), 4),
        "gold_members_total": total_members,
        "intention_members": soft_members,
        "share_of_members": round(soft_members / total_members, 4) if total_members else 0.0,
        "by_scope": dict(scopes),
        "per_probe": sorted(affected, key=lambda a: -a["intention_members"]),
        "examples": examples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"count probes ({args.half})        : {len(probes)}")
    print(f"gold containing an intention     : {len(affected)}  ({payload['share_affected']:.0%})")
    print(
        f"gold members affected            : {soft_members}/{total_members}"
        f"  ({payload['share_of_members']:.0%})"
    )
    print(f"by scope                         : {dict(scopes)}")
    if affected:
        mean = soft_members / len(affected)
        print(f"affected probes overstate gold by: {mean:.1f} members on average")
    try:
        shown = args.out.relative_to(REPO)
    except ValueError:  # --out pointed outside the repository
        shown = args.out
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
