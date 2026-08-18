"""Report the held-out run the way it has to be reported.

    python scripts/heldout_report.py

The metrics are fixed here rather than chosen after seeing the numbers, and the
one that matters most is the split. On dev50 the headline 72.0% turned out to be
54.0% answered from structured memory plus 18.0% rescued by the raw archive, and
memory alone tied `naive_rag`. A single accuracy figure reads as "the memory
system answered these", which for a quarter of the correct ones was not true. So
the three appear together or not at all.

Questions whose evidence the provider refused to read are reported separately.
Their turns are archived and reachable by the fallback, but they produced no
memories, and a failure there is a content-policy refusal rather than a property
of this system. Folding them into the accuracy would be quietly charging the
system for someone else's decision.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

RESULT = REPO / "results" / "raw" / "two_stage_hydrated.heldout100.jsonl"
CKPT = REPO / "stores" / "heldout100-ingest.json"
DEV = REPO / "results" / "raw" / "two_stage_hydrated.a2-clean-p10-fallback-v2.jsonl"


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def split(rows: list[dict], label: str) -> dict:
    total = len(rows)
    correct = [r for r in rows if r["correct"]]
    by = Counter(str(r["notes"].get("fallback_level")) for r in correct)
    memory_only = by.get("none", 0)
    rescued = sum(v for k, v in by.items() if k != "none")
    fired = sum(1 for r in rows if str(r["notes"].get("fallback_level")) != "none")
    print(f"\n  {label}  (n={total})")
    print(f"    {'final accuracy':32s} {len(correct):3d}/{total}  {len(correct) / total:6.1%}")
    mo = f"{memory_only:3d}/{total}  {memory_only / total:6.1%}"
    print(f"    {'from structured memory alone':32s} {mo}")
    print(f"    {'rescued by the raw archive':32s} {rescued:3d}/{total}  {rescued / total:6.1%}")
    print(f"    {'fallback fired':32s} {fired:3d}/{total}  {fired / total:6.1%}")
    if fired:
        print(f"    {'  and was right':32s} {rescued:3d}/{fired}  {rescued / fired:6.1%}")
    return {
        "n": total,
        "final": len(correct),
        "memory_only": memory_only,
        "rescued": rescued,
        "fired": fired,
    }


def main() -> int:
    rows = load(RESULT)
    if not rows:
        print(f"{RESULT.name} does not exist — the held-out run has not happened.")
        return 1

    blocked_questions: set[str] = set()
    if CKPT.exists():
        ck = json.loads(CKPT.read_text(encoding="utf-8"))
        blocked_questions = {k.split(":", 1)[0] for k in ck.get("blocked_sessions", [])}

    clean = [r for r in rows if r["question_id"] not in blocked_questions]
    affected = [r for r in rows if r["question_id"] in blocked_questions]

    print("=" * 92)
    print("HELD-OUT 100 — one run, on the system frozen in results/frozen/p10-final")
    print("=" * 92)

    if affected:
        print(f"\n  {len(affected)} question(s) had evidence the provider refused to read:")
        for r in affected:
            print(f"    {r['question_id']:15s} {r['question_type']:26s} correct={r['correct']}")
        print("    Their turns are archived and the fallback can reach them; they produced")
        print("    no memories. Reported separately rather than charged to the system.")

    split(rows, "all 100")
    if affected:
        split(clean, f"excluding the {len(affected)} refused")

    print("\n  accuracy by question type")
    for t in sorted({r["question_type"] for r in rows}):
        g = [r for r in rows if r["question_type"] == t]
        c = sum(r["correct"] for r in g)
        print(f"    {t:28s} {c:3d}/{len(g):<3d} {c / len(g):6.1%}")

    print("\n  supply and cost")
    rec = sum(1 for r in rows if r.get("source_session_recalled"))
    ev = sum(1 for r in rows if r.get("evidence_recalled"))
    print(f"    {'source-session recall':32s} {rec:3d}/{len(rows)}  {rec / len(rows):6.1%}")
    print(f"    {'evidence recall':32s} {ev:3d}/{len(rows)}  {ev / len(rows):6.1%}")
    print(f"    {'median context tokens':32s} {median(r['context_tokens'] for r in rows):8,.0f}")
    print(f"    {'median answer latency (ms)':32s} {median(r['latency_ms'] for r in rows):8,.0f}")

    dev = load(DEV)
    if dev:
        print("\n  against dev50, where every design decision was made")
        d = split(dev, "dev50 (development)")
        h = split(rows, "heldout100 (unseen)")
        for k, name in (
            ("final", "final accuracy"),
            ("memory_only", "memory alone"),
            ("rescued", "archive rescue"),
        ):
            a, b = d[k] / d["n"], h[k] / h["n"]
            print(f"    {name:24s} dev {a:6.1%}   held-out {b:6.1%}   {b - a:+.1f}pp")

    print("\n  Every design decision in this system was made against dev50, including")
    print("  six modules cancelled on its evidence. This is the first measurement that")
    print("  was not fitted to. It is not to be used to tune anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
