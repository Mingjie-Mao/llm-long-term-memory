"""Read the batch-position pilot and answer the four questions it was run for.

    python scripts/batch_position_analyse.py

Separate from the runner so the analysis can be re-read and re-run without
touching quota, and so a resumed run ends in one command rather than in a
half-remembered sequence of ad-hoc queries.

The four questions, and what answers each:

  1. Does batch position cause omission?   Experiment 1, paired within session.
  2. Does batch size cause omission?       Experiment 2, one cohort at 15 / 5 / 1.
  3. Is the extra yield real or noise?     Redundancy and groundedness, below.
  4. What should the product do?           Coverage x quality x cost, printed last.

**Quality here is a floor, not a verdict.** Redundancy is measured with the same
local encoder retrieval uses, and groundedness is content-word overlap with the
session's own turns. Both are free and neither is a judge: overlap catches a
memory that invents a proper noun, and misses one that recombines real words into
a false claim. A memory that scores well here has not been shown to be *useful* —
only that it is not obviously fabricated and not a restatement of its neighbour.
An LLM-judged fidelity pass would cost quota and is the honest next step if the
cheap floor does not settle the question.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

PILOT = REPO / "results" / "raw" / "batch-position-pilot.json"
STORE = REPO / "stores" / "two-stage-p10.db"
COHORT = 60
STOPWORDS = set(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "from",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "can",
        "could",
        "should",
        "may",
        "might",
        "must",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "as",
        "not",
        "no",
        "yes",
        "user",
        "assistant",
        "they",
        "them",
        "their",
        "he",
        "she",
        "his",
        "her",
        "you",
        "your",
        "i",
        "me",
        "my",
        "we",
        "our",
        "about",
        "into",
        "over",
        "under",
        "more",
        "most",
        "some",
        "any",
    ]
)


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOPWORDS and len(w) > 2}


def exact_mcnemar(wins: int, losses: int) -> float:
    n = wins + losses
    if not n:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n)


def h(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ------------------------------------------------------------- 0. integrity


def integrity(records: list[dict]) -> None:
    h("0. INTEGRITY — the run was resumed, so this is not a formality")
    keys = [(r["arm"], r["arrangement"], r["batch_index"], r["session_id"]) for r in records]
    dupes = len(keys) - len(set(keys))
    print(f"  rows {len(records)}, duplicate (arm, arrangement, batch, session) keys: {dupes}")
    if dupes:
        print(
            "  \033[31mSTOP: a resume re-recorded work; the counts below would double-count\033[0m"
        )

    for arm in dict.fromkeys(r["arm"] for r in records):
        arrs = defaultdict(set)
        for r in records:
            if r["arm"] == arm:
                arrs[r["arrangement"]].add(r["session_id"])
        for name, ids in sorted(arrs.items()):
            flag = (
                "" if len(ids) == COHORT else f"  \033[33m← incomplete ({len(ids)}/{COHORT})\033[0m"
            )
            print(f"  {arm:12s} {name:10s} {len(ids):3d} distinct sessions{flag}")

    # Every arm must be the same cohort, or the arms are not comparable.
    cohorts = {r["arm"]: set() for r in records}
    for r in records:
        cohorts[r["arm"]].add(r["session_id"])
    full = [a for a, s in cohorts.items() if len(s) == COHORT]
    if len(full) > 1:
        base = cohorts[full[0]]
        same = all(cohorts[a] == base for a in full)
        print(f"  complete arms share one cohort: {same}")


# ------------------------------------------------------ 1. position, causal


def position(records: list[dict]) -> None:
    h("1. POSITION — batch size fixed at 15, orderings randomised and reversed")
    rows = [r for r in records if r["arm"] == "position"]
    if not rows:
        print("  no data")
        return

    bands = defaultdict(lambda: [0, 0])
    for r in rows:
        p = r["position"]
        key = "front 0-3" if p <= 3 else "back 11-14" if p >= 11 else "mid 4-10"
        bands[key][0] += 1
        bands[key][1] += r["n_memories"] == 0
    for key in ("front 0-3", "mid 4-10", "back 11-14"):
        n, z = bands[key]
        if n:
            print(f"  {key:12s} n={n:4d}  zero={z:3d}  {z / n:6.1%}")

    front, back = defaultdict(list), defaultdict(list)
    for r in rows:
        if r["position"] <= 3:
            front[r["session_id"]].append(r)
        elif r["position"] >= 11:
            back[r["session_id"]].append(r)

    paired = sorted(set(front) & set(back))
    wins = sum(
        1
        for s in paired
        if not any(x["n_memories"] == 0 for x in front[s])
        and all(x["n_memories"] == 0 for x in back[s])
    )
    losses = sum(
        1
        for s in paired
        if all(x["n_memories"] == 0 for x in front[s])
        and not any(x["n_memories"] == 0 for x in back[s])
    )
    print(f"\n  paired within session (n={len(paired)} seen both front and back)")
    print(f"    yielded in front, zero in back : {wins}")
    print(f"    zero in front, yielded in back : {losses}")
    print(f"    exact McNemar p = {exact_mcnemar(wins, losses):.4f}")

    # Coverage is the binary edge of a continuous loss; the count is the rest of it.
    fm = [sum(x["n_memories"] for x in front[s]) / len(front[s]) for s in paired]
    bm = [sum(x["n_memories"] for x in back[s]) / len(back[s]) for s in paired]
    better = sum(1 for a, b in zip(fm, bm, strict=True) if a > b)
    worse = sum(1 for a, b in zip(fm, bm, strict=True) if a < b)
    print(
        f"\n  memories per session: front {sum(fm) / len(fm):.1f} vs back {sum(bm) / len(bm):.1f}"
    )
    print(
        f"    more in front: {better}   more in back: {worse}   "
        f"exact McNemar p = {exact_mcnemar(better, worse):.4f}"
    )


# ---------------------------------------------------- 2. batch size, causal


def batch_size(records: list[dict]) -> dict[str, list[dict]]:
    h("2. BATCH SIZE — one cohort, three sizes, nothing else changed")
    arms = {}
    for size in (15, 5, 1):
        arm = f"batchsize{size}"
        rows = [r for r in records if r["arm"] == arm]
        if rows:
            arms[arm] = rows
    for arm, rows in arms.items():
        z = sum(1 for r in rows if r["n_memories"] == 0)
        m = sum(r["n_memories"] for r in rows)
        note = "" if len(rows) == COHORT else f"   \033[33m(incomplete {len(rows)}/{COHORT})\033[0m"
        print(
            f"  {arm:12s} n={len(rows):3d}  zero {z:3d} ({z / len(rows):5.1%})  "
            f"memories {m:4d} ({m / len(rows):5.1f}/session){note}"
        )

    print("\n  by production history — did these sessions yield when the store was built?")
    for arm, rows in arms.items():
        for hist, label in ((True, "was zero"), (False, "was fine")):
            g = [r for r in rows if r["historically_zero"] == hist]
            if g:
                z = sum(1 for r in g if r["n_memories"] == 0)
                m = sum(r["n_memories"] for r in g) / len(g)
                print(
                    f"    {arm:12s} {label}: n={len(g):3d}  zero {z:3d} ({z / len(g):5.1%})  "
                    f"{m:5.1f} memories/session"
                )

    # Paired on the sessions every complete arm shares.
    complete = [a for a, r in arms.items() if len(r) == COHORT]
    if len(complete) >= 2:
        print("\n  paired across sizes, on sessions present in both arms:")
        for i, a in enumerate(complete):
            for b in complete[i + 1 :]:
                x = {r["session_id"]: r["n_memories"] for r in arms[a]}
                y = {r["session_id"]: r["n_memories"] for r in arms[b]}
                shared = sorted(set(x) & set(y))
                up = sum(1 for s in shared if y[s] > x[s])
                down = sum(1 for s in shared if y[s] < x[s])
                print(
                    f"    {b} vs {a}: more {up}, fewer {down}, "
                    f"p = {exact_mcnemar(up, down):.4g}  (n={len(shared)})"
                )
    return arms


# ------------------------------------------- 3. is the extra yield any good?


def quality(arms: dict[str, list[dict]]) -> dict[str, dict]:
    h("3. QUALITY — a free floor: is the extra yield redundant, or ungrounded?")
    conn = sqlite3.connect(f"file:{STORE}?mode=ro", uri=True)
    try:
        turns = defaultdict(list)
        for sid, content in conn.execute("SELECT session_id, content FROM turns"):
            turns[sid].append(content)
    finally:
        conn.close()

    try:
        from llm_long_term_memory.embed import Encoder

        encoder = Encoder("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as exc:  # reported, not silently downgraded
        print(f"  encoder unavailable ({type(exc).__name__}), redundancy skipped")
        encoder = None

    out = {}
    for arm, rows in arms.items():
        grounded_num = grounded_den = 0
        dup_pairs = tot_pairs = 0
        for r in rows:
            source = words(" ".join(turns.get(r["session_id"], [])))
            for content in r["memories"]:
                w = words(content)
                if w:
                    grounded_num += len(w & source) / len(w)
                    grounded_den += 1
            if encoder is not None and len(r["memories"]) > 1:
                import numpy as np

                v = encoder.encode(r["memories"], show_progress=False)
                v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
                sim = v @ v.T
                n = len(r["memories"])
                iu = np.triu_indices(n, k=1)
                dup_pairs += int((sim[iu] > 0.90).sum())
                tot_pairs += len(iu[0])
        g = grounded_num / grounded_den if grounded_den else 0
        d = dup_pairs / tot_pairs if tot_pairs else 0
        out[arm] = {"memories": grounded_den, "grounded": g, "near_dup_pairs": d}
        print(
            f"  {arm:12s} memories {grounded_den:5d}  "
            f"mean content-word overlap with source {g:5.1%}  "
            f"near-duplicate pairs {d:5.1%}"
        )
    print("\n  Overlap is a fabrication floor, not a fidelity score, and near-duplicate")
    print("  pairs are within one session only. Neither says the extra memories are")
    print("  worth keeping — only that they are not obviously invented or repeated.")
    return out


# ------------------------------------------------------------ 4. the choice


def cost(arms: dict[str, list[dict]], usage: dict | None) -> None:
    h("4. COST — requests are the binding constraint, not tokens")
    print(f"  pilot usage so far: {usage}")
    print()
    print("  Extraction requests for the full 2,400-session corpus, two-stage:")
    print("    requests = 2 x ceil(sessions / batch) per namespace, ~50 namespaces\n")
    print("  strategy                     batch   corpus requests   zero-yield   mem/session")
    print("  " + "-" * 74)
    for size in (15, 5, 1):
        arm = f"batchsize{size}"
        rows = arms.get(arm)
        batches = sum(-(-48 // size) for _ in range(50))  # ~48 sessions per namespace
        z = f"{sum(1 for r in rows if r['n_memories'] == 0) / len(rows):6.1%}" if rows else "     ?"
        m = f"{sum(r['n_memories'] for r in rows) / len(rows):6.1f}" if rows else "     ?"
        star = "" if not rows or len(rows) == COHORT else "  (partial)"
        print(
            f"  batch {size:<2}                       {size:>3}   {batches * 2:>14,}   "
            f"{z}      {m}{star}"
        )
    print()
    print("  The retry strategy cannot be costed from this pilot and should not be")
    print("  guessed at: `batch 15 + retry the zero-yield` only revisits sessions that")
    print("  produced *nothing*, and the loss measured above is mostly not that. A")
    print("  session that should yield ten memories and yields two never triggers it.")


def main() -> int:
    if not PILOT.exists():
        print(f"{PILOT} not found — run scripts/batch_position_pilot.py first")
        return 1
    data = json.loads(PILOT.read_text(encoding="utf-8"))
    records = data.get("records", [])

    integrity(records)
    position(records)
    arms = batch_size(records)
    quality(arms)
    cost(arms, data.get("usage"))

    incomplete = [a for a, r in arms.items() if len(r) != COHORT]
    if incomplete:
        print(
            f"\n\033[33mIncomplete arms: {', '.join(incomplete)}. "
            f"Re-run the pilot to finish before quoting these rows.\033[0m"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
