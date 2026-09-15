"""Re-derive the BEAM budget from what the ingest actually cost.

The registration sized deduplication as "not predictable" and said the remaining budget
would be re-derived from the measured rate on day one rather than copied from the table.
This is that re-derivation.

Two things the original table got wrong by construction:

* **Adjudication is not a constant.** It rises with store density, because a denser store
  surfaces more near-duplicate candidates. On `dev100` it ended up overtaking extraction —
  236 extraction requests against 264 adjudications in a single day — so any figure taken
  early in a run is a floor, not an estimate.
* **The three models do not share a day.** Extraction, answering and judging run on
  separate pools with separate daily caps, and the judge's is three times the others'. A
  plan costed in "requests" hides that; a plan costed in quota days per pool does not.

Pass 2's adjudication is projected rather than assumed equal to pass 1's: namespace-scoped
dedup adjudicates strictly fewer pairs, and how many fewer is measurable on the store pass 1
just built — `dedup_namespace_probe.py` counts what share of above-threshold neighbours in
dedup's three-hit window cross a conversation. That share is measured on the finished store
where every namespace is present, so it overstates what was reachable early in the run; the
projection is a ceiling on the saving, and is labelled as one.

    python3 tools/beam_budget.py --store-name beam-dev
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

OUT = REPO / "results/analysis/beam-budget.json"

DEV_QUESTIONS = 440
TEST_QUESTIONS = 660
TEST_100K_QUESTIONS = 240
"""Where a whole history fits, so the full-context baseline runs only there."""

FALLBACK_RATE = 1.34
"""LongMemEval's measured answerer requests per question, until BEAM measures its own."""

CANDIDATE_REPEATS = 3
"""Registered: a dev comparison repeats the candidate three times."""

MIN_NEIGHBOUR_PAIRS = 30
"""Below this the cross-conversation share is not a measurement. Early in a run the store
holds a few namespaces and almost no above-threshold neighbours, and projecting pass 2 from
four pairs against two would be arithmetic wearing a number."""


def learned_limits() -> dict[str, int]:
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.llm import Limits, QuotaManager

    settings = Settings()
    quota = QuotaManager(
        state_dir=settings.store_dir / "quota", default=Limits(rpm=10, tpm=250_000, rpd=500)
    )
    return {model: limit.rpd for model, limit in quota.load_learned().items()}


def checkpoint(settings, store_name: str) -> dict | None:
    path = settings.store_dir / f"{store_name}-ingest.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def zero_yield(settings, store_name: str) -> dict | None:
    """Sessions that reached a terminal state and produced no memory.

    The comparison that matters is against `train150`'s 15.6%: a rate far off it means the
    BEAM chunking is not the operating point the frozen extractor was measured at, and that
    is a reason to stop and re-cut rather than to keep paying.
    """
    from llm_long_term_memory.store import SQLiteMemoryStore

    path = settings.store_dir / f"{store_name}.db"
    if not path.is_file():
        return None
    store = SQLiteMemoryStore(path, read_only=True)
    store.initialize()
    try:
        zero, substantive = store.zero_yield_sessions()
    finally:
        store.close()
    return {
        "zero_memory_sessions": zero,
        "substantive_sessions": substantive,
        "rate": (zero / substantive) if substantive else None,
        "train150_reference": 0.156,
    }


def cross_namespace_share(settings, store_name: str, sample: int) -> dict | None:
    import dedup_namespace_probe

    path = settings.store_dir / f"{store_name}.db"
    if not path.is_file():
        return None
    stem = path.with_name(path.stem + "-index")
    if not Path(f"{stem}.npy").is_file():
        return None
    result = dedup_namespace_probe.probe(path, stem, sample, seed=0)
    above = result["neighbours_above_threshold"]
    total = above["same_namespace"] + above["cross_namespace"]
    return {
        "same_namespace": above["same_namespace"],
        "cross_namespace": above["cross_namespace"],
        "share_crossing": (above["cross_namespace"] / total) if total else None,
        "measured_on": "the finished store, where every namespace is present",
    }


def days(requests: int, cap: int) -> int:
    return math.ceil(requests / cap) if requests else 0


def plan(extraction: int, adjudication: int, pass2_adjudication: int, caps: dict) -> dict:
    """Every remaining stage, in requests and in quota days on its own pool."""
    extractor = caps.get("extractor", 500)
    answerer = caps.get("answerer", 500)
    judge = caps.get("judge", 1500)

    answer_per_pass = round(DEV_QUESTIONS * FALLBACK_RATE)
    stages = [
        ("3.2 ingest pass 1 (frozen dedup)", {"extractor": extraction + adjudication}),
        ("3.2 ingest pass 2 (scoped dedup, replayed)", {"extractor": pass2_adjudication}),
        ("3.3 answering pass A + judging", {"answerer": answer_per_pass, "judge": DEV_QUESTIONS}),
        ("noise pass B (judge only)", {"judge": DEV_QUESTIONS}),
        ("noise pass C (answer + judge)", {"answerer": answer_per_pass, "judge": DEV_QUESTIONS}),
        (
            "4.2 candidate on dev, 3 repeats",
            {
                "answerer": answer_per_pass * CANDIDATE_REPEATS,
                "judge": DEV_QUESTIONS * CANDIDATE_REPEATS,
            },
        ),
        (
            "4.4 final run, 4 arms, once",
            {
                "answerer": round(TEST_QUESTIONS * FALLBACK_RATE * 3 + TEST_100K_QUESTIONS),
                "judge": TEST_QUESTIONS * 3 + TEST_100K_QUESTIONS,
            },
        ),
    ]
    caps_by_pool = {"extractor": extractor, "answerer": answerer, "judge": judge}
    rows = []
    totals = {"extractor": 0, "answerer": 0, "judge": 0}
    for name, pools in stages:
        row = {"stage": name, "requests": pools}
        row["quota_days"] = max(
            (days(n, caps_by_pool[pool]) for pool, n in pools.items()), default=0
        )
        for pool, n in pools.items():
            totals[pool] += n
        rows.append(row)
    return {
        "daily_caps": caps_by_pool,
        "answerer_requests_per_pass": answer_per_pass,
        "stages": rows,
        "total_requests": totals,
        "total_quota_days_if_stages_run_one_at_a_time": sum(row["quota_days"] for row in rows),
        "floor_if_pools_are_saturated_in_parallel": max(
            days(totals[pool], caps_by_pool[pool]) for pool in totals
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store-name", default="beam-dev")
    parser.add_argument("--sample", type=int, default=1500)
    args = parser.parse_args()

    from llm_long_term_memory.config import Settings

    settings = Settings()
    frozen, scoped = f"{args.store_name}-frozen", f"{args.store_name}-scoped"
    first = checkpoint(settings, frozen)
    if first is None:
        print(f"STOP: no ingest checkpoint for {frozen}")
        return 2

    extraction = first["extraction_requests"]
    adjudication = first["adjudication_requests"]
    done = len(first.get("done_sessions") or [])
    batches_done = math.ceil(extraction / 2)
    total_batches = 256
    complete = done >= 3679

    crossing = cross_namespace_share(settings, frozen, args.sample)
    share = (crossing or {}).get("share_crossing")
    pairs = (crossing or {}).get("same_namespace", 0) + (crossing or {}).get("cross_namespace", 0)
    if crossing is not None and pairs < MIN_NEIGHBOUR_PAIRS:
        crossing["too_few_pairs_to_project_from"] = pairs
        crossing["minimum"] = MIN_NEIGHBOUR_PAIRS
        share = None

    # Scale an incomplete pass to its finish before projecting pass 2, and say the rate is
    # a floor: adjudication per batch grows as the store fills.
    per_batch = adjudication / batches_done if batches_done else 0
    projected_adjudication = adjudication if complete else round(per_batch * total_batches)
    pass2 = (
        round(projected_adjudication * (1 - share)) if share is not None else projected_adjudication
    )

    caps = {}
    limits = learned_limits()
    for model, rpd in limits.items():
        if "3.1-flash" in model:
            caps["extractor"] = rpd
        elif "3.5-flash" in model:
            caps["answerer"] = rpd
        else:
            caps["judge"] = rpd

    payload = {
        "name": "beam-budget",
        "recomputed_from": "the ingest checkpoint, not the registration's table",
        "provider_calls": 0,
        "pass_1": {
            "complete": complete,
            "sessions_done": done,
            "batches_done": batches_done,
            "extraction_requests": extraction,
            "adjudication_requests": adjudication,
            "adjudication_per_extraction": round(adjudication / extraction, 3)
            if extraction
            else None,
            "adjudication_per_batch": round(per_batch, 2),
            "memories_written": first.get("memories_written"),
            "duplicates_dropped": first.get("duplicates_dropped"),
            "projected_adjudication_at_completion": projected_adjudication,
            "caveat": "adjudication per batch rises with store density, so a rate measured "
            "before the run finishes is a floor rather than an estimate",
        },
        "pass_2_projection": {
            "extraction_requests": 0,
            "adjudication_requests": pass2,
            "basis": "pass 1's adjudications less the share that crossed a conversation, "
            "which namespace-scoped dedup never adjudicates",
            "cross_namespace": crossing,
            "this_is_a_ceiling_on_the_saving": share is not None,
        },
        "zero_yield": zero_yield(settings, frozen),
        "plan": plan(extraction if complete else 512, projected_adjudication, pass2, caps),
        "observed_limits": limits,
    }
    scoped_checkpoint = checkpoint(settings, scoped)
    if scoped_checkpoint:
        payload["pass_2_actual"] = {
            "extraction_requests": scoped_checkpoint["extraction_requests"],
            "adjudication_requests": scoped_checkpoint["adjudication_requests"],
            "memories_written": scoped_checkpoint.get("memories_written"),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    p1 = payload["pass_1"]
    print(
        f"pass 1 {'complete' if complete else 'in progress'}: "
        f"{p1['batches_done']}/{total_batches} batches, {p1['sessions_done']:,} sessions\n"
        f"  {p1['extraction_requests']} extraction + {p1['adjudication_requests']} adjudication"
        f"  ({p1['adjudication_per_extraction']} per extraction, "
        f"{p1['adjudication_per_batch']} per batch)"
    )
    if payload["zero_yield"] and payload["zero_yield"]["rate"] is not None:
        z = payload["zero_yield"]
        print(
            f"  zero-memory sessions {z['zero_memory_sessions']:,} of {z['substantive_sessions']:,}"
            f" = {z['rate']:.1%}  (train150: {z['train150_reference']:.1%})"
        )
    print()
    print(f"{'stage':44} {'requests':>34} {'days':>5}")
    for row in payload["plan"]["stages"]:
        pools = ", ".join(f"{pool} {n:,}" for pool, n in row["requests"].items())
        print(f"{row['stage']:44} {pools:>34} {row['quota_days']:>5}")
    totals = payload["plan"]["total_requests"]
    print(
        f"\ntotal: extractor {totals['extractor']:,}  answerer {totals['answerer']:,}  "
        f"judge {totals['judge']:,}"
    )
    print(
        f"quota days: {payload['plan']['total_quota_days_if_stages_run_one_at_a_time']} "
        f"sequential, {payload['plan']['floor_if_pools_are_saturated_in_parallel']} if the "
        f"three pools were saturated in parallel"
    )
    print(f"\nwritten: {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
