"""Is zero-yield caused by where a session sits in the extraction batch?

Observationally it looks that way. Over the 2,135 substantive extraction attempts
in the clean P10 store, positions 0-3 of a 15-session batch yielded nothing 6.4% of
the time and positions 4-14 did so 18.3% of the time — a 2.86x risk ratio that
survives a within-batch permutation of the position labels (p = 0.00005, N=20,000)
and a batch-clustered bootstrap (95% CI 2.15-4.35).

That is observational. Every session sat at exactly one position, so the comparison
is still between *different* sessions, and something about the sessions that happen
to arrive later could explain it. This script randomises the thing that was never
randomised.

Two questions, deliberately separated, because a single "batch 15 vs batch 5" run
would confound them:

  Experiment 1 — position.  One fixed set of sessions, batch size held at 15, run
  under several orderings. Each ordering is paired with its reverse, so a session at
  position p in one arrangement sits at position 14-p in the other. The comparison
  is then *within session*: the same conversation, the same batch size, the same
  prompt, seen early and seen late.

  Experiment 2 — batch size.  The same sessions again at 15, 5 and 1 sessions per
  request. Batch size 1 is the upper bound: one conversation, one request, nowhere
  to be buried.

Nothing else moves. Same model, same prompts, same schema, same decoding, same
sessions. The extractor is driven directly rather than through `IngestionPipeline`,
so batch composition and position are set by this script and **no store is
written** — the clean P10 store is an input here, never an output.

    python scripts/batch_position_pilot.py --plan    # sample and cost it, no calls
    python scripts/batch_position_pilot.py --run     # spend the quota

Checkpointed per batch: a daily-quota stop loses the batch in flight, not the run.

A note on what this cannot settle. Positions 0-3 still yield nothing 6.4% of the
time, so position cannot be the whole story, and this pilot is not powered to
characterise the remainder. If the mechanism is confirmed, the report should say
"larger batched extraction shows position-dependent omission" and leave *why* —
output budget, enumeration drift, long-context allocation — as the hypothesis it
currently is.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from llm_long_term_memory.config import ExperimentConfig, Settings  # noqa: E402
from llm_long_term_memory.evaluation.datasets import longmemeval as lme  # noqa: E402
from llm_long_term_memory.ingest.pipeline import (  # noqa: E402
    batched,
    group_by_namespace,
    namespaced_sessions,
)

STORE = REPO / "stores" / "two-stage-p10.db"
OUT = REPO / "results" / "raw" / "batch-position-pilot.json"
SEED = 20260817
N_PER_GROUP = 30
PRODUCTION_BATCH = 15


# --------------------------------------------------------------- the cohort


def attempts(cfg, settings):
    """Every substantive extraction attempt, with the position it actually had.

    The unit is a `(namespace, session)` pair, not a session: 39 sessions appear in
    more than one namespace and are deliberately extracted once per namespace, so
    one can yield and the other not. Keying yield on `source_session_id` alone
    conflates them and under-counts zero-yield by 11.
    """
    conn = sqlite3.connect(f"file:{STORE}?mode=ro", uri=True)
    try:
        turns = dict(conn.execute("SELECT session_id, COUNT(*) FROM turns GROUP BY 1"))
        roles = defaultdict(lambda: [0, 0])
        for sid, role, n in conn.execute(
            "SELECT session_id, role, COUNT(*) FROM turns GROUP BY 1, 2"
        ):
            roles[sid][0 if role == "assistant" else 1] += n
        yielded = {
            (ns, sid)
            for sid, ns in conn.execute(
                "SELECT DISTINCT source_session_id, user_id FROM memories "
                "WHERE source_session_id IS NOT NULL"
            )
        }
    finally:
        conn.close()

    instances = lme.load(cfg.dataset_variant, settings.data_dir, limit=cfg.dataset_limit)
    by_id = {s.session_id: s for _, s in namespaced_sessions(instances)}

    out = []
    for ns, sessions in group_by_namespace(namespaced_sessions(instances)):
        for batch_index, batch in enumerate(batched(sessions, PRODUCTION_BATCH)):
            for position, sess in enumerate(batch):
                n = turns.get(sess.session_id, 0)
                if n < 6:
                    continue
                a, u = roles[sess.session_id]
                out.append(
                    {
                        "namespace": ns,
                        "session_id": sess.session_id,
                        "turns": n,
                        "assistant_share": round(a / max(1, a + u), 3),
                        "batch_index": batch_index,
                        "position": position,
                        "zero_yield": (ns, sess.session_id) not in yielded,
                    }
                )
    return out, by_id


def matched_sample(rows, rng):
    """30 historically zero-yield attempts and 30 that yielded, matched on shape.

    Both groups are carried because the question is not only "can a failure be
    rescued" but "can a success be broken" — if batch position is real, moving a
    session that worked into a late slot should cost yield too, and a design with
    only failures in it could never see that.

    Matching is on turn count and assistant share, the two properties an earlier
    audit compared zero-yield sessions against normal ones on. It found them
    indistinguishable, which is what made "extractor variance" the standing
    explanation; holding them fixed here means any difference this pilot finds
    cannot be attributed back to them.
    """
    zero = [r for r in rows if r["zero_yield"]]
    good = [r for r in rows if not r["zero_yield"]]
    rng.shuffle(zero)
    rng.shuffle(good)

    picked_zero, picked_good, used = [], [], set()
    for z in zero:
        if len(picked_zero) >= N_PER_GROUP:
            break
        candidates = [
            g
            for g in good
            if g["session_id"] not in used
            and abs(g["turns"] - z["turns"]) <= 2
            and abs(g["assistant_share"] - z["assistant_share"]) <= 0.08
        ]
        if not candidates:
            continue
        picked_zero.append(z)
        picked_good.append(candidates[0])
        used.add(z["session_id"])
        used.add(candidates[0]["session_id"])
    return picked_zero, picked_good


# ------------------------------------------------------------ arrangements


def arrangements(cohort, rng, n_pairs=2):
    """Orderings of the cohort, each paired with its own reverse.

    Reversal is what makes the comparison paired: position p becomes 14-p, so every
    session that sat in the first four slots sits in the last four, with the same
    batch size and the same neighbours. Random orderings alone would leave the
    front/back contrast to chance and to whichever sessions happened to land there.
    """
    out = []
    for i in range(n_pairs):
        order = list(cohort)
        rng.shuffle(order)
        out.append((f"perm{i}", order))
        out.append((f"perm{i}rev", list(reversed(order))))
    return out


def plan(cohort, arrs):
    """Batches to run, as (arm, arrangement, batch_size, [entries])."""
    jobs = []
    for name, order in arrs:
        for bi, chunk in enumerate(batched(order, PRODUCTION_BATCH)):
            jobs.append(("position", name, PRODUCTION_BATCH, bi, chunk))
    for size in (PRODUCTION_BATCH, 5, 1):
        order = list(cohort)
        for bi, chunk in enumerate(batched(order, size)):
            jobs.append((f"batchsize{size}", "fixed", size, bi, chunk))
    return jobs


# ------------------------------------------------------------------- runner


def run(jobs, by_id, cfg, settings, done):
    from llm_long_term_memory.ingest import TwoStageExtractor
    from llm_long_term_memory.llm import Limits, QuotaManager, UsageTracker
    from llm_long_term_memory.llm.client import DailyQuotaExhausted, GeminiClient

    quota = QuotaManager(
        state_dir=settings.store_dir / "quota",
        default=Limits(rpm=cfg.quota.rpm, tpm=cfg.quota.tpm, rpd=cfg.quota.rpd),
    )
    quota.load_learned()
    usage = UsageTracker()
    client = GeminiClient(settings.require_api_key(), quota=quota, usage=usage)
    extractor = TwoStageExtractor(client, cfg.models.extractor)

    records = list(done)
    seen = {(r["arm"], r["arrangement"], r["batch_index"]) for r in records}
    for arm, arrangement, size, batch_index, chunk in jobs:
        key = (arm, arrangement, batch_index)
        if key in seen:
            continue
        sessions = [by_id[e["session_id"]] for e in chunk]
        try:
            outcome = extractor.extract(sessions)
        except DailyQuotaExhausted as exc:
            print(f"\n\033[33mdaily quota reached: {exc}\033[0m")
            print("progress is saved; re-run after the reset to continue")
            break
        except Exception as exc:  # recorded, not swallowed
            print(f"  \033[31mFAIL\033[0m {arm}/{arrangement}#{batch_index}: {exc}")
            continue

        produced = defaultdict(list)
        for memory in outcome.memories:
            produced[memory.source_session_id].append(memory.content)
        for position, entry in enumerate(chunk):
            contents = produced.get(entry["session_id"], [])
            records.append(
                {
                    "arm": arm,
                    "arrangement": arrangement,
                    "batch_size": size,
                    "batch_index": batch_index,
                    "position": position,
                    "session_id": entry["session_id"],
                    "namespace": entry["namespace"],
                    "turns": entry["turns"],
                    "assistant_share": entry["assistant_share"],
                    "historically_zero": entry["zero_yield"],
                    "n_memories": len(contents),
                    "memories": contents,
                }
            )
        zero = sum(1 for e in chunk if not produced.get(e["session_id"]))
        print(
            f"  {arm}/{arrangement} batch {batch_index} (size {len(chunk)}): "
            f"{len(outcome.memories)} memories, {zero}/{len(chunk)} zero"
        )
        save(records, usage)
    return records, usage


def save(records, usage=None):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": (
            "Batch-position and batch-size pilot. One fixed cohort of sessions "
            "extracted under several orderings and batch sizes; model, prompts, "
            "schema and decoding are identical throughout and no store is written. "
            "See scripts/batch_position_pilot.py."
        ),
        "seed": SEED,
        "records": records,
    }
    if usage is not None:
        payload["usage"] = {
            "requests": usage.summary().get("total_requests"),
            "tokens": usage.summary().get("total_tokens"),
        }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="spend quota; otherwise plan only")
    parser.add_argument("--pairs", type=int, default=2, help="permutation pairs for experiment 1")
    args = parser.parse_args()

    settings = Settings()
    cfg = ExperimentConfig.from_yaml(str(REPO / "configs" / "baselines.yaml"))
    rng = random.Random(SEED)

    rows, by_id = attempts(cfg, settings)
    zero_rows = [r for r in rows if r["zero_yield"]]
    print(f"cohort source: {len(rows)} substantive attempts, {len(zero_rows)} zero-yield")

    picked_zero, picked_good = matched_sample(rows, rng)
    cohort = picked_zero + picked_good
    rng.shuffle(cohort)
    print(
        f"matched pilot: {len(picked_zero)} historically zero + "
        f"{len(picked_good)} historically yielding = {len(cohort)} sessions"
    )

    arrs = arrangements(cohort, rng, n_pairs=args.pairs)
    jobs = plan(cohort, arrs)
    calls = sum(2 for _ in jobs)  # two-stage: Stage A + Stage B per batch
    print(f"\nplanned: {len(jobs)} batches, ~{calls} extractor requests")
    for arm in dict.fromkeys(j[0] for j in jobs):
        n = sum(1 for j in jobs if j[0] == arm)
        print(f"   {arm:14s} {n:3d} batches  ~{n * 2:3d} requests")

    if not args.run:
        print("\nRe-run with --run to start.")
        return 0

    done = []
    if OUT.exists():
        done = json.loads(OUT.read_text(encoding="utf-8")).get("records", [])
        if done:
            print(f"\nresuming: {len(done)} rows already recorded")
    print()
    records, usage = run(jobs, by_id, cfg, settings, done)
    save(records, usage)
    print(f"\n→ {OUT}  ({len(records)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
