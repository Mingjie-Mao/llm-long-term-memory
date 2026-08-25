"""Can retrieval put a gold session in the top M, without being told which it is?

    python scripts/session_recall.py --store heldout100 --questions heldout100.json

**No LLM calls.** Local encoder, the real retriever, the real store. This is the
gate [the context-shape pre-registration](../results/prereg-context-shape.md) runs
before spending any answerer quota:

> If session recall @ top-M is below 80% on `train150`, `coherent-auto` is not run
> on `dev100` — the development work moves to session selection first, and no
> validation quota is spent on an arm already known to be starved.

The reason to expect it to pass is that the gold session is already being found.
`source_session_recalled` is 94% on `heldout100` and source-session recall has been
93.5-94.0% on every store measured; the question is only whether it survives being
aggregated into a *session* ranking and cut at M.

Those are different things, and the difference is the whole point. A gold memory
ranked 18th of 20 counts for `source_session_recalled` and may still lose its
session to two conversations that each contributed three stronger memories.

## On which set to run it

`train150` is the set this decides on. `heldout100` is spent, so running it there
costs nothing that is not already gone — but its numbers **may not be used to pick
M**. The budget is a `train150` decision by pre-registration, and a threshold
chosen on a spent test set is still a threshold chosen by looking at outcomes.
Use `heldout100` to check that this code does what it says.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from llm_long_term_memory.config import Settings  # noqa: E402
from llm_long_term_memory.evaluation.datasets import longmemeval as lme  # noqa: E402
from llm_long_term_memory.evaluation.manifest import load_manifest  # noqa: E402
from llm_long_term_memory.ingest.pipeline import IngestProgress, _key  # noqa: E402
from llm_long_term_memory.retrieve import (  # noqa: E402
    SessionBudget,
    build_coherent_context,
    rank_sessions,
)
from llm_long_term_memory.store import external_session_id  # noqa: E402

MS = (1, 2, 3, 5, 10)
AGGREGATES = ("max", "mean", "sum_top3", "sum")


def gate_is_complete(
    *,
    ingest_complete: bool,
    manifest_questions: int,
    scored_questions: int,
    skipped_without_gold: int,
) -> bool:
    return ingest_complete and scored_questions == manifest_questions and skipped_without_gold == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default="train150")
    parser.add_argument("--questions", default="train150.json")
    parser.add_argument("--config", default="configs/fallback.yaml")
    parser.add_argument(
        "--memory-limit",
        "--candidate-limit",
        dest="memory_limit",
        type=int,
        help=(
            "number of ranked memories handed to session aggregation; defaults to "
            "retrieval.top_k from the config. --candidate-limit is a deprecated alias"
        ),
    )
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--aggregate",
        choices=AGGREGATES,
        help="configuration to apply to the 80%% gate; all choices are still reported",
    )
    parser.add_argument(
        "--window-radius",
        type=int,
        help="development override; defaults to context.window_radius from config",
    )
    parser.add_argument(
        "--max-total-memories",
        type=int,
        help="development override; defaults to context.max_total_memories from config",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="diagnose only fully ingested questions; never use this as the final gate",
    )
    args = parser.parse_args()

    if args.store != "train150" and (
        args.aggregate is not None
        or args.window_radius is not None
        or args.max_total_memories is not None
    ):
        print(
            "refusing development overrides outside train150; validation/test stores "
            "must read their frozen context budget from the config"
        )
        return 2

    from llm_long_term_memory.api.service import MemoryService

    settings = Settings()
    manifest = load_manifest(settings.results_dir / "manifests" / args.questions)
    inst = {i.question_id: i for i in lme.load(manifest.variant, settings.data_dir)}
    missing = set(manifest.question_ids) - set(inst)
    if missing:
        print(f"manifest has {len(missing)} question(s) outside its declared dataset")
        return 2
    progress = IngestProgress.load(settings.store_dir / f"{args.store}-ingest.json")
    terminal = progress.done_sessions | progress.blocked_sessions
    complete_ids = [
        qid
        for qid in manifest.question_ids
        if all(_key(qid, session) in terminal for session in inst[qid].sessions)
    ]
    complete = len(complete_ids) == len(manifest.question_ids)
    if not complete and not args.allow_partial:
        print(
            f"refusing final gate: only {len(complete_ids)}/{len(manifest.question_ids)} "
            "questions are fully ingested; use --allow-partial for a provisional diagnostic"
        )
        return 2
    question_ids = manifest.question_ids if complete else complete_ids

    svc = MemoryService(store_name=args.store, config_path=args.config)
    memory_limit = (
        args.memory_limit if args.memory_limit is not None else svc.config.retrieval.top_k
    )
    if memory_limit < 1:
        print("memory limit must be positive")
        svc.close()
        return 2
    label = "final" if complete else "PARTIAL — diagnostic only"
    print(
        f"session recall · store {args.store} · {len(question_ids)} fully ingested questions "
        f"· top {memory_limit} memories · {label} · no LLM calls\n"
    )

    selected_aggregate = args.aggregate or svc.config.context.aggregate
    if selected_aggregate not in AGGREGATES:
        print(f"unsupported configured aggregate: {selected_aggregate}")
        svc.close()
        return 2
    window_radius = (
        args.window_radius if args.window_radius is not None else svc.config.context.window_radius
    )
    max_total_memories = (
        args.max_total_memories
        if args.max_total_memories is not None
        else svc.config.context.max_total_memories
    )
    if window_radius is not None and window_radius < 0:
        print("window radius cannot be negative")
        svc.close()
        return 2
    if max_total_memories is not None and max_total_memories < 1:
        print("max total memories must be positive")
        svc.close()
        return 2

    hits_at = {aggregate: Counter() for aggregate in AGGREGATES}
    assembled_hits = Counter()
    assembled_tokens = {aggregate: [] for aggregate in AGGREGATES}
    assembled_memories = {aggregate: [] for aggregate in AGGREGATES}
    truncated = Counter()
    flat_tokens: list[int] = []
    memory_level = 0
    ranks = {aggregate: [] for aggregate in AGGREGATES}
    skipped: list[str] = []
    rows = []

    try:
        for qid in question_ids:
            question = inst[qid]
            gold = set(question.answer_session_ids)
            if not gold:
                skipped.append(qid)
                continue

            result = svc.search(qid, question.question, limit=memory_limit)
            hits = list(result.memories)
            flat_tokens.append(sum(hit.memory.token_count for hit in hits))
            if any(
                h.memory.source_session_id
                and external_session_id(h.memory.source_session_id) in gold
                for h in hits
            ):
                memory_level += 1

            positions: dict[str, int | None] = {}
            assembled: dict[str, bool] = {}
            session_counts: dict[str, int] = {}
            by_session: dict[str, list] = {}
            for memory in svc.store.iter_all(qid):
                if memory.source_session_id:
                    by_session.setdefault(memory.source_session_id, []).append(memory)
            for aggregate in AGGREGATES:
                ranked = [external_session_id(s) for s, _ in rank_sessions(hits, aggregate)]
                position = next((i + 1 for i, s in enumerate(ranked) if s in gold), None)
                positions[aggregate] = position
                session_counts[aggregate] = len(ranked)
                ranks[aggregate].append(position)
                for m in MS:
                    if position is not None and position <= m:
                        hits_at[aggregate][m] += 1
                context = build_coherent_context(
                    hits,
                    lambda session_id, sessions=by_session: sessions.get(session_id, []),
                    SessionBudget(
                        max_sessions=svc.config.context.max_sessions,
                        window_radius=window_radius,
                        max_total_memories=max_total_memories,
                        aggregate=aggregate,
                        session_order=svc.config.context.session_order,
                        include_superseded=svc.config.context.include_superseded,
                    ),
                )
                assembled[aggregate] = bool(
                    {external_session_id(session_id) for session_id in context.sessions} & gold
                )
                assembled_hits[aggregate] += assembled[aggregate]
                truncated[aggregate] += context.truncated
                assembled_tokens[aggregate].append(
                    sum(memory.token_count for memory in context.memories)
                )
                assembled_memories[aggregate].append(len(context.memories))
            rows.append(
                {
                    "question_id": qid,
                    "gold_session_rank": positions,
                    "assembled_gold_session": assembled,
                    "sessions": session_counts[selected_aggregate],
                }
            )
    finally:
        svc.close()

    n = len(ranks[selected_aggregate])
    if not n:
        print("no questions with gold sessions; nothing measured")
        return 1
    gate_complete = gate_is_complete(
        ingest_complete=complete,
        manifest_questions=len(manifest.question_ids),
        scored_questions=n,
        skipped_without_gold=len(skipped),
    )
    if complete and not gate_complete:
        print(
            f"refusing final gate: {len(skipped)} question(s) have no gold session; "
            "the Top-3 denominator may not silently shrink"
        )
        return 2

    print(f"{'aggregate':<14}" + "".join(f"@{m:>2}".rjust(9) for m in MS))
    for aggregate in AGGREGATES:
        print(f"{aggregate:<14}" + "".join(f"{hits_at[aggregate][m] / n:>8.1%}" for m in MS))
    print(
        f"\n{'aggregate':<14}{'assembled':>11}{'median mem':>12}{'median tok':>12}{'truncated':>12}"
    )
    for aggregate in AGGREGATES:
        print(
            f"{aggregate:<14}{assembled_hits[aggregate] / n:>10.1%}"
            f"{statistics.median(assembled_memories[aggregate]):>12.1f}"
            f"{statistics.median(assembled_tokens[aggregate]):>12.1f}"
            f"{truncated[aggregate]:>12}"
        )
    selected_ranks = ranks[selected_aggregate]
    print(f"\n  selected aggregate: {selected_aggregate}")
    print(
        f"  {'any rank at all':<24}{sum(r is not None for r in selected_ranks):>6}"
        f"{sum(r is not None for r in selected_ranks) / n:>8.1%}"
    )
    print(f"  {'memory-level (for contrast)':<24}{memory_level:>6}{memory_level / n:>8.1%}")
    print(f"  {'flat median context tokens':<30}{statistics.median(flat_tokens):>8.1f}")
    found = [r for r in selected_ranks if r is not None]
    if found:
        found.sort()
        print(f"  median gold-session rank  {found[len(found) // 2]}")
    if skipped:
        print(f"\n  {len(skipped)} question(s) carry no gold session and were excluded")

    gate = hits_at[selected_aggregate][3] / n
    verdict = "PASS" if gate >= 0.80 else "STOP"
    colour = "\033[32m" if gate >= 0.80 else "\033[31m"
    print(
        f"\n{colour}{verdict}\033[0m  the pre-registered gate is 80% at the chosen M "
        f"(M=3 here: {gate:.1%})"
    )
    if args.store == "heldout100":
        print("\033[33mheldout100 is spent — this run checks the code, and its numbers")
        print("may not be used to choose M. That decision belongs to train150.\033[0m")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "store": args.store,
                    "questions": args.questions,
                    "complete": gate_complete,
                    "ingest_complete": complete,
                    "manifest_questions": len(manifest.question_ids),
                    "ingested_questions": len(question_ids),
                    "evaluated_questions": n,
                    "skipped_without_gold": len(skipped),
                    "n": n,
                    "memory_limit": memory_limit,
                    "candidate_limit": svc.config.retrieval.candidate_limit,
                    "context_budget": {
                        "max_sessions": svc.config.context.max_sessions,
                        "window_radius": window_radius,
                        "max_total_memories": max_total_memories,
                        "session_order": svc.config.context.session_order,
                        "include_superseded": svc.config.context.include_superseded,
                    },
                    "selected_aggregate": selected_aggregate,
                    "recall_at": {str(m): hits_at[selected_aggregate][m] / n for m in MS},
                    "recall_by_aggregate": {
                        aggregate: {str(m): hits_at[aggregate][m] / n for m in MS}
                        for aggregate in AGGREGATES
                    },
                    "assembled_by_aggregate": {
                        aggregate: {
                            "recall": assembled_hits[aggregate] / n,
                            "median_memories": statistics.median(assembled_memories[aggregate]),
                            "median_tokens": statistics.median(assembled_tokens[aggregate]),
                            "truncated_questions": truncated[aggregate],
                        }
                        for aggregate in AGGREGATES
                    },
                    "flat_median_tokens": statistics.median(flat_tokens),
                    "memory_level": memory_level / n,
                    "rows": rows,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
