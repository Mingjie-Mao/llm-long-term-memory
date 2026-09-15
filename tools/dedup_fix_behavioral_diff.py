"""Is the namespace-scoped dedup fix a measurement fix or a system change?

A fix that changes only bookkeeping can be applied to the frozen system and recorded as an
amendment. A fix that changes **which memories reach the model** is a new arm, and calling
it `v2` afterwards would be the thing freezing exists to prevent. The difference is not a
matter of opinion, so this measures it: the same conversations are ingested twice, once
with `Deduplicator._neighbours` as frozen and once with a namespace condition, and the two
runs are compared on what actually reaches the answerer —

    the memories written  ·  ranked_memory_ids  ·  the memories put in the context

Zero provider calls: a fake provider answers both runs identically, so any difference is
the fix and nothing else. That is the same construction Gate 0 uses, one layer down.

**What a dry run cannot settle, and says so.** Whether a cross-namespace pair is called
DUPLICATE is a model decision, and there is no model here. So two arms are measured:

    as-is        the fake's own verdict rule, which calls almost nothing a duplicate
    worst case   every cross-namespace adjudication forced to DUPLICATE

The first says whether the fix is inert on this data under a benign model; the second
bounds how much a hostile draw could move, and it is the bound that matters, because the
frozen store was built by a real model whose verdicts nobody recorded pairwise.

    python3 tools/dedup_fix_behavioral_diff.py --conversations 2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

OUT = REPO / "results/analysis/dedup-fix-behavioral-diff.json"


def namespace_scoped(original):
    """`_neighbours`, with the condition retrieval already applies and dedup does not."""

    def patched(self, candidate, vector, seen_vectors, seen_memories):
        found = original(self, candidate, vector, seen_vectors, seen_memories)
        return [memory for memory in found if memory.user_id == candidate.user_id]

    return patched


def force_cross_namespace_duplicates(original):
    """The worst a model could do: every cross-conversation pair called a duplicate."""

    def patched(self, old, new):
        if old.user_id != new.user_id:
            from llm_long_term_memory.ingest.schemas import DedupDecision

            return DedupDecision(verdict="DUPLICATE", reason="forced: worst-case probe")
        return original(self, old, new)

    return patched


def run_arm(cli, client_module, settings, args, manifest, label, *, scoped, hostile):
    """Ingest and answer once, under one dedup configuration, and report what reached
    the model."""
    import beam_dry_run

    from llm_long_term_memory.evaluation.beam_report import load_rows
    from llm_long_term_memory.ingest.dedup import Deduplicator
    from llm_long_term_memory.store import SQLiteMemoryStore

    store_name = f"dedup-diff-{label}"
    neighbours, adjudicate = Deduplicator._neighbours, Deduplicator.adjudicate
    crossings = []

    def watched(self, old, new):
        if old.user_id != new.user_id:
            crossings.append((old.user_id, new.user_id))
        return adjudicate(self, old, new)

    Deduplicator.adjudicate = force_cross_namespace_duplicates(watched) if hostile else watched
    if scoped:
        Deduplicator._neighbours = namespace_scoped(neighbours)
    try:
        client_module.GeminiClient = lambda *a, **k: beam_dry_run.FakeProvider()
        cli.ingest_run(
            config=args.config,
            limit=None,
            sessions=None,
            fresh=True,
            store_name=store_name,
            questions=str(manifest),
        )
        cli.eval_run(
            variant=args.variant,
            config=args.config,
            limit=None,
            questions=str(manifest),
            store_name=store_name,
            fresh=True,
            top_k=None,
            rerank=None,
            label=f"dedup-diff-{label}",
        )
    finally:
        Deduplicator._neighbours, Deduplicator.adjudicate = neighbours, adjudicate

    store = SQLiteMemoryStore(settings.store_dir / f"{store_name}.db", read_only=True)
    store.initialize()
    written = {
        namespace: sorted(m.content for m in store.iter_all(namespace))
        for namespace in sorted(store.user_ids())
    }
    store.close()

    rows = load_rows(settings.results_dir / "raw" / f"{args.variant}.dedup-diff-{label}.jsonl")
    reaching = {
        row["question_id"]: {
            "ranked": list((row.get("notes") or {}).get("ranked_memory_ids") or []),
            "context": [
                hit["memory_id"] for hit in (row.get("notes") or {}).get("retrieval") or []
            ],
            "context_tokens": row.get("context_tokens"),
        }
        for row in rows
    }
    return {
        "memories": written,
        "memory_count": sum(len(v) for v in written.values()),
        "reaching": reaching,
        "cross_namespace_adjudications": len(crossings),
    }


def diff(before: dict, after: dict) -> dict:
    """What changed in the memories written and in what reached the answerer."""
    dropped = {}
    for namespace, contents in after["memories"].items():
        gained = sorted(set(contents) - set(before["memories"].get(namespace, [])))
        if gained:
            dropped[namespace] = gained
    questions = sorted(set(before["reaching"]) | set(after["reaching"]))
    ranked_changed = [
        q
        for q in questions
        if before["reaching"].get(q, {}).get("ranked") != after["reaching"].get(q, {}).get("ranked")
    ]
    context_changed = [
        q
        for q in questions
        if before["reaching"].get(q, {}).get("context")
        != after["reaching"].get(q, {}).get("context")
    ]
    identical = not dropped and not ranked_changed and not context_changed
    return {
        "memories_before": before["memory_count"],
        "memories_after": after["memory_count"],
        "memories_the_fix_saved": sum(len(v) for v in dropped.values()),
        "conversations_affected": sorted(dropped),
        "questions": len(questions),
        "questions_with_different_ranked_memory_ids": len(ranked_changed),
        "questions_with_a_different_context": len(context_changed),
        "identical": identical,
        "verdict": (
            "measurement fix: nothing that reaches the model changed"
            if identical
            else "system change: what reaches the model differs, so this is a second arm"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", type=int, default=2)
    parser.add_argument("--config", default="configs/v2.yaml")
    parser.add_argument("--variant", default="two_stage_fallback")
    parser.add_argument("--work", default="stores/dedup-fix-diff")
    args = parser.parse_args()

    work = (REPO / args.work).resolve()
    if work.exists():
        shutil.rmtree(work)
    (work / "stores").mkdir(parents=True, exist_ok=True)
    (work / "results").mkdir(parents=True, exist_ok=True)
    os.environ["LLTM_STORE_DIR"] = str(work / "stores")
    os.environ["LLTM_RESULTS_DIR"] = str(work / "results")
    os.environ.setdefault("GEMINI_API_KEY", "dry-run-no-request-is-sent")

    import beam_dry_run

    from llm_long_term_memory import cli
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.llm import client as client_module

    settings = Settings()
    manifest, instances = beam_dry_run.manifest_for(args.conversations, work, settings.data_dir)

    arms = {}
    for name, scoped, hostile in (
        ("frozen", False, False),
        ("scoped", True, False),
        ("frozen-worst-case", False, True),
        ("scoped-worst-case", True, True),
    ):
        arms[name] = run_arm(
            cli, client_module, settings, args, manifest, name, scoped=scoped, hostile=hostile
        )

    payload = {
        "name": "dedup-fix-behavioral-diff",
        "provider_calls": 0,
        "question": "does namespace-scoping deduplication change what reaches the answerer?",
        "conversations": args.conversations,
        "questions": len(instances),
        "cross_namespace_adjudications": arms["frozen"]["cross_namespace_adjudications"],
        "under_the_dry_run_verdicts": diff(arms["frozen"], arms["scoped"]),
        "under_forced_duplicate_verdicts": diff(
            arms["frozen-worst-case"], arms["scoped-worst-case"]
        ),
        "reading": (
            "The first comparison uses the dry run's own verdict rule, which calls a pair a "
            "duplicate only when the two texts are identical — a benign model. The second "
            "forces every cross-conversation pair to DUPLICATE, which is what the fix exists "
            "to prevent and what a real model may do on a seed-twin pair. A real store was "
            "built by a real model whose pairwise verdicts were never recorded, so the second "
            "comparison is the one that decides whether this can be called the frozen system."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "reading"}, indent=2))
    print(f"\nwritten: {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
