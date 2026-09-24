"""Both extraction rulers, read off a built store rather than a fresh extraction.

`lltm ingest fidelity` extracts a cohort and scores what came back. That is the right
instrument for comparing extractors, and the wrong one for asking what a *finished
ingest* retained: a store has been through deduplication, supersession and the repair,
and every one of those can remove what extraction produced. Option C of
`results/measurement-ceiling-decision.md` is accepted on what the store holds, so this
reads the store.

For each question in a manifest it takes that question's haystack sessions — the same
`(namespace, session)` pairs ingestion wrote, deduplicated within a namespace exactly as
`namespaced_sessions` does — finds the memories whose `source_session_id` is that
session's scoped id, and scores them with the recall ruler (`ingest.fidelity`) and the
support ruler (`ingest.precision`). Superseded memories are included: a fact that was
retained and later replaced was retained.

Per-specific outcomes are written too, so two stores built over the same manifest can be
compared pairwise rather than by headline (`--pair-with`). `--exclude-prefix g_` scores a
store without its repaired memories, which separates what the repair added from how far
a second extraction drifted from the first: the repair's ids start `g_` and the
extractor's `mem_`.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import read_json, write_report  # noqa: E402


def pair(before: dict[str, bool], after: dict[str, bool]) -> dict:
    """Specifics kept by one measurement and not the other, over the keys both scored."""
    shared = before.keys() & after.keys()
    return {
        "shared": len(shared),
        "gained": sum(1 for key in shared if after[key] and not before[key]),
        "lost": sum(1 for key in shared if before[key] and not after[key]),
        "only_before": len(before.keys() - after.keys()),
        "only_after": len(after.keys() - before.keys()),
    }


def measure(store_path: Path, manifest_path: Path, exclude_prefix: str | None = None) -> dict:
    from llm_long_term_memory.evaluation.datasets import longmemeval as lme
    from llm_long_term_memory.ingest.fidelity import (
        _extract_facets,
        score_sessions,
        user_assertions,
    )
    from llm_long_term_memory.ingest.pipeline import namespaced_sessions
    from llm_long_term_memory.ingest.precision import score_support
    from llm_long_term_memory.store import SQLiteMemoryStore, scoped_session_id

    manifest = read_json(manifest_path)
    instances = {item.question_id: item for item in lme.load(manifest["variant"], REPO / "data")}
    chosen = [instances[qid] for qid in manifest["question_ids"]]
    pairs = namespaced_sessions(chosen)

    store = SQLiteMemoryStore(store_path, read_only=True)
    store.initialize()
    try:
        by_scoped: dict[str, list] = defaultdict(list)
        for namespace in {namespace for namespace, _ in pairs}:
            for memory in store.iter_all(namespace):
                if exclude_prefix and memory.id.startswith(exclude_prefix):
                    continue
                if memory.source_session_id:
                    by_scoped[memory.source_session_id].append(memory)
    finally:
        store.close()

    scored = []
    outcomes = {}
    for namespace, session in pairs:
        memories = by_scoped.get(scoped_session_id(namespace, session.session_id), [])
        scored.append((session, memories))
        blob = " || ".join(f"{m.content} {m.object or ''}" for m in memories).lower()
        for facet, values in _extract_facets(user_assertions(session)).items():
            for value in values:
                outcomes[f"{namespace}|{session.session_id}|{facet}|{value}"] = value in blob

    recall = score_sessions(scored)
    support = score_support(scored)
    return {
        "store": str(store_path.relative_to(REPO))
        if store_path.is_relative_to(REPO)
        else str(store_path),
        "manifest": str(manifest_path.relative_to(REPO)),
        "exclude_prefix": exclude_prefix,
        "sessions": len(pairs),
        "memories": support.memories,
        "memories_per_session": support.memories / max(1, len(pairs)),
        "recall": recall.overall,
        "specifics": len(outcomes),
        "specifics_kept": sum(outcomes.values()),
        "per_facet_recall": {
            name: score.recall for name, score in sorted(recall.per_facet.items())
        },
        "checkable_memories": support.memories_with_a_checkable_specific,
        "unsupported_memories": support.memories_with_an_unsupported_specific,
        "unsupported_rate": support.unsupported_memory_rate,
        "specific_support": support.specific_support_rate,
        "outcomes": outcomes,
    }


def render(result: dict) -> str:
    kept_per_memory = result["specifics_kept"] / max(1, result["memories"])
    excluded = result.get("exclude_prefix")
    lines = [
        f"# Extraction rulers on `{result['store']}`",
        "",
        f"> Manifest `{result['manifest']}`. Read from the built store: after deduplication,",
        "> supersession and any repair. No model calls.",
        *([f"> Memories whose id starts `{excluded}` are excluded."] if excluded else []),
        "",
        f"- sessions: **{result['sessions']:,}**, memories: **{result['memories']:,}** "
        f"({result['memories_per_session']:.2f} per session)",
        f"- recall: **{result['recall']:.1%}** — {result['specifics_kept']:,} of "
        f"{result['specifics']:,} specifics kept, {kept_per_memory:.3f} per memory",
        f"- support: **{result['specific_support']:.1%}** of asserted specifics are in the "
        f"conversation; {result['unsupported_memories']} of {result['checkable_memories']} "
        f"checkable memories carry one that is not ({result['unsupported_rate']:.1%})",
        "",
        "| facet | recall |",
        "|---|---:|",
    ]
    lines += [f"| {name} | {value:.1%} |" for name, value in result["per_facet_recall"].items()]
    if "paired" in result:
        paired = result["paired"]
        lines += [
            "",
            f"Paired against `{paired['against']}` over {paired['shared']:,} shared specifics: "
            f"**+{paired['gained']} gained, −{paired['lost']} lost** "  # noqa: RUF001
            f"({paired['only_before']} only there, {paired['only_after']} only here).",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("store", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument("--exclude-prefix", default=None)
    parser.add_argument(
        "--pair-with", type=Path, default=None, help="an earlier --json-out to pair against"
    )
    args = parser.parse_args()
    store = args.store if args.store.is_absolute() else (REPO / args.store).resolve()
    manifest = args.manifest if args.manifest.is_absolute() else (REPO / args.manifest).resolve()
    if not store.is_file():
        raise SystemExit(f"{store} does not exist")
    result = measure(store, manifest, exclude_prefix=args.exclude_prefix)
    if args.pair_with is not None:
        earlier = read_json(args.pair_with)
        result["paired"] = {
            "against": str(args.pair_with),
            **pair(earlier["outcomes"], result["outcomes"]),
        }
    print(write_report(result, render, json_out=args.json_out, md_out=args.md_out), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
