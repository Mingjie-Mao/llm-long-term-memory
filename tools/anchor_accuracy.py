"""How often does a memory point at the turn that actually says it? No calls, no gold.

    python tools/anchor_accuracy.py                 # train150
    python tools/anchor_accuracy.py --store dev100 --questions dev100.json

Registered in `results/prereg-anchor-v2-offline-v1.md`. A memory is checkable when one
of its specifics appears in exactly one turn of its source session; that turn is the
reference. Compares the stored anchor, v1 recomputed, and v2.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import write_report  # noqa: E402

MARGIN = 0.03
KEEP = 0.99


def reference_turn(content: str, session) -> int | None:
    """The one turn that carries this memory's specific, or None when it is ambiguous."""
    from llm_long_term_memory.ingest.fidelity import _extract_facets

    specifics = {
        v.rstrip(",.;:!?").strip() for values in _extract_facets(content).values() for v in values
    } - {""}
    turns: set[int] = set()
    for value in specifics:
        holders = [i for i, t in enumerate(session.turns) if value in t.content.lower()]
        if len(holders) == 1:
            turns.add(holders[0])
    return turns.pop() if len(turns) == 1 else None


def analyse(store_name: str, manifest_name: str) -> dict:
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.evaluation.datasets import longmemeval as lme
    from llm_long_term_memory.evaluation.manifest import load_manifest
    from llm_long_term_memory.ingest.provenance import source_span_for, source_span_for_v2
    from llm_long_term_memory.store import SQLiteMemoryStore, external_session_id

    settings = Settings()
    manifest = load_manifest(settings.results_dir / "manifests" / manifest_name)
    instances = {i.question_id: i for i in lme.load(manifest.variant, settings.data_dir)}
    store = SQLiteMemoryStore(settings.store_dir / f"{store_name}.db", read_only=True)
    store.initialize()
    counts = Counter()
    by_role = Counter()
    examples = []
    try:
        for qid in manifest.question_ids:
            sessions = {s.session_id: s for s in instances[qid].sessions}
            for memory in store.iter_all(qid):
                if not memory.source_session_id or memory.source_turn_index is None:
                    continue
                session = sessions.get(external_session_id(memory.source_session_id))
                if session is None:
                    continue
                reference = reference_turn(memory.content, session)
                if reference is None:
                    continue
                counts["checkable"] += 1
                v1 = source_span_for(memory.content, session)
                v2 = source_span_for_v2(memory.content, session)
                v3 = source_span_for_v2(memory.content, session, role=memory.source_role)
                v3_ok = v3 is not None and v3[0] == reference
                counts["v3_correct"] += v3_ok
                counts["v1_ok_v3_ok"] += (v1 is not None and v1[0] == reference) and v3_ok
                stored_ok = memory.source_turn_index == reference
                v1_ok = v1 is not None and v1[0] == reference
                v2_ok = v2 is not None and v2[0] == reference
                counts["stored_correct"] += stored_ok
                counts["v1_correct"] += v1_ok
                counts["v2_correct"] += v2_ok
                counts["v1_equals_stored"] += v1 is not None and v1[0] == memory.source_turn_index
                counts["v1_ok_v2_ok"] += v1_ok and v2_ok
                counts["v1_ok_v2_wrong"] += v1_ok and not v2_ok
                counts["v1_wrong_v2_ok"] += (not v1_ok) and v2_ok
                by_role[(session.turns[reference].role, v2_ok)] += 1
                if not v1_ok and v2_ok and len(examples) < 5:
                    examples.append(
                        {
                            "content": memory.content[:140],
                            "v1_turn": v1[0] if v1 else None,
                            "reference_turn": reference,
                        }
                    )
    finally:
        store.close()
    n = counts["checkable"]
    accuracy = {k: counts[f"{k}_correct"] / n for k in ("stored", "v1", "v2", "v3")}
    kept = counts["v1_ok_v2_ok"] / max(1, counts["v1_correct"])
    rules = {
        "v2_at_least_3_points_above_v1": accuracy["v2"] - accuracy["v1"] >= MARGIN,
        "v2_keeps_99pct_of_v1_correct": kept >= KEEP,
    }
    return {
        "store": store_name,
        "checkable": n,
        "accuracy": accuracy,
        "v1_equals_stored": counts["v1_equals_stored"] / n,
        "flips": {
            "v1_wrong_v2_right": counts["v1_wrong_v2_ok"],
            "v1_right_v2_wrong": counts["v1_ok_v2_wrong"],
        },
        "v2_kept_of_v1_correct": kept,
        "v3_kept_of_v1_correct": counts["v1_ok_v3_ok"] / max(1, counts["v1_correct"]),
        "v2_accuracy_by_reference_role": {
            role: by_role[(role, True)] / max(1, by_role[(role, True)] + by_role[(role, False)])
            for role in ("user", "assistant")
        },
        "rules": rules,
        "switch_to_v2": all(rules.values()),
        "examples": examples,
    }


def render(result: dict) -> str:
    a, f = result["accuracy"], result["flips"]
    lines = [
        f"# Memory anchor accuracy — `{result['store']}`",
        "",
        "> Zero calls, no gold labels. Registered in `results/prereg-anchor-v2-offline-v1.md`.",
        "> Reference: the only turn in the source session that carries one of the memory's",
        "> specifics.",
        "",
        f"Checkable memories: **{result['checkable']:,}**. "
        f"v1 recomputed equals the stored anchor for {result['v1_equals_stored']:.1%}.",
        "",
        "| anchor | accuracy |",
        "|---|---:|",
        f"| stored | {a['stored']:.1%} |",
        f"| v1 recomputed | {a['v1']:.1%} |",
        f"| v2 | {a['v2']:.1%} |",
        f"| **v3** (v2 among the memory's own speaker's turns) | **{a['v3']:.1%}** |",
        "",
        f"v1 wrong, v2 right: {f['v1_wrong_v2_right']}; v1 right, v2 wrong: "
        f"{f['v1_right_v2_wrong']} (v2 keeps {result['v2_kept_of_v1_correct']:.1%} of v1's "
        "correct anchors).",
        "",
    ]
    lines += [f"- {'PASS' if ok else 'FAIL'} — `{k}`" for k, ok in result["rules"].items()]
    lines += [
        "",
        f"**Decision: {'switch new ingests to v2' if result['switch_to_v2'] else 'keep v1'}**",
    ]
    if result["examples"]:
        lines += ["", "Fixed by v2:"]
        lines += [
            f"- turn {e['v1_turn']} -> {e['reference_turn']}: {e['content']}"
            for e in result["examples"]
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default="train150")
    parser.add_argument("--questions", default="train150.json")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args()
    result = analyse(args.store, args.questions)
    print(write_report(result, render, json_out=args.json_out, md_out=args.md_out), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
