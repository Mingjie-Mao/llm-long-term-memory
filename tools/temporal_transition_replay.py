"""Zero-call regression replay for explicit temporal transitions.

The source database is immutable input. SQLite's backup API copies it to a temporary
database, the current resolver runs there twice, and the report records every state
change plus the idempotence check. Nothing under results/frozen or results/sealed is
modified.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import write_report  # noqa: E402

from llm_long_term_memory.store import SQLiteMemoryStore  # noqa: E402
from llm_long_term_memory.temporal import TemporalResolver  # noqa: E402


def _states(store: SQLiteMemoryStore) -> dict[str, dict]:
    out = {}
    for user_id in store.user_ids():
        for memory in store.iter_all(user_id):
            out[memory.id] = {
                "user_id": memory.user_id,
                "content": memory.content,
                "predicate": memory.predicate,
                "object": memory.object,
                "target_object": memory.target_object,
                "update_op": memory.update_op,
                "status": memory.status,
                "superseded_by": memory.superseded_by,
                "valid_to": memory.valid_to.isoformat() if memory.valid_to else None,
            }
    return out


def _copy(source: Path, destination: Path) -> None:
    original = sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True)
    clone = sqlite3.connect(destination)
    try:
        original.backup(clone)
    finally:
        clone.close()
        original.close()


def analyse(source: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="lltm-transition-replay-") as directory:
        candidate_path = Path(directory) / "candidate.db"
        _copy(source, candidate_path)
        store = SQLiteMemoryStore(candidate_path)
        store.initialize()
        try:
            before = _states(store)
            first = TemporalResolver(store).resolve_everything()
            after = _states(store)
            second = TemporalResolver(store).resolve_everything()
            replayed = _states(store)
        finally:
            store.close()

    changed = [
        {"id": memory_id, "before": before[memory_id], "after": after[memory_id]}
        for memory_id in sorted(before)
        if before[memory_id] != after[memory_id]
    ]
    removals = [row for row in after.values() if row["update_op"] == "removes"]
    legacy_nike = [
        row for row in after.values() if "replaced their Nike Air Zoom Pegasus 38" in row["content"]
    ]
    adidas = [row for row in after.values() if "Adidas Ultraboost 22" in row["content"]]
    gates = {
        "no_active_removal": all(row["status"] != "active" for row in removals),
        "known_nike_is_not_current": all(row["status"] != "active" for row in legacy_nike),
        "known_adidas_is_current": bool(adidas)
        and all(row["status"] == "active" for row in adidas),
        "idempotent_second_replay": after == replayed and second.writes == 0,
    }
    return {
        "experiment": "temporal-transition-replay-v1",
        "class": "regression",
        "source_store": str(source.relative_to(REPO)),
        "model_calls": 0,
        "memories": len(before),
        "changed": changed,
        "changed_count": len(changed),
        "first_replay": {
            "writes": first.writes,
            "superseded": first.superseded,
            "terminated": first.terminated,
            "ambiguous": first.skipped_ambiguous,
        },
        "second_replay_writes": second.writes,
        "gates": gates,
        "decision": "PASS" if all(gates.values()) else "FAIL",
    }


def render(result: dict) -> str:
    lines = [
        "# Temporal transition replay v1",
        "",
        "> Regression evidence over an inspected development store; zero model calls.",
        "",
        f"Decision: **{result['decision']}**",
        "",
        f"- memories: **{result['memories']:,}**",
        f"- changed states: **{result['changed_count']}**",
        f"- first replay writes: **{result['first_replay']['writes']}**",
        f"- ambiguous transitions: **{result['first_replay']['ambiguous']}**",
        f"- second replay writes: **{result['second_replay_writes']}**",
        "",
        "## Gates",
        "",
    ]
    lines += [
        f"- {'PASS' if value else 'FAIL'} — `{name}`" for name, value in result["gates"].items()
    ]
    lines += ["", "## Changed states", ""]
    if not result["changed"]:
        lines.append("None.")
    for change in result["changed"]:
        lines += [
            f"### `{change['id']}`",
            "",
            f"- content: {change['after']['content']}",
            f"- before: `{json.dumps(change['before'], ensure_ascii=False, sort_keys=True)}`",
            f"- after: `{json.dumps(change['after'], ensure_ascii=False, sort_keys=True)}`",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("store", type=Path)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    args = parser.parse_args()
    source = args.store if args.store.is_absolute() else (REPO / args.store).resolve()
    result = analyse(source)
    print(write_report(result, render, json_out=args.json_out, md_out=args.md_out), end="")
    return 0 if result["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
