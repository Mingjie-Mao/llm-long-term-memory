"""Migrate a partial LongMemEval store to user-scoped session primary keys.

Dry-run is the default. ``--apply`` creates a SQLite-consistent backup before any
row changes, then preserves all memories and vector ids while rebuilding the raw
session archive from the dataset.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from llm_long_term_memory.config import Settings  # noqa: E402
from llm_long_term_memory.evaluation.datasets import longmemeval as lme  # noqa: E402
from llm_long_term_memory.evaluation.manifest import load_manifest  # noqa: E402
from llm_long_term_memory.ingest.pipeline import _key, namespaced_sessions  # noqa: E402
from llm_long_term_memory.ingest.session_migration import migrate_scoped_sessions  # noqa: E402
from llm_long_term_memory.store import SQLiteMemoryStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default="train150")
    parser.add_argument("--questions", default="results/manifests/train150.json")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--reuse-backup",
        action="store_true",
        help="apply an idempotent cleanup while retaining an existing migration backup",
    )
    args = parser.parse_args()

    settings = Settings()
    store_path = settings.store_dir / f"{args.store}.db"
    checkpoint_path = settings.store_dir / f"{args.store}-ingest.json"
    progress = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    done = set(progress.get("done_sessions", []))
    manifest = load_manifest(args.questions)
    by_id = {item.question_id: item for item in lme.load(manifest.variant, settings.data_dir)}
    pairs = [
        pair
        for pair in namespaced_sessions([by_id[qid] for qid in manifest.question_ids])
        if _key(*pair) in done
    ]
    print(f"completed checkpoint rows: {len(done):,}; matched dataset sessions: {len(pairs):,}")
    if len(pairs) != len(done):
        print("refusing: checkpoint and dataset do not describe the same completed work")
        return 2
    if not args.apply:
        print("dry run only; re-run with --apply to create a backup and migrate")
        return 0

    backup_path = store_path.with_name(f"{store_path.stem}.pre-scoped-backup.db")
    if backup_path.exists():
        if not args.reuse_backup:
            print(f"refusing to overwrite existing backup: {backup_path}")
            return 2
        print(f"retaining existing backup: {backup_path}")
    else:
        source = sqlite3.connect(store_path)
        backup = sqlite3.connect(backup_path)
        try:
            source.backup(backup)
        finally:
            backup.close()
            source.close()

    store = SQLiteMemoryStore(store_path)
    store.initialize()
    try:
        report = migrate_scoped_sessions(store, pairs)
    finally:
        store.close()
    print(report)
    print(f"backup: {backup_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
