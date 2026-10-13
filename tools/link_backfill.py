"""Stage D on an existing store: run the write path's `UpdateLinker` over every user.

    python tools/link_backfill.py dev100        # resumable; stops cleanly on daily quota

Copies `stores/<split>.db` and its index to `stores/<split>-d1.*` (SQLite backup API,
the original untouched) and links each user's memories exactly as
`ingest.update_linking` would: user-stated memories, candidates of any date with
similarity >= 0.65, one judgment per pair (`transition-pair-v3`), confirmed replacements
moved onto the earlier fact's key and resolved. The copy starts from the original store,
not the anchor-v2 copy, so an end-to-end comparison against t4 changes one thing.

Judgments go through a response cache, so a run stopped by quota resumes without
paying again; progress is kept per user in `stores/<split>-d1-link.json`.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

STORES = REPO / "stores"


def _prepare(split: str) -> Path:
    target = STORES / f"{split}-d1.db"
    if not target.exists():
        with (
            sqlite3.connect(f"file:{STORES / f'{split}.db'}?mode=ro", uri=True) as src,
            sqlite3.connect(target) as dst,
        ):
            src.backup(dst)
        for ext in ("-index.npy", "-index.ids.json"):
            shutil.copyfile(STORES / f"{split}{ext}", STORES / f"{split}-d1{ext}")
    return target


def backfill(split: str) -> dict:
    from llm_long_term_memory.config import ExperimentConfig, Settings
    from llm_long_term_memory.llm import Limits, QuotaManager, UsageTracker
    from llm_long_term_memory.llm.cache import CachingClient
    from llm_long_term_memory.llm.client import DailyQuotaExhausted, GeminiClient
    from llm_long_term_memory.store import NumpyFlatIndex, SQLiteMemoryStore
    from llm_long_term_memory.temporal import TemporalResolver
    from llm_long_term_memory.temporal.transition import UpdateLinker

    settings = Settings()
    cfg = ExperimentConfig.from_yaml(REPO / "configs/fallback.yaml")
    quota = QuotaManager(
        state_dir=settings.store_dir / "quota",
        default=Limits(rpm=cfg.quota.rpm, tpm=cfg.quota.tpm, rpd=cfg.quota.rpd),
    )
    client = CachingClient(
        GeminiClient(settings.require_api_key(), quota=quota, usage=UsageTracker()),
        STORES / "cache" / "link-backfill-d1.db",
        sample="rep1",
        scope={"split": split},
    )
    linker = UpdateLinker(client, cfg.models.extractor)
    progress_path = STORES / f"{split}-d1-link.json"
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.exists()
        else {"identity": linker.identity, "done": {}, "stopped": None}
    )
    if progress["identity"] != linker.identity:
        raise SystemExit("the linker changed since this backfill started; start a new copy")
    store = SQLiteMemoryStore(_prepare(split))
    index = NumpyFlatIndex(STORES / f"{split}-d1-index", dim=384)
    resolver = TemporalResolver(store)
    qids = json.loads((REPO / f"results/manifests/{split}.json").read_text(encoding="utf-8"))[
        "question_ids"
    ]
    try:
        for qid in qids:
            if qid in progress["done"]:
                continue
            memories = sorted(
                store.iter_all(qid), key=lambda m: (m.occurred_at is None, m.occurred_at, m.id)
            )
            try:
                stats = linker.link(store, index, resolver, memories)
            except DailyQuotaExhausted as exc:
                progress["stopped"] = f"daily quota at {qid}: {exc}"
                break
            progress["done"][qid] = stats
            progress["stopped"] = None
            progress_path.write_text(json.dumps(progress, indent=1) + "\n", encoding="utf-8")
    finally:
        progress_path.write_text(json.dumps(progress, indent=1) + "\n", encoding="utf-8")
        store.close()
    done = progress["done"].values()
    return {
        "split": split,
        "users_done": len(progress["done"]),
        "users": len(qids),
        "judgments": sum(s["judgments"] for s in done),
        "linked": sum(s["linked"] for s in done),
        "stopped": progress["stopped"],
    }


if __name__ == "__main__":
    result = backfill(sys.argv[1])
    print(json.dumps(result, indent=1))
    raise SystemExit(3 if result["stopped"] else 0)
