"""Put the recovered facts back, and see whether the answers change.

    python scripts/counterfactual_qa.py            # plan, no calls
    python scripts/counterfactual_qa.py --run      # build the overlay and evaluate

The targeted diagnostic showed that six of seven facts production lost at
extraction come back when the same sessions are extracted one per request. That
establishes the loss. It does not establish that recovering it would answer
anything, and one result warns against assuming it would: recovering "the user
recently finished a 416-page novel" also produced three assistant recommendations
for books that happen to be 416 pages long. More facts is also more to confuse.

So this builds the counterfactual store and runs the real thing over it.

    stores/cf-overlay.db  =  the clean P10 store
                             minus the memories those eight sessions produced at
                             batch 15
                             plus the memories they produce at batch 1

Nothing else differs — same turns, same other namespaces, same config, same
answerer and judge, same retrieval. `two-stage-p10.db` is copied, never modified.

All seven questions are evaluated, including `73d42213`, whose fact batch 1 did
*not* recover. It is the control: if its answer changes, something other than the
intended edit is moving.

What this can and cannot show. A question that flips to correct is evidence that
batched extraction cost that answer. A question that stays wrong is not evidence
that the fact was useless — retrieval still has to surface it and the answerer
still has to use it, and the layer that fails next is worth knowing either way.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from llm_long_term_memory.config import ExperimentConfig, Settings  # noqa: E402
from llm_long_term_memory.evaluation.datasets import longmemeval as lme  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
from targeted_batch1 import WANTED  # noqa: E402

OVERLAY = "cf-overlay"
BASELINE = REPO / "results" / "raw" / "two_stage_hydrated.a2-clean-p10-fallback-v2.jsonl"
MANIFEST = REPO / "results" / "manifests" / "cf-recovered.json"
LABEL = "cf-overlay"


def build_overlay(settings, cfg, sessions_by_id) -> dict:
    """Copy the store, swap in batch=1 memories for the eight sessions, reindex."""
    from llm_long_term_memory.embed import Encoder
    from llm_long_term_memory.ingest import TwoStageExtractor
    from llm_long_term_memory.llm import Limits, QuotaManager, UsageTracker
    from llm_long_term_memory.llm.client import GeminiClient
    from llm_long_term_memory.store import NumpyFlatIndex, SQLiteMemoryStore

    src = settings.store_dir / "two-stage-p10.db"
    dst = settings.store_dir / f"{OVERLAY}.db"
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)
    print(f"copied {src.name} -> {dst.name}")

    quota = QuotaManager(
        state_dir=settings.store_dir / "quota",
        default=Limits(rpm=cfg.quota.rpm, tpm=cfg.quota.tpm, rpd=cfg.quota.rpd),
    )
    quota.load_learned()
    usage = UsageTracker()
    client = GeminiClient(settings.require_api_key(), quota=quota, usage=usage)
    extractor = TwoStageExtractor(client, cfg.models.extractor)

    store = SQLiteMemoryStore(dst)
    store.initialize()
    swapped = {}
    try:
        for question_id, (_, _, session_ids) in WANTED.items():
            for session_id in session_ids:
                before = store._conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE source_session_id=? AND user_id=?",
                    (session_id, question_id),
                ).fetchone()[0]
                outcome = extractor.extract([sessions_by_id[session_id]])
                fresh = [m for m in outcome.memories if m.source_session_id == session_id]
                # The extractor stamps its own `user_id`; these belong to the
                # question's namespace, exactly as ingestion would have filed them.
                for memory in fresh:
                    memory.user_id = question_id
                store._conn.execute(
                    "DELETE FROM memories WHERE source_session_id=? AND user_id=?",
                    (session_id, question_id),
                )
                store._conn.commit()
                store.add_memories(fresh)
                swapped[f"{question_id}/{session_id}"] = {"before": before, "after": len(fresh)}
                print(f"  {question_id}/{session_id}: {before} -> {len(fresh)} memories")

        # No removal on the flat index, so it is rebuilt from the rows. Local
        # encoder, no quota, and it guarantees the index and the store agree —
        # which a partial update would not.
        print("re-encoding the whole store …")
        encoder = Encoder(cfg.models.embedder)
        rows = store._conn.execute("SELECT id, content FROM memories").fetchall()
        index_path = settings.store_dir / f"{OVERLAY}-index"
        for suffix in (".npy", ".ids.json"):
            p = Path(str(index_path) + suffix)
            if p.exists():
                p.unlink()
        index = NumpyFlatIndex(index_path, dim=cfg.models.embedding_dim)
        vectors = encoder.encode([c for _, c in rows], show_progress=True)
        index.add([i for i, _ in rows], vectors)
        index.save()
        print(f"index rebuilt: {len(index)} vectors over {len(rows)} memories")
    finally:
        store.close()
    return {"swapped": swapped, "extractor_requests": usage.summary().get("total_requests")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    settings = Settings()
    cfg = ExperimentConfig.from_yaml(str(REPO / "configs" / "baselines.yaml"))
    question_ids = sorted(WANTED)
    sessions = sum(len(v[2]) for v in WANTED.values())

    print(f"counterfactual: {len(question_ids)} questions, {sessions} sessions re-extracted")
    print(f"  ~{sessions * 2} extractor requests + ~{len(question_ids) * 2} answerer/judge")
    for q in question_ids:
        print(f"    {q:14s} {WANTED[q][0]}")
    if not args.run:
        print("\nRe-run with --run to start.")
        return 0

    instances = lme.load(cfg.dataset_variant, settings.data_dir, limit=cfg.dataset_limit)
    sessions_by_id = {s.session_id: s for i in instances for s in i.sessions}

    report = build_overlay(settings, cfg, sessions_by_id)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(
            {
                "name": "cf-recovered",
                "note": (
                    "The seven dev50 questions whose gold sessions were re-extracted at "
                    "batch=1 for the counterfactual overlay. Not a sample of anything — "
                    "these are exactly the questions the targeted diagnostic covered."
                ),
                "question_ids": question_ids,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nevaluating over the overlay …")
    result = subprocess.run(
        [
            str(REPO / ".venv" / "bin" / "python"),
            "-m",
            "llm_long_term_memory.cli",
            "eval",
            "run",
            "two_stage_hydrated",
            "--config",
            "configs/fallback.yaml",
            "--store-name",
            OVERLAY,
            "--questions",
            str(MANIFEST),
            "--label",
            LABEL,
        ],
        cwd=REPO,
    )
    if result.returncode != 0:
        print(f"evaluation exited {result.returncode}")
        return result.returncode

    out = REPO / "results" / "raw" / f"two_stage_hydrated.{LABEL}.jsonl"
    after = {
        json.loads(x)["question_id"]: json.loads(x)
        for x in out.read_text(encoding="utf-8").splitlines()
        if x.strip()
    }
    before = {
        json.loads(x)["question_id"]: json.loads(x)
        for x in BASELINE.read_text(encoding="utf-8").splitlines()
        if x.strip()
    }

    print("\n" + "=" * 78)
    print(f"{'question':14s} {'production':>11s} {'overlay':>9s}   fact recovered?")
    print("-" * 78)
    flipped = []
    for q in question_ids:
        b = before[q]["correct"]
        a = after[q]["correct"]
        if a and not b:
            flipped.append(q)
        mark = "  ->  FIXED" if a and not b else ("  ->  BROKE" if b and not a else "")
        rec = "no (control)" if q == "73d42213" else "yes"
        print(f"{q:14s} {b!s:>11s} {a!s:>9s}   {rec}{mark}")
    print("-" * 78)
    print(f"{len(flipped)} of {len(question_ids)} flipped to correct: {flipped}")
    print(f"extractor requests for the overlay: {report['extractor_requests']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
