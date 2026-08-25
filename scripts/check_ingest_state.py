"""Verify that an ingestion store is complete or safe to resume, without API calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from llm_long_term_memory.config import ExperimentConfig, Settings  # noqa: E402
from llm_long_term_memory.evaluation.datasets import longmemeval as lme  # noqa: E402
from llm_long_term_memory.evaluation.ingest_state import inspect_ingest_state  # noqa: E402
from llm_long_term_memory.evaluation.manifest import load_manifest  # noqa: E402
from llm_long_term_memory.ingest import fingerprint, namespaced_sessions  # noqa: E402
from llm_long_term_memory.ingest.pipeline import resolved_sessions_per_request  # noqa: E402
from llm_long_term_memory.llm import Limits, QuotaManager  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/fallback.yaml"))
    parser.add_argument("--manifest", type=Path, default=Path("results/manifests/train150.json"))
    parser.add_argument("--store-name", default="train150")
    args = parser.parse_args()

    settings = Settings()
    cfg = ExperimentConfig.from_yaml(args.config)
    manifest = load_manifest(args.manifest)
    by_id = {item.question_id: item for item in lme.load(manifest.variant, settings.data_dir)}
    missing = set(manifest.question_ids) - set(by_id)
    if missing:
        print(json.dumps({"safe_to_resume": False, "issues": ["manifest ids missing from data"]}))
        return 1
    pairs = namespaced_sessions([by_id[qid] for qid in manifest.question_ids])

    quota = QuotaManager(
        state_dir=settings.store_dir / "quota",
        default=Limits(rpm=cfg.quota.rpm, tpm=cfg.quota.tpm, rpd=cfg.quota.rpd),
    )
    quota.load_learned()
    batch_size = resolved_sessions_per_request(
        cfg.ingest.sessions_per_request,
        quota.for_model(cfg.models.extractor).limits.tpm,
    )
    expected = fingerprint.from_config(cfg, sessions_per_request=batch_size).as_dict()
    try:
        state = inspect_ingest_state(
            store_dir=settings.store_dir,
            store_name=args.store_name,
            pairs=pairs,
            expected_fingerprint=expected,
            batch_size=batch_size,
            embedding_dim=cfg.models.embedding_dim,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"safe_to_resume": False, "issues": [f"{type(exc).__name__}: {exc}"]}))
        return 1
    print(json.dumps(state.to_dict(), indent=2))
    if not state.safe_to_resume:
        print("STOP: store is not safe to resume", file=sys.stderr)
        return 1
    print("COMPLETE" if state.complete else "SAFE TO RESUME")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
