"""Project v3.3 context cost locally without making any provider call.

The tune42 questions are development data and may be read. The script replays the
unchanged retrieval path against train150, verifies that it reconstructs v3.2's
recorded hydration, swaps only the evidence packer, and carries forward the already
observed fallback size. It estimates context cost, not answer accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from llm_long_term_memory.config import ExperimentConfig, Settings  # noqa: E402
from llm_long_term_memory.embed import Encoder  # noqa: E402
from llm_long_term_memory.evaluation.datasets import longmemeval as lme  # noqa: E402
from llm_long_term_memory.evaluation.manifest import load_manifest  # noqa: E402
from llm_long_term_memory.evaluation.runners.memory import (  # noqa: E402
    TEMPORAL_NOTE,
    MemoryRunner,
    render_grouped,
)
from llm_long_term_memory.evaluation.runners.reasoning import reasoning_kind  # noqa: E402
from llm_long_term_memory.retrieve import EvidenceHydrator, render_evidence  # noqa: E402
from llm_long_term_memory.store import (  # noqa: E402
    NumpyFlatIndex,
    SQLiteMemoryStore,
    external_session_id,
)

OLD_CONFIG = Path("configs/v3-phase4-adaptive.yaml")
NEW_CONFIG = Path("configs/v3-phase5-compact.yaml")
TRAIN_MANIFEST = Path("results/manifests/train150.json")
TUNE_MANIFEST = Path("results/manifests/v3-reasoning-tune42.json")
OBSERVED_ROWS = Path("results/sealed/v3-phase4-tune2/v3.2-adaptive.jsonl")
TUNE2_AGGREGATE = Path("results/validation/v3-phase4-tune2.json")
OUTPUT = Path("results/analysis/v3-phase5-context-projection.json")
TARGET_OPERATIONS = {"temporal", "multi_session_aggregation", "current_state"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_rows(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 42 or len({row["question_id"] for row in rows}) != 42:
        raise ValueError("v3.2 observed artifact must contain 42 unique rows")
    return {row["question_id"]: row for row in rows}


def _render(memories, hydrator: EvidenceHydrator, max_tokens: int, hydrate: bool):
    body = render_grouped(memories, temporal=True)
    context = f"{TEMPORAL_NOTE}\n\n{body}"
    result = hydrator.hydrate(memories, max_tokens=max_tokens) if hydrate else None
    if result and result.evidence:
        context = (
            f"Structured memories:\n{context}\n\n"
            f"Verbatim source evidence:\n{render_evidence(result.evidence)}"
        )
    return context, result


def _percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def project() -> dict:
    settings = Settings()
    old_config = ExperimentConfig.from_yaml(str(REPO / OLD_CONFIG))
    new_config = ExperimentConfig.from_yaml(str(REPO / NEW_CONFIG))
    if old_config.retrieval != new_config.retrieval:
        raise ValueError("projection requires identical retrieval configuration")
    if old_config.temporal_resolution != new_config.temporal_resolution:
        raise ValueError("projection requires identical temporal rendering")

    tune = load_manifest(REPO / TUNE_MANIFEST)
    train = load_manifest(REPO / TRAIN_MANIFEST)
    if tune.name != "v3-reasoning-tune42" or len(tune) != 42:
        raise ValueError("projection is restricted to the inspectable tune42 manifest")
    instances = {
        instance.question_id: instance
        for instance in lme.load(train.variant, REPO / settings.data_dir)
        if instance.question_id in set(tune.question_ids)
    }
    if set(instances) != set(tune.question_ids):
        raise ValueError("tune42 does not match the train150 dataset")
    observed = _load_rows(REPO / OBSERVED_ROWS)

    store = SQLiteMemoryStore(REPO / settings.store_dir / "train150.db")
    index = NumpyFlatIndex(
        REPO / settings.store_dir / "train150-index", dim=new_config.models.embedding_dim
    )
    runner = MemoryRunner(
        None,  # Retrieval and context assembly below never call the provider.
        model=new_config.models.answerer,
        encoder=Encoder(new_config.models.embedder),
        store=store,
        index=index,
        temporal=True,
        top_k=new_config.retrieval.top_k,
        retrieval_weights=new_config.retrieval.weights.model_dump(),
        candidate_limit=new_config.retrieval.candidate_limit,
        recency_halflife_days=new_config.retrieval.recency_halflife_days,
    )
    old_hydrator = EvidenceHydrator(
        store,
        neighbouring_sentences=old_config.hydration.neighbouring_sentences,
        allocation=old_config.hydration.allocation,
    )
    new_hydrator = EvidenceHydrator(
        store,
        neighbouring_sentences=new_config.hydration.neighbouring_sentences,
        allocation=new_config.hydration.allocation,
    )

    projected_context: list[int] = []
    hydration_tokens: list[int] = []
    hydrated_questions = 0
    eligible_sessions = 0
    hydrated_sessions = 0
    redundant_anchors = 0
    old_gold_coverage: list[float] = []
    new_gold_coverage: list[float] = []
    try:
        for question_id in tune.question_ids:
            instance = instances[question_id]
            memories = [hit.memory for hit in runner.retrieve(instance)]
            hydrate = reasoning_kind(instance.question) in TARGET_OPERATIONS
            old_context, old_hydration = _render(
                memories, old_hydrator, old_config.hydration.max_tokens, hydrate
            )
            new_context, new_hydration = _render(
                memories, new_hydrator, new_config.hydration.max_tokens, hydrate
            )
            row = observed[question_id]
            old_tokens = int(len(old_context) / runner.chars_per_token)
            if old_hydration and old_hydration.tokens != row["notes"]["hydrated_tokens"]:
                raise ValueError(f"v3.2 hydration replay changed for {question_id}")
            fallback_tokens = max(0, row["context_tokens"] - old_tokens)
            projected_context.append(
                int(len(new_context) / runner.chars_per_token) + fallback_tokens
            )
            if new_hydration and new_hydration.evidence:
                hydrated_questions += 1
                hydration_tokens.append(new_hydration.tokens)
                eligible_sessions += new_hydration.eligible_sessions
                hydrated_sessions += new_hydration.hydrated_sessions
                redundant_anchors += new_hydration.redundant_anchors
                old_coverage = row["notes"]["recall_coverage"]["hydrated"]
                if old_coverage is not None:
                    old_gold_coverage.append(float(old_coverage))
                gold_sessions = set(instance.answer_session_ids)
                hydrated_gold_sessions = {
                    external_session_id(item.session_id) for item in new_hydration.evidence
                }
                if gold_sessions:
                    new_gold_coverage.append(
                        len(gold_sessions & hydrated_gold_sessions) / len(gold_sessions)
                    )
    finally:
        store.close()

    aggregate = json.loads((REPO / TUNE2_AGGREGATE).read_text(encoding="utf-8"))
    v2_median = aggregate["arms"][0]["median_context_tokens"]
    observed_v32_median = aggregate["arms"][1]["median_context_tokens"]
    projected_median = median(projected_context)
    return {
        "schema_version": 1,
        "kind": "offline_context_projection",
        "provider_calls": 0,
        "questions": len(projected_context),
        "source": {
            "observed_rows": OBSERVED_ROWS.as_posix(),
            "observed_rows_sha256": _sha256(REPO / OBSERVED_ROWS),
            "old_config_sha256": _sha256(REPO / OLD_CONFIG),
            "new_config_sha256": _sha256(REPO / NEW_CONFIG),
        },
        "assumption": (
            "Retrieval is replayed exactly; the observed raw-fallback token increment is "
            "carried forward. Answer accuracy and future fallback decisions are not predicted."
        ),
        "v2_median_context_tokens": v2_median,
        "v3_2_observed": {
            "median_context_tokens": observed_v32_median,
            "mean_hydrated_gold_session_coverage": (
                sum(old_gold_coverage) / len(old_gold_coverage)
            ),
            "full_hydrated_gold_session_coverage_rate": (
                sum(value == 1.0 for value in old_gold_coverage) / len(old_gold_coverage)
            ),
        },
        "v3_3_projected": {
            "median_context_tokens": projected_median,
            "context_ratio_vs_v2": projected_median / v2_median,
            "p95_context_tokens": _percentile(projected_context, 0.95),
            "max_context_tokens": max(projected_context),
            "hydrated_questions": hydrated_questions,
            "median_hydration_tokens": median(hydration_tokens),
            "eligible_sessions": eligible_sessions,
            "hydrated_sessions": hydrated_sessions,
            "session_coverage": hydrated_sessions / eligible_sessions,
            "mean_hydrated_gold_session_coverage": (
                sum(new_gold_coverage) / len(new_gold_coverage)
            ),
            "full_hydrated_gold_session_coverage_rate": (
                sum(value == 1.0 for value in new_gold_coverage) / len(new_gold_coverage)
            ),
            "deduplicated_source_anchors": redundant_anchors,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    payload = project()
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.write:
        destination = REPO / OUTPUT
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
        print(f"wrote {destination}")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
