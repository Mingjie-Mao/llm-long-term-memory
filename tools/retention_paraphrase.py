"""Replay delayed paraphrased queries against a durable, already built store.

Regression protocol: results/prereg-retention-paraphrase-v1.md. No LLM calls.
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import read_json, write_report  # noqa: E402


def analyse() -> dict:
    from llm_long_term_memory.config import ExperimentConfig
    from llm_long_term_memory.embed import Encoder
    from llm_long_term_memory.evaluation.datasets import longmemeval as lme
    from llm_long_term_memory.ingest.extract import _parse_date
    from llm_long_term_memory.retrieve import HybridRetriever
    from llm_long_term_memory.store import (
        NumpyFlatIndex,
        SQLiteMemoryStore,
        external_session_id,
    )

    manifest = read_json(REPO / "results/manifests/retention-paraphrase-v1.json")
    source_manifest = read_json(REPO / manifest["source_manifest"])
    items = manifest["items"]
    ids = [item["question_id"] for item in items]
    if len(ids) != len(set(ids)) or set(ids) != set(source_manifest["question_ids"]):
        raise ValueError("retention items must cover each source-manifest question exactly once")
    instances = {item.question_id: item for item in lme.load("s", REPO / "data")}
    cfg = ExperimentConfig.from_yaml(REPO / "configs/v2b-batch8-repair.yaml")
    store = SQLiteMemoryStore(REPO / "stores/v2b-gate16-repair.db", read_only=True)
    store.initialize()
    try:
        index = NumpyFlatIndex(
            REPO / "stores/v2b-gate16-repair-index", dim=cfg.models.embedding_dim
        )
        index.validate_ids(store.memory_ids())
        encoder = Encoder(cfg.models.embedder)
        texts = [
            text
            for item in items
            for text in (instances[item["question_id"]].question, item["paraphrase"])
        ]
        vectors = encoder.encode(texts)
        retriever = HybridRetriever(
            store,
            index,
            weights=cfg.retrieval.weights.model_dump(),
            candidate_limit=cfg.retrieval.candidate_limit,
            recency_halflife_days=cfg.retrieval.recency_halflife_days,
        )
        rows = []
        for item_index, item in enumerate(items):
            instance = instances[item["question_id"]]
            namespace = instance.store_namespace
            gold = set(instance.answer_session_ids)
            if not gold:
                raise ValueError(f"{item['question_id']} has no gold source sessions")
            durable = {
                external_session_id(memory.source_session_id)
                for memory in store.iter_all(namespace)
                if memory.source_session_id is not None
            }
            asked = _parse_date(instance.question_date)
            if asked is None:
                raise ValueError(f"{item['question_id']} has an unparseable question date")
            for days in (1, 7, 30):
                for variant_index, (variant, text) in enumerate(
                    (("original", instance.question), ("paraphrase", item["paraphrase"]))
                ):
                    vector = vectors[2 * item_index + variant_index]
                    hits, trace = retriever.retrieve_with_trace(
                        vector,
                        text,
                        namespace,
                        temporal=True,
                        limit=20,
                        as_of=asked + timedelta(days=days),
                    )
                    candidate_sessions = {
                        external_session_id(sid) for sid in trace.candidate_session_ids
                    }
                    selected_sessions = {
                        external_session_id(hit.memory.source_session_id)
                        for hit in hits
                        if hit.memory.source_session_id
                    }
                    rows.append(
                        {
                            "question_id": item["question_id"],
                            "variant": variant,
                            "delay_days": days,
                            "gold_source_session_ids": sorted(gold),
                            "durable_gold_sources": sorted(gold & durable),
                            "candidate_gold_sources": sorted(gold & candidate_sessions),
                            "top20_gold_sources": sorted(gold & selected_sessions),
                            "durable_all": gold <= durable,
                            "candidate_any": bool(gold & candidate_sessions),
                            "candidate_all": gold <= candidate_sessions,
                            "top20_any": bool(gold & selected_sessions),
                            "top20_all": gold <= selected_sessions,
                            "cross_tenant_hits": sum(
                                hit.memory.user_id != namespace for hit in hits
                            ),
                            "retrieved_ids": [hit.memory.id for hit in hits],
                        }
                    )
        summary = {}
        for days in (1, 7, 30):
            for variant in ("original", "paraphrase"):
                subset = [
                    row for row in rows if row["delay_days"] == days and row["variant"] == variant
                ]
                summary[f"{variant}_day{days}"] = {
                    metric: sum(bool(row[metric]) for row in subset)
                    for metric in (
                        "durable_all",
                        "candidate_any",
                        "candidate_all",
                        "top20_any",
                        "top20_all",
                    )
                }
        return {
            "experiment": "retention-paraphrase-v1",
            "class": "regression",
            "source_manifest": manifest["source_manifest"],
            "model_calls": 0,
            "local_encoder": cfg.models.embedder,
            "questions": len(items),
            "recency_weight": cfg.retrieval.weights.recency,
            "cross_tenant_hits": sum(row["cross_tenant_hits"] for row in rows),
            "summary": summary,
            "rows": rows,
            "limit": (
                "Delayed dates are simulated ranking references on a reopened store, "
                "not elapsed wall-clock time; source-session recall is not answer accuracy. "
                "The configured recency weight is zero, so changing only the delay "
                "cannot change this ranking."
            ),
        }
    finally:
        store.close()


def render(result: dict) -> str:
    lines = [
        "# Delayed paraphrase retention v1",
        "",
        "> Regression on 16 inspected development questions. No LLM calls or answer scores.",
        "",
        "| query | delay | durable all | candidate any/all | top-20 any/all |",
        "|---|---:|---:|---:|---:|",
    ]
    for days in (1, 7, 30):
        for variant in ("original", "paraphrase"):
            row = result["summary"][f"{variant}_day{days}"]
            n = result["questions"]
            lines.append(
                f"| {variant} | {days} days | {row['durable_all']}/{n} | "
                f"{row['candidate_any']}/{n} / {row['candidate_all']}/{n} | "
                f"{row['top20_any']}/{n} / {row['top20_all']}/{n} |"
            )
    lines += [
        "",
        f"Cross-tenant retrieved hits: **{result['cross_tenant_hits']}**.",
        "",
        result["limit"],
        "",
        "## Per-item top-20 source coverage at day 1",
        "",
        "| question | original | paraphrase | gold source sessions |",
        "|---|---:|---:|---:|",
    ]
    for qid in sorted({row["question_id"] for row in result["rows"]}):
        pair = [
            row for row in result["rows"] if row["question_id"] == qid and row["delay_days"] == 1
        ]
        original, paraphrase = pair
        lines.append(
            f"| `{qid}` | {len(original['top20_gold_sources'])} | "
            f"{len(paraphrase['top20_gold_sources'])} | "
            f"{len(original['gold_source_session_ids'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-out", type=Path, default=REPO / "results/analysis/retention-paraphrase-v1.json"
    )
    parser.add_argument(
        "--md-out", type=Path, default=REPO / "results/analysis/retention-paraphrase-v1.md"
    )
    args = parser.parse_args()
    result = analyse()
    print(write_report(result, render, json_out=args.json_out, md_out=args.md_out), end="")
    return 0 if result["cross_tenant_hits"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
