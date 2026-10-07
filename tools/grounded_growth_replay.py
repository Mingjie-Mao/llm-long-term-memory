"""Replay actual v15 context at 1/2/4x exposed train history; zero provider calls."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]

from analysis_io import sha256_file  # noqa: E402
from grounded_context_replay import StubClient  # noqa: E402
from history_growth_retention import _copy_store, _grow  # noqa: E402


def measure(version=15):
    from llm_long_term_memory import cli
    from llm_long_term_memory.api.service import MemoryService
    from llm_long_term_memory.conversation import AnswerRequest
    from llm_long_term_memory.evaluation.datasets import longmemeval as lme
    from llm_long_term_memory.evaluation.manifest import load_manifest
    from llm_long_term_memory.store import NumpyFlatIndex, external_session_id

    manifest_path = REPO / "results/manifests/train150.json"
    manifest = load_manifest(manifest_path)
    cfg, settings, runner, _, _ = cli._build(
        f"two_stage_raw_primary_grounded_v{version}",
        "configs/fallback.yaml",
        "train150",
        client_override=StubClient(),
        read_only_store=True,
    )
    instances = {i.question_id: i for i in lme.load(manifest.variant, settings.data_dir)}
    source_index = runner.raw_primary_turn_index
    positions = {tid: i for i, tid in enumerate(source_index.ids)}
    original_store, original_memory_index = runner.store, runner.index
    result = {
        "candidate_version": version,
        "source_sha256": sha256_file(
            REPO / "src/llm_long_term_memory/runtime/grounded_answering.py"
        ),
        "experiment_class": "development-offline",
        "provider_calls": 0,
        "manifest_sha256": sha256_file(manifest_path),
        "models": cfg.models.model_dump(),
        "limitations": "Synthetic donor history, no contradictions or answer scoring.",
        "by_factor": {},
    }
    try:
        for factor in (1, 2, 4):
            started = time.perf_counter()
            with tempfile.TemporaryDirectory() as scratch:
                directory = Path(scratch)
                _copy_store("train150", directory)
                svc = MemoryService(
                    store_name="growth",
                    config_path="configs/fallback.yaml",
                    settings=settings.model_copy(update={"store_dir": directory}),
                )
                try:
                    _grow(svc.store, svc.index, list(manifest.question_ids), factor)
                    runner.store = runner.retriever.store = runner.grounded.store = svc.store
                    runner.index = runner.retriever.index = svc.index
                    rows = []
                    for qid in manifest.question_ids:
                        instance = instances[qid]
                        turns = [
                            t
                            for sid in sorted(svc.store.session_ids_for_user(qid))
                            for t in svc.store.turns_for_session(sid)
                        ]
                        idx = NumpyFlatIndex(directory / "unsaved-turns", dim=source_index.dim)
                        idx.add(
                            [t.id for t in turns],
                            np.vstack(
                                [
                                    source_index._vectors[positions[t.id.split("~")[0]]]
                                    for t in turns
                                ]
                            ),
                        )
                        runner.grounded.turn_index = idx
                        answer = runner.answer_request(
                            AnswerRequest(instance.question, instance.question_date, qid)
                        )
                        gold = {
                            (s.session_id, i)
                            for s in instance.sessions
                            for i, t in enumerate(s.turns)
                            if t.has_answer
                        }
                        if not gold:
                            continue
                        sources = answer.notes["grounded_calls"][0]["evidence"]["sources"]
                        selected = {
                            (external_session_id(s["session_id"]), s["turn_index"])
                            for s in sources
                            if s["kind"] == "raw" and "~" not in s["session_id"]
                        }
                        mismatch = any(
                            t.content != instance_turn.content
                            for s in instance.sessions
                            for i, instance_turn in enumerate(s.turns)
                            if (s.session_id, i) in gold
                            for t in turns
                            if "~" not in t.session_id
                            and external_session_id(t.session_id) == s.session_id
                            and t.turn_index == i
                        )
                        rows.append(
                            {
                                "question_id": qid,
                                "covered": gold <= selected and not mismatch,
                                "missing": sorted(gold - selected),
                                "context": answer.context_tokens,
                                "first_loss": None if gold <= selected else "S4b_rank_budget",
                                "content_mismatch": mismatch,
                            }
                        )
                    result["by_factor"][str(factor)] = {
                        "scored": len(rows),
                        "covered": sum(r["covered"] for r in rows),
                        "coverage": sum(r["covered"] for r in rows) / len(rows),
                        "median_context": float(np.median([r["context"] for r in rows])),
                        "max_context": max(r["context"] for r in rows),
                        "rows": rows,
                        "wall_seconds": time.perf_counter() - started,
                    }
                    print(
                        f"{factor}x: {result['by_factor'][str(factor)]['covered']}/{len(rows)}",
                        flush=True,
                    )
                finally:
                    svc.close()
        return result
    finally:
        runner.store = runner.retriever.store = runner.grounded.store = original_store
        runner.index = runner.retriever.index = original_memory_index
        runner.grounded.turn_index = source_index
        original_store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=15)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    output = REPO / f"results/analysis/grounded-growth-v{args.version}{args.label}.train150.json"
    if output.exists():
        raise SystemExit("Refusing to overwrite prior evidence")
    result = measure(args.version)
    result["generator_sha256"] = sha256_file(Path(__file__))
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
