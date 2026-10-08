"""Zero-call archive-page coverage and projected cost on exposed train150 growth."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]

from analysis_io import sha256_file  # noqa: E402
from history_growth_retention import _copy_store, _grow  # noqa: E402

from llm_long_term_memory.config import Settings  # noqa: E402
from llm_long_term_memory.evaluation.datasets.longmemeval import load  # noqa: E402
from llm_long_term_memory.evaluation.manifest import load_manifest  # noqa: E402
from llm_long_term_memory.runtime.evidence_pages import archive_pages  # noqa: E402
from llm_long_term_memory.runtime.grounded_answering import personal_archive_review  # noqa: E402
from llm_long_term_memory.store import (  # noqa: E402
    NumpyFlatIndex,
    SQLiteMemoryStore,
    external_session_id,
)


def measure():
    source_paths = (
        "src/llm_long_term_memory/runtime/evidence_pages.py",
        "src/llm_long_term_memory/runtime/grounded_answering.py",
        "tools/role_paged_growth_replay.py",
    )
    source_hashes = {p: sha256_file(REPO / p) for p in source_paths}
    settings = Settings()
    manifest_path = REPO / "results/manifests/train150.json"
    manifest = load_manifest(manifest_path)
    instances = {i.question_id: i for i in load("s", settings.data_dir)}
    baseline_path = REPO / "results/analysis/grounded-growth-v15.train150.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    result = {
        "baseline_sha256": sha256_file(baseline_path),
        "experiment_class": "development-offline-composite-role-routing",
        "provider_calls": 0,
        "manifest_sha256": sha256_file(manifest_path),
        "by_factor": {},
        "metric": (
            "USER pages plus archived v15 advice coverage; "
            "NOT final answer accuracy or single-context coverage."
        ),
        "limits": (
            "Synthetic donors, 32 pages, 6000 estimated tokens per page. Selection not scored."
        ),
    }
    for factor in (1, 2, 4):
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            _copy_store("train150", directory)
            store = SQLiteMemoryStore(directory / "growth.db")
            store.initialize()
            index = NumpyFlatIndex(directory / "growth-index", 384)
            try:
                _grow(store, index, list(manifest.question_ids), factor)
                rows = []
                for qid in manifest.question_ids:
                    routed = personal_archive_review(instances[qid].question)
                    pages = (
                        archive_pages(store, qid, roles=("user",), split_oversized=True)
                        if routed
                        else None
                    )
                    old = next(
                        (
                            r
                            for r in baseline["by_factor"][str(factor)]["rows"]
                            if r["question_id"] == qid
                        ),
                        {"covered": False},
                    )
                    original = {
                        (s.session_id, i): t.content
                        for s in instances[qid].sessions
                        for i, t in enumerate(s.turns)
                        if t.has_answer
                    }
                    sources = [s for page in pages.pages for s in page.sources] if pages else []
                    pieces = {}
                    for source in sources:
                        if "~" in source.session_id:
                            continue
                        key = (external_session_id(source.session_id), source.turn_index)
                        pieces.setdefault(key, []).append(
                            (getattr(source, "char_start", 0), source.text)
                        )
                    delivered = {k: "".join(t for _, t in sorted(v)) for k, v in pieces.items()}
                    covered = bool(original) and all(
                        delivered.get(k) == text for k, text in original.items()
                    )
                    if not routed:
                        covered = old["covered"]
                    rows.append(
                        {
                            "question_id": qid,
                            "has_gold": bool(original),
                            "covered": covered,
                            "routed": routed,
                            "complete": pages.complete if pages else True,
                            "pages": len(pages.pages) if pages else 0,
                            "page_tokens": sum(
                                int(len(p.render()) / 4.6) for p in (pages.pages if pages else [])
                            ),
                            "max_page_tokens": max(
                                (
                                    int(len(p.render()) / 4.6)
                                    for p in (pages.pages if pages else [])
                                ),
                                default=0,
                            ),
                            "omitted": len(pages.omitted_turn_ids) if pages else 0,
                        }
                    )
                scored = [r for r in rows if r["has_gold"]]
                result["by_factor"][str(factor)] = {
                    "scored": len(scored),
                    "covered": sum(r["covered"] for r in scored),
                    "complete_archives": sum(r["complete"] for r in rows),
                    "median_pages": statistics.median(r["pages"] for r in rows),
                    "max_pages": max(r["pages"] for r in rows),
                    "projected_selection_calls": sum(r["pages"] for r in rows if r["complete"]),
                    "median_review_tokens": statistics.median(r["page_tokens"] for r in rows),
                    "rows": rows,
                }
                print(
                    f"{factor}x: {result['by_factor'][str(factor)]['covered']}/{len(scored)}, "
                    f"median {result['by_factor'][str(factor)]['median_pages']} pages",
                    flush=True,
                )
            finally:
                store.close()
    if source_hashes != {p: sha256_file(REPO / p) for p in source_paths}:
        raise ValueError("implementation changed during offline run")
    result["sources"] = source_hashes
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    output = REPO / f"results/analysis/{args.label}.train150.json"
    if output.exists():
        raise SystemExit("Refusing to overwrite historical evidence")
    output.write_text(json.dumps(measure(), indent=2) + "\n", encoding="utf-8")
