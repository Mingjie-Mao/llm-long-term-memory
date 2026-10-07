"""Zero-call archive-page coverage and projected cost on exposed train150 growth."""

from __future__ import annotations

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
from llm_long_term_memory.store import (  # noqa: E402
    NumpyFlatIndex,
    SQLiteMemoryStore,
    external_session_id,
)


def measure():
    settings = Settings()
    manifest_path = REPO / "results/manifests/train150.json"
    manifest = load_manifest(manifest_path)
    instances = {i.question_id: i for i in load("s", settings.data_dir)}
    result = {
        "experiment_class": "development-offline-page-coverage",
        "provider_calls": 0,
        "manifest_sha256": sha256_file(manifest_path),
        "by_factor": {},
        "metric": (
            "Gold source delivery across all review pages; "
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
                    pages = archive_pages(store, qid)
                    original = {
                        (s.session_id, i): t.content
                        for s in instances[qid].sessions
                        for i, t in enumerate(s.turns)
                        if t.has_answer
                    }
                    sources = [s for page in pages.pages for s in page.sources]
                    delivered = {
                        (external_session_id(s.session_id), s.turn_index): s.text
                        for s in sources
                        if "~" not in s.session_id
                    }
                    covered = bool(original) and all(
                        delivered.get(k) == text for k, text in original.items()
                    )
                    rows.append(
                        {
                            "question_id": qid,
                            "has_gold": bool(original),
                            "covered": covered,
                            "complete": pages.complete,
                            "pages": len(pages.pages),
                            "page_tokens": sum(int(len(p.render()) / 4.6) for p in pages.pages),
                            "max_page_tokens": max(
                                (int(len(p.render()) / 4.6) for p in pages.pages), default=0
                            ),
                            "omitted": len(pages.omitted_turn_ids),
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
    result["sources"] = {
        p: sha256_file(REPO / p)
        for p in (
            "src/llm_long_term_memory/runtime/evidence_pages.py",
            "src/llm_long_term_memory/runtime/grounded_answering.py",
            "tools/paged_growth_replay.py",
        )
    }
    return result


if __name__ == "__main__":
    output = REPO / "results/analysis/paged-growth-v1-r2.train150.json"
    if output.exists():
        raise SystemExit("Refusing to overwrite historical evidence")
    output.write_text(json.dumps(measure(), indent=2) + "\n", encoding="utf-8")
