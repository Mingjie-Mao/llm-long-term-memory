"""Zero-call evidence gate for the completed v2b gate16 store."""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

MANIFEST = REPO / "results/manifests/v2b-gate16.json"
OUT_JSON = REPO / "results/analysis/v2b-gate16-evidence.json"
OUT_MD = REPO / "results/analysis/v2b-gate16-evidence.md"
ARMS = {
    "batch15": (REPO / "stores/train150.db", REPO / "stores/train150-index"),
    "batch8": (REPO / "stores/v2b-gate16.db", REPO / "stores/v2b-gate16-index"),
}

_WORD = re.compile(r"[A-Za-z0-9']+")
_STOP = frozenset(
    _WORD.findall(
        "the a an and or but if of to in on at for with from by is are was were be been "
        "being have has had do does did will would can could should may might must this "
        "that these those it its as not no yes user assistant they them their he she his "
        "her you your i me my we our about into over under more most some any"
    )
)


def _words(text: str) -> set[str]:
    return {
        word.lower() for word in _WORD.findall(text) if word.lower() not in _STOP and len(word) > 2
    }


def _external(session_id: str | None) -> str | None:
    from llm_long_term_memory.store import external_session_id

    return external_session_id(session_id) if session_id else None


def analyse() -> dict:
    from llm_long_term_memory.config import ExperimentConfig
    from llm_long_term_memory.embed import Encoder
    from llm_long_term_memory.evaluation.datasets.longmemeval import load
    from llm_long_term_memory.evaluation.extraction_coverage import (
        UNMEASURABLE_TYPES,
        answer_present,
        source_literal_present,
    )
    from llm_long_term_memory.evaluation.manifest import load_manifest
    from llm_long_term_memory.ingest.fidelity import score_sessions
    from llm_long_term_memory.retrieve import HybridRetriever
    from llm_long_term_memory.store import NumpyFlatIndex, SQLiteMemoryStore

    manifest = load_manifest(MANIFEST)
    by_id = {instance.question_id: instance for instance in load(manifest.variant, REPO / "data")}
    instances = [by_id[qid] for qid in manifest.question_ids]
    cfg = ExperimentConfig.from_yaml(REPO / "configs/v2.yaml")
    encoder = Encoder(cfg.models.embedder)
    stores = {}
    indexes = {}
    retrievers = {}
    for arm, (db_path, index_path) in ARMS.items():
        store = SQLiteMemoryStore(db_path, read_only=True)
        store.initialize()
        index = NumpyFlatIndex(index_path, cfg.models.embedding_dim)
        stores[arm] = store
        indexes[arm] = index
        retrievers[arm] = HybridRetriever(
            store,
            index,
            weights=cfg.retrieval.weights.model_dump(),
            candidate_limit=cfg.retrieval.candidate_limit,
            recency_halflife_days=cfg.retrieval.recency_halflife_days,
        )

    try:
        rows = []
        all_memories: dict[str, dict[str, list]] = {arm: {} for arm in ARMS}
        for arm, store in stores.items():
            for instance in instances:
                all_memories[arm][instance.question_id] = store.iter_active(
                    instance.store_namespace
                )

        for instance in instances:
            query = encoder.encode_one(instance.question)
            row = {
                "question_id": instance.question_id,
                "question_type": instance.question_type,
                "is_abstention": instance.is_abstention,
                "measurable_literal": (
                    not instance.is_abstention and instance.question_type not in UNMEASURABLE_TYPES
                ),
                "source_literal_present": (
                    False
                    if instance.is_abstention
                    else source_literal_present(str(instance.answer), instance)
                ),
                "arms": {},
            }
            required = set(instance.answer_session_ids)
            for arm in ARMS:
                memories = all_memories[arm][instance.question_id]
                hits = retrievers[arm].retrieve(
                    query,
                    instance.question,
                    instance.store_namespace,
                    temporal=True,
                    limit=cfg.retrieval.top_k,
                )
                selected = [hit.memory for hit in hits]
                selected_sessions = {
                    sid for memory in selected if (sid := _external(memory.source_session_id))
                }
                raw = stores[arm].search_turns(instance.store_namespace, instance.question, limit=2)
                raw_sessions = {_external(turn.session_id) for turn in raw}
                row["arms"][arm] = {
                    "active_memories": len(memories),
                    "all_store_literal": (
                        False
                        if instance.is_abstention
                        else answer_present(str(instance.answer), memories)
                    ),
                    "top20_literal": (
                        False
                        if instance.is_abstention
                        else answer_present(str(instance.answer), selected)
                    ),
                    "top20_any_source": bool(required & selected_sessions),
                    "top20_all_sources": bool(required) and required <= selected_sessions,
                    "top20_context_tokens": sum(memory.token_count for memory in selected),
                    "raw_top2_any_source": bool(required & raw_sessions),
                    "raw_top2_all_sources": bool(required) and required <= raw_sessions,
                }
            rows.append(row)

        fidelity = {}
        support = {}
        for arm in ARMS:
            pairs = []
            support_values = []
            by_instance_session: dict[str, dict[str, list]] = {}
            for instance in instances:
                grouped = defaultdict(list)
                for memory in all_memories[arm][instance.question_id]:
                    grouped[_external(memory.source_session_id)].append(memory)
                by_instance_session[instance.question_id] = grouped
                for session in instance.sessions:
                    memories = grouped.get(session.session_id, [])
                    pairs.append((session, memories))
                    source_words = _words("\n".join(turn.content for turn in session.turns))
                    for memory in memories:
                        memory_words = _words(f"{memory.content} {memory.object or ''}")
                        if memory_words:
                            support_values.append(
                                len(memory_words & source_words) / len(memory_words)
                            )
            report = score_sessions(pairs)
            fidelity[arm] = {
                "overall": report.overall,
                "memories": report.memories,
                "memories_per_session": report.memories_per_session,
                "per_facet": {
                    facet: {
                        "stated": score.stated,
                        "retained": score.retained,
                        "recall": score.recall,
                    }
                    for facet, score in report.per_facet.items()
                },
            }
            ordered = sorted(support_values)
            support[arm] = {
                "n": len(ordered),
                "median": ordered[len(ordered) // 2] if ordered else 0.0,
                "p10": ordered[max(0, (len(ordered) - 1) // 10)] if ordered else 0.0,
                "below_0_25": sum(value < 0.25 for value in ordered),
                "zero": sum(value == 0 for value in ordered),
            }

        def count(field: str, arm: str, eligible=lambda row: True) -> int:
            return sum(bool(row["arms"][arm][field]) for row in rows if eligible(row))

        def measurable(row: dict) -> bool:
            return bool(row["measurable_literal"])

        def answerable(row: dict) -> bool:
            return not bool(row["is_abstention"])

        summary = {}
        for arm in ARMS:
            summary[arm] = {
                "store_literal_measurable": count("all_store_literal", arm, measurable),
                "top20_literal_measurable": count("top20_literal", arm, measurable),
                "top20_any_source_answerable": count("top20_any_source", arm, answerable),
                "top20_all_sources_answerable": count("top20_all_sources", arm, answerable),
                "raw_top2_any_source_answerable": count("raw_top2_any_source", arm, answerable),
                "raw_top2_all_sources_answerable": count("raw_top2_all_sources", arm, answerable),
                "median_top20_context_tokens": sorted(
                    row["arms"][arm]["top20_context_tokens"] for row in rows
                )[len(rows) // 2],
            }

        changed = []
        fields = (
            "all_store_literal",
            "top20_literal",
            "top20_any_source",
            "top20_all_sources",
            "raw_top2_any_source",
            "raw_top2_all_sources",
        )
        for row in rows:
            differences = {
                field: [row["arms"]["batch15"][field], row["arms"]["batch8"][field]]
                for field in fields
                if row["arms"]["batch15"][field] != row["arms"]["batch8"][field]
            }
            if differences:
                changed.append({"question_id": row["question_id"], "changes": differences})

        return {
            "calls": 0,
            "questions": len(rows),
            "answerable_questions": sum(not row["is_abstention"] for row in rows),
            "measurable_literal_questions": sum(row["measurable_literal"] for row in rows),
            "fidelity": fidelity,
            "lexical_support_floor": support,
            "summary": summary,
            "changed_questions": changed,
            "rows": rows,
        }
    finally:
        for store in stores.values():
            store.close()


def _pct(value: float) -> str:
    return f"{value:.1%}"


def render(result: dict) -> str:
    old = result["fidelity"]["batch15"]
    new = result["fidelity"]["batch8"]
    lines = [
        "# v2b gate16 — zero-call evidence gate",
        "",
        "**No model/API calls were made.** Both stores are compared on the same frozen "
        "16 questions.",
        "",
        "## Extraction fidelity on all 780 gate sessions",
        "",
        f"Batch15: **{_pct(old['overall'])}**, {old['memories']:,} source-linked active "
        "memories.  ",
        f"Batch8: **{_pct(new['overall'])}**, {new['memories']:,} source-linked active memories.  ",
        f"Change: **+{(new['overall'] - old['overall']) * 100:.1f} points**.",
        "",
        "| facet | batch15 | batch8 | retained change |",
        "|---|---:|---:|---:|",
    ]
    for facet, old_facet in old["per_facet"].items():
        new_facet = new["per_facet"][facet]
        old_rate = _pct(old_facet["recall"]) if old_facet["stated"] else "n/a"
        new_rate = _pct(new_facet["recall"]) if new_facet["stated"] else "n/a"
        lines.append(
            f"| {facet} | {old_rate} | {new_rate} | "
            f"{new_facet['retained'] - old_facet['retained']:+d} |"
        )
    lines.extend(
        [
            "",
            "## Evidence endpoints",
            "",
            "| endpoint | batch15 | batch8 |",
            "|---|---:|---:|",
        ]
    )
    labels = {
        "store_literal_measurable": "whole-store literal (measurable)",
        "top20_literal_measurable": "top-20 literal (measurable)",
        "top20_any_source_answerable": "top-20 any gold session",
        "top20_all_sources_answerable": "top-20 all gold sessions",
        "raw_top2_any_source_answerable": "raw top-2 any gold session",
        "raw_top2_all_sources_answerable": "raw top-2 all gold sessions",
        "median_top20_context_tokens": "median top-20 memory tokens",
    }
    for field, label in labels.items():
        lines.append(
            f"| {label} | {result['summary']['batch15'][field]} | "
            f"{result['summary']['batch8'][field]} |"
        )
    old_support = result["lexical_support_floor"]["batch15"]
    new_support = result["lexical_support_floor"]["batch8"]
    lines.extend(
        [
            "",
            "## Support floor",
            "",
            f"Batch15 median/p10 lexical support: {_pct(old_support['median'])} / "
            f"{_pct(old_support['p10'])}; below 25%: {old_support['below_0_25']}.",
            f"Batch8 median/p10 lexical support: {_pct(new_support['median'])} / "
            f"{_pct(new_support['p10'])}; below 25%: {new_support['below_0_25']}.",
            "",
            "This remains a low-support screen, not a hallucination rate.",
            "",
            "## Changed questions",
            "",
        ]
    )
    if result["changed_questions"]:
        for row in result["changed_questions"]:
            changes = ", ".join(
                f"{field}: {values[0]}→{values[1]}" for field, values in row["changes"].items()
            )
            lines.append(f"- `{row['question_id']}` — {changes}")
    else:
        lines.append("No binary evidence endpoint changed.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    result = analyse()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    OUT_MD.write_text(render(result))
    print(render(result))
    print(f"wrote {OUT_JSON.relative_to(REPO)}")
    print(f"wrote {OUT_MD.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
