"""Zero-call attribution of the three existing batch-8 fidelity readings.

The cohorts differ, so this reports observable composition and accounting rather
than attributing the residual to a causal mechanism. Registration:
results/prereg-batch8-cohort-attribution-v1.md.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import read_json, write_report  # noqa: E402


def _summary(pairs: list[tuple], *, source: str) -> dict:
    from llm_long_term_memory.ingest.fidelity import (
        _extract_facets,
        score_sessions,
        user_assertions,
    )

    report = score_sessions(pairs)
    source_lengths = [session.char_count for session, _ in pairs]
    assertion_lengths = [len(user_assertions(session)) for session, _ in pairs]
    specifics = Counter()
    for session, _ in pairs:
        facets = _extract_facets(user_assertions(session))
        specifics.update({facet: len(values) for facet, values in facets.items()})
    return {
        "source": source,
        "sessions": len(pairs),
        "session_ids": sorted({session.session_id for session, _ in pairs}),
        "memories": report.memories,
        "memories_per_session": report.memories_per_session,
        "specifics": sum(specifics.values()),
        "specifics_kept": sum(item.retained for item in report.per_facet.values()),
        "recall": report.overall,
        "facets": {
            facet: {"stated": item.stated, "kept": item.retained, "recall": item.recall}
            for facet, item in report.per_facet.items()
        },
        "median_source_chars": statistics.median(source_lengths),
        "median_user_assertion_chars": statistics.median(assertion_lengths),
        "total_source_chars": sum(source_lengths),
        "total_user_assertion_chars": sum(assertion_lengths),
        "sessions_without_specifics": sum(
            not any(_extract_facets(user_assertions(session)).values()) for session, _ in pairs
        ),
    }


def reweight(source: dict, target: dict) -> dict:
    """Apply source facet recalls to a target's facet counts, without extrapolation."""
    shared = [
        facet
        for facet in source["facets"]
        if source["facets"][facet]["stated"] and target["facets"][facet]["stated"]
    ]
    covered = sum(target["facets"][facet]["stated"] for facet in shared)
    expected = sum(
        target["facets"][facet]["stated"] * source["facets"][facet]["recall"] for facet in shared
    )
    return {
        "recall_on_shared_facets": expected / covered if covered else None,
        "target_specifics_covered": covered,
        "target_specifics_total": target["specifics"],
        "facets_without_source_support": sorted(
            facet
            for facet in target["facets"]
            if target["facets"][facet]["stated"] and not source["facets"][facet]["stated"]
        ),
    }


def analyse() -> dict:
    from llm_long_term_memory.evaluation.datasets import longmemeval as lme
    from llm_long_term_memory.ingest.pipeline import namespaced_sessions
    from llm_long_term_memory.store import Memory, SQLiteMemoryStore, scoped_session_id

    archive = read_json(REPO / "results/raw/extraction-batch-size.extractions.json")
    pilot = read_json(REPO / "results/raw/specificity-repair-pilot.checkpoint.json")
    manifest = read_json(REPO / "results/manifests/v2b-gate16.json")
    recorded_a = read_json(REPO / "results/analysis/extraction-batch-size.json")
    recorded_b = read_json(REPO / "results/analysis/specificity-repair-pilot.json")
    recorded_c = read_json(REPO / "results/analysis/store-fidelity.v2b-gate16.json")

    instances = lme.load("s", REPO / "data")
    _, heldout = lme.split_dev_test(instances)
    heldout_sessions = [session for instance in heldout for session in instance.sessions]
    a_sessions = heldout_sessions[:60]
    b_sessions = heldout_sessions[60:120]
    if [s.session_id for s in a_sessions] != archive["session_ids"]:
        raise ValueError("cohort A does not match the archived session IDs")
    if [s.session_id for s in b_sessions] != pilot["session_ids"]:
        raise ValueError("cohort B does not match the pilot checkpoint session IDs")
    if archive["model"] != pilot["model"]:
        raise ValueError("cohort A and B models differ")

    a_payload = archive["arms"]["batch8"]["extractions"]
    a_pairs = [
        (
            session,
            [
                Memory(**{**item, "event_time": None, "valid_from": None, "valid_to": None})
                for item in a_payload[session.session_id]
            ],
        )
        for session in a_sessions
    ]
    b_pairs = [
        (session, [Memory(**item) for item in pilot["baseline"][session.session_id]])
        for session in b_sessions
    ]
    by_id = {item.question_id: item for item in instances}
    c_scoped = namespaced_sessions([by_id[qid] for qid in manifest["question_ids"]])
    store = SQLiteMemoryStore(REPO / "stores/v2b-gate16.db", read_only=True)
    store.initialize()
    try:
        memories_by_session: dict[str, list[Memory]] = {}
        for namespace in {namespace for namespace, _ in c_scoped}:
            for memory in store.iter_all(namespace):
                if memory.source_session_id:
                    memories_by_session.setdefault(memory.source_session_id, []).append(memory)
    finally:
        store.close()
    c_pairs = [
        (session, memories_by_session.get(scoped_session_id(namespace, session.session_id), []))
        for namespace, session in c_scoped
    ]
    cohorts = {
        "A_first60_extraction": _summary(a_pairs, source="archived batch8 extraction"),
        "B_second60_extraction": _summary(b_pairs, source="pilot baseline extraction"),
        "C_gate16_store": _summary(c_pairs, source="built store after deduplication"),
    }
    a, b, c = cohorts.values()
    gates = {
        "A_matches_history": a["specifics"] == recorded_a["specifics"]
        and a["recall"]
        == next(row["overall"] for row in recorded_a["arms"] if row["arm"] == "batch8"),
        "B_matches_history": b["recall"] == recorded_b["baseline"]["overall"],
        "C_matches_history": c["specifics"] == recorded_c["specifics"]
        and c["specifics_kept"] == recorded_c["specifics_kept"],
    }
    if not all(gates.values()):
        raise ValueError(f"recomputed totals do not match history: {gates}")
    ids = {name: set(row["session_ids"]) for name, row in cohorts.items()}
    names = list(ids)
    overlap = {
        f"{names[i]}__{names[j]}": len(ids[names[i]] & ids[names[j]])
        for i in range(len(names))
        for j in range(i + 1, len(names))
    }
    for row in cohorts.values():
        del row["session_ids"]
    return {
        "experiment": "batch8-cohort-attribution-v1",
        "class": "development",
        "model": archive["model"],
        "provider_calls": 0,
        "gates": gates,
        "cohorts": cohorts,
        "session_id_overlap": overlap,
        "C_reweighted_to_A_facet_mix": reweight(c, a),
        "C_reweighted_to_B_facet_mix": reweight(c, b),
        "A_reweighted_to_C_facet_mix": reweight(a, c),
        "B_reweighted_to_C_facet_mix": reweight(b, c),
        "limitation": (
            "The three readings use different sessions. C has no archived pre-dedup "
            "extraction for those same sessions, so cohort and pipeline effects cannot "
            "be causally separated from these artifacts."
        ),
    }


def render(result: dict) -> str:
    rows = result["cohorts"]
    lines = [
        "# Batch 8 cohort attribution v1",
        "",
        "> Retrospective development analysis; zero model calls. Historical results unchanged.",
        "",
        "| cohort | stage | sessions | kept | recall | memories/session | median source chars |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in rows.items():
        lines.append(
            f"| {name} | {row['source']} | {row['sessions']} | "
            f"{row['specifics_kept']}/{row['specifics']} | {row['recall']:.1%} | "
            f"{row['memories_per_session']:.2f} | {row['median_source_chars']:.0f} |"
        )
    lines += [
        "",
        "## Facet composition",
        "",
        "| facet | A stated/kept | B stated/kept | C stated/kept |",
        "|---|---:|---:|---:|",
    ]
    for facet in rows["C_gate16_store"]["facets"]:
        values = [
            f"{row['facets'][facet]['stated']}/{row['facets'][facet]['kept']}"
            for row in rows.values()
        ]
        lines.append(f"| {facet} | {' | '.join(values)} |")
    lines += ["", "## Cohort and stage limits", ""]
    for name, value in result["session_id_overlap"].items():
        lines.append(f"- Session-ID overlap `{name}`: {value}.")
    for label in ("C_reweighted_to_A_facet_mix", "C_reweighted_to_B_facet_mix"):
        row = result[label]
        lines.append(
            f"- `{label}`: {row['recall_on_shared_facets']:.1%} over "
            f"{row['target_specifics_covered']}/{row['target_specifics_total']} "
            "specifics in shared facets."
        )
    lines += ["", result["limitation"], "", "## Historical-total checks", ""]
    lines += [f"- {name}: {'PASS' if value else 'FAIL'}" for name, value in result["gates"].items()]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-out", type=Path, default=REPO / "results/analysis/batch8-cohort-attribution-v1.json"
    )
    parser.add_argument(
        "--md-out", type=Path, default=REPO / "results/analysis/batch8-cohort-attribution-v1.md"
    )
    args = parser.parse_args()
    result = analyse()
    print(write_report(result, render, json_out=args.json_out, md_out=args.md_out), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
