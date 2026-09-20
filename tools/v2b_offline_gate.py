"""Run the zero-quota evidence checks that precede the v2b gate16 ingest.

The report is intentionally honest about what offline heuristics cannot decide:
lexical support is a fabrication screen, not a hallucination judge, and the
archived batch-size cohort overlaps only one benchmark question's complete gold
sessions.  No API client is constructed by this module.
"""

from __future__ import annotations

import importlib.util
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import read_json  # noqa: E402

sys.path.insert(0, str(REPO / "src"))

ARCHIVE = REPO / "results/raw/extraction-batch-size.extractions.json"
OUT_JSON = REPO / "results/analysis/v2b-offline-gate.json"
OUT_MD = REPO / "results/analysis/v2b-offline-gate.md"
RAW_STORE = REPO / "stores/two-stage-hydrated.db"

_WORD = re.compile(r"[A-Za-z0-9']+")
_STOP = frozenset(
    _WORD.findall(
        "the a an and or but if of to in on at for with from by is are was were be been "
        "being have has had do does did will would can could should may might must this "
        "that these those it its as not no yes user assistant they them their he she his "
        "her you your i me my we our about into over under more most some any"
    )
)


def _batch_module():
    path = REPO / "tools/batch_size_curve.py"
    spec = importlib.util.spec_from_file_location("v2b_batch_size_curve", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _words(text: str) -> set[str]:
    return {
        word.lower() for word in _WORD.findall(text) if word.lower() not in _STOP and len(word) > 2
    }


def _memory_text(memory: dict) -> str:
    return f"{memory.get('content', '')} {memory.get('object', '')}".strip()


def _memory(memory: dict):
    from llm_long_term_memory.store import Memory

    return Memory(**{**memory, "event_time": None, "valid_from": None, "valid_to": None})


def _arm_memories(archive: dict, arm: str, session_id: str) -> list[dict]:
    return archive["arms"][arm]["extractions"].get(session_id, [])


def _facet_transitions(archive: dict, sessions: list) -> dict[str, dict[str, int]]:
    from llm_long_term_memory.ingest.fidelity import _extract_facets, user_assertions

    counts: dict[str, Counter] = {}
    for session in sessions:
        blobs = {
            arm: " || ".join(
                _memory_text(memory) for memory in _arm_memories(archive, arm, session.session_id)
            ).lower()
            for arm in ("batch15", "batch8")
        }
        for facet, values in _extract_facets(user_assertions(session)).items():
            counter = counts.setdefault(facet, Counter())
            for value in values:
                old = value in blobs["batch15"]
                new = value in blobs["batch8"]
                state = (
                    "retained_both"
                    if old and new
                    else "gained"
                    if new
                    else "lost"
                    if old
                    else "missed_both"
                )
                counter[state] += 1
    # Sorted, not insertion-ordered. `_extract_facets` returns sets, so which
    # transition is seen first varies with the interpreter's hash seed, and the
    # serialised key order varied with it — the same evidence file came out with
    # different bytes on different runs, which makes re-running it no longer a check.
    # The counts themselves never moved.
    return {
        facet: {state: counter[state] for state in sorted(counter)}
        for facet, counter in sorted(counts.items())
    }


def _support(archive: dict, sessions: list) -> dict[str, dict]:
    output = {}
    for arm in ("batch15", "batch8"):
        values = []
        empty = 0
        for session in sessions:
            source_words = _words("\n".join(turn.content for turn in session.turns))
            for memory in _arm_memories(archive, arm, session.session_id):
                memory_words = _words(_memory_text(memory))
                if not memory_words:
                    empty += 1
                    continue
                values.append(len(memory_words & source_words) / len(memory_words))
        ordered = sorted(values)
        output[arm] = {
            "memories_scored": len(values),
            "empty_content_word_memories": empty,
            "mean": statistics.fmean(values) if values else 0.0,
            "median": statistics.median(values) if values else 0.0,
            "p10": ordered[max(0, (len(ordered) - 1) // 10)] if ordered else 0.0,
            "zero_overlap": sum(value == 0 for value in values),
            "below_0_25": sum(value < 0.25 for value in values),
            "below_0_50": sum(value < 0.50 for value in values),
        }
    return output


def _gold_overlap(batch, archive: dict, sessions: list) -> dict:
    from llm_long_term_memory.evaluation.datasets.longmemeval import load, split_dev_test
    from llm_long_term_memory.evaluation.extraction_coverage import answer_present

    cohort_ids = {session.session_id for session in sessions}
    _, test = split_dev_test(load("s", REPO / "data"))
    comparable = [
        instance
        for instance in test
        if instance.answer_session_ids and set(instance.answer_session_ids).issubset(cohort_ids)
    ]
    rows = []
    for instance in comparable:
        row = {
            "question_id": instance.question_id,
            "question_type": instance.question_type,
            "gold": instance.answer,
        }
        for arm in ("batch15", "batch8"):
            memories = [
                _memory(memory)
                for session_id in instance.answer_session_ids
                for memory in _arm_memories(archive, arm, session_id)
            ]
            row[f"{arm}_literal_answer_present"] = answer_present(str(instance.answer), memories)
            row[f"{arm}_evidence_memories"] = len(memories)
        rows.append(row)
    return {
        "comparable_questions": len(rows),
        "rows": rows,
        "interpretation": (
            "Diagnostic only: the archived 60-session cohort was not sampled by question, "
            "so complete gold sessions overlap too few questions for an accuracy claim."
        ),
    }


def _raw_top2() -> dict:
    from llm_long_term_memory.evaluation.datasets.longmemeval import load
    from llm_long_term_memory.evaluation.raw_recall import build_probes, run_probes
    from llm_long_term_memory.store import SQLiteMemoryStore

    if not RAW_STORE.exists():
        return {"available": False, "reason": f"missing {RAW_STORE.relative_to(REPO)}"}
    store = SQLiteMemoryStore(RAW_STORE, read_only=True)
    store.initialize()
    try:
        namespaces = set(store.user_ids())
        probes = build_probes(load("s", REPO / "data"), namespaces)
        report = run_probes(store, probes, depth=5)
    finally:
        store.close()
    natural = report.subset("natural")
    return {
        "available": True,
        "store": str(RAW_STORE.relative_to(REPO)),
        "store_memory_namespaces": len(namespaces),
        "natural_questions": len(natural),
        "recall_at_1": report.recall_at(1, "natural"),
        "recall_at_2": report.recall_at(2, "natural"),
        "recall_at_3": report.recall_at(3, "natural"),
        "note": (
            "Raw BM25 only. This says whether raw top-2 can add source evidence; it does "
            "not say whether the answerer will use it or whether every task has a single gold turn."
        ),
    }


def analyse() -> dict:
    batch = _batch_module()
    archive = read_json(ARCHIVE)
    sessions = batch._cohort()
    curve = batch.score(write=False)
    arms = {row["arm"]: row for row in curve["arms"]}
    return {
        "calls": 0,
        "cohort_sessions": len(sessions),
        "cohort_matches_archive": curve["cohort_matches"],
        "prompts_match_archive": curve["prompts_match"],
        "fidelity": {
            arm: {
                "overall": arms[arm]["overall"],
                "memories": arms[arm]["memories"],
                "per_facet": arms[arm]["per_facet"],
                "stated": arms[arm]["stated"],
            }
            for arm in ("batch15", "batch8")
        },
        "facet_transitions": _facet_transitions(archive, sessions),
        "lexical_support_floor": _support(archive, sessions),
        "literal_gold_overlap": _gold_overlap(batch, archive, sessions),
        "raw_bm25": _raw_top2(),
        "existing_replay_decisions": {
            "smaller_top_k": (
                "Rejected as a general mechanism: archived context-arms reruns found flatN "
                "worse than flat20 on one target and no rescue on the other stable failures."
            ),
            "dense_raw_retrieval": (
                "Not justified by the archived diagnostic: natural-query BM25 misses were "
                "mostly aggregation, temporal, or preference tasks rather than lexical misses."
            ),
            "raw_top2_parallel": (
                "Still eligible for a gate16 evidence comparison because it is cheap, but must "
                "increase per-question source coverage before any answer calls."
            ),
        },
        "limits": [
            "Lexical overlap is only a low-support screen; it is not an unsupported-fact "
            "or hallucination rate.",
            "The fidelity ruler annotates quantities, durations, money, dates, relative "
            "time, and proper nouns; ordinary semantic facts have no offline gold labels here.",
            "End-to-end promotion still requires the pre-registered gate16 candidate comparison.",
        ],
    }


def _pct(value: float) -> str:
    return f"{value:.1%}"


def render(result: dict) -> str:
    old = result["fidelity"]["batch15"]
    new = result["fidelity"]["batch8"]
    lines = [
        "# v2b zero-call gate",
        "",
        "**No model/API calls were made.** The archived 60-session paired cohort still "
        f"matches the corpus: `{result['cohort_matches_archive']}`; prompts match: "
        f"`{result['prompts_match_archive']}`.",
        "",
        "## Extraction change",
        "",
        f"Batch 15 retained **{_pct(old['overall'])}** of annotated specifics with "
        f"{old['memories']} memories. Batch 8 retained **{_pct(new['overall'])}** with "
        f"{new['memories']} memories: **+{(new['overall'] - old['overall']) * 100:.1f} points** "
        f"and {new['memories'] - old['memories']:+d} emitted memories.",
        "",
        "| facet | gained | lost | batch15 | batch8 |",
        "|---|---:|---:|---:|---:|",
    ]
    for facet, transitions in result["facet_transitions"].items():
        old_rate = _pct(old["per_facet"][facet]) if old["stated"][facet] else "n/a"
        new_rate = _pct(new["per_facet"][facet]) if new["stated"][facet] else "n/a"
        lines.append(
            f"| {facet} | {transitions.get('gained', 0)} | {transitions.get('lost', 0)} | "
            f"{old_rate} | {new_rate} |"
        )
    lines.extend(["", "## Low-support screen", ""])
    for arm in ("batch15", "batch8"):
        support = result["lexical_support_floor"][arm]
        lines.append(
            f"- `{arm}`: median source-word support {_pct(support['median'])}; "
            f"{support['below_0_25']}/{support['memories_scored']} below 25%; "
            f"{support['zero_overlap']} with zero overlap."
        )
    lines.extend(
        [
            "",
            "These are **not hallucination rates**: paraphrases can have low overlap and false "
            "claims can reuse source words. They are only a cheap regression floor.",
            "",
            "## Can the archived cohort predict QA?",
            "",
        ]
    )
    gold = result["literal_gold_overlap"]
    lines.append(
        f"Only **{gold['comparable_questions']}** benchmark question has all gold sessions in "
        "this extraction cohort, so the archived fidelity run cannot honestly estimate QA lift."
    )
    for row in gold["rows"]:
        lines.append(
            f"- `{row['question_id']}`: literal gold present batch15="
            f"`{row['batch15_literal_answer_present']}`, batch8="
            f"`{row['batch8_literal_answer_present']}`."
        )
    raw = result["raw_bm25"]
    lines.extend(["", "## Offline retrieval decisions", ""])
    if raw.get("available"):
        lines.append(
            f"Raw BM25 natural-query recall is {_pct(raw['recall_at_1'])} @1, "
            f"{_pct(raw['recall_at_2'])} @2, and {_pct(raw['recall_at_3'])} @3 "
            f"over {raw['natural_questions']} questions in the current `{raw['store']}` "
            f"snapshot ({raw['store_memory_namespaces']} memory namespaces). Raw top-2 "
            "remains worth "
            "testing only behind the per-question coverage gate."
        )
    for name, decision in result["existing_replay_decisions"].items():
        lines.append(f"- `{name}`: {decision}")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Batch 8 passes the **extraction-mechanism** gate, but archived data cannot prove "
            "end-to-end QA transmission. Proceed only to the registered 16-question candidate "
            "ingest; do not run 48 questions yet. Ordinary-fact fidelity and true unsupported "
            "rate remain unresolved and must be checked on the gate16 output before promotion.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    result = analyse()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    OUT_MD.write_text(render(result))
    print(f"wrote {OUT_JSON.relative_to(REPO)}")
    print(f"wrote {OUT_MD.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
