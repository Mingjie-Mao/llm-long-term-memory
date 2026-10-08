"""Where the wrong answers of `paged-paired-dev100-v18-v1` were lost. No provider calls.

    python tools/error_taxonomy_dev100_v18.py

Reads the run's combined rows after its gate was written (the run is closed; this
does not change it). For every wrong row it asks first whether the gold turns
(LongMemEval `has_answer`) reached the answer context, and only then what the answer
did — the first-loss order of AGENTS.md:

- baseline: verbatim turns = the 4,000-token BM25 excerpts, replayed from the
  read-only store and checked against the recorded turn count, plus the recorded
  fallback turns and the turns of the hydrated memories;
- candidate: the raw sources of its final grounded call.

A gold turn reached by a memory that points at it, but not verbatim, is counted
separately. Reading dev100 per question here makes it development evidence with
per-item failures read; nothing found here may be scored on dev100 again as if blind.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
from failure_taxonomy import _ABSTAIN, _numbers  # noqa: E402

STEM = "paged-paired-dev100-v18-v1"
OUT = REPO / "results/analysis/error-taxonomy-dev100-v18-v1"
ARMS = ("baseline", "candidate")


def turn_key(session_id, index):
    from llm_long_term_memory.store import external_session_id

    return (external_session_id(session_id), index)


def delivered(arm, row, store, instance):
    """(verbatim gold-candidate keys, memory-anchored keys) in the final context."""
    from llm_long_term_memory.retrieve.excerpts import archive_excerpts

    notes = row["notes"]
    verbatim, anchored = set(), set()
    if arm == "baseline":
        excerpts = archive_excerpts(
            store, instance.store_namespace, instance.question, notes["raw_primary_tokens_budget"]
        )
        if len(excerpts.turns) != notes["raw_primary_turns"]:
            raise ValueError(f"baseline excerpt replay differs for {row['question_id']}")
        verbatim |= {turn_key(t.session_id, t.turn_index) for t in excerpts.turns}
        for ref in notes.get("fallback_turns") or []:
            session, _, index = ref.rpartition(":")
            verbatim.add((session, int(index)))
        for memory_id in notes.get("hydrated_memory_ids") or []:
            memory = store.get(memory_id)
            if memory and memory.source_session_id and memory.source_turn_index is not None:
                verbatim.add(turn_key(memory.source_session_id, memory.source_turn_index))
        for memory_id in notes.get("ranked_memory_ids") or []:
            memory = store.get(memory_id)
            if memory and memory.source_session_id and memory.source_turn_index is not None:
                anchored.add(turn_key(memory.source_session_id, memory.source_turn_index))
    else:
        calls = notes.get("grounded_calls") or []
        for source in calls[-1]["evidence"]["sources"] if calls else []:
            if source.get("session_id") is None or source.get("turn_index") is None:
                continue
            key = turn_key(source["session_id"], source["turn_index"])
            (verbatim if source["kind"] == "raw" else anchored).add(key)
    return verbatim, anchored


def extracted(store, instance, gold):
    """Gold turns at least one stored memory of this user is anchored to."""
    return {
        turn_key(m.source_session_id, m.source_turn_index)
        for m in store.iter_all(instance.store_namespace)
        if m.source_session_id and m.source_turn_index is not None
    } & gold


def behaviour(row):
    hypothesis, gold = str(row["hypothesis"]), str(row["gold"])
    if row["notes"].get("answer_status") in {"no_evidence", "insufficient"} or _ABSTAIN.search(
        hypothesis
    ):
        return "abstained"
    g, h = _numbers(gold), _numbers(hypothesis)
    if g and h and g != h:
        return "wrong_number"
    return "wrong_content"


def classify(row, gold, verbatim, anchored, stored):
    if row["is_abstention"]:
        return "abstention_question_answered"
    if not gold:
        return "no_gold_label"
    reached = gold & verbatim
    if reached == gold:
        return "reader:" + behaviour(row)
    if reached:
        return "partial_evidence"
    if gold & anchored:
        return "memory_only_no_verbatim"
    if not stored:
        return "not_extracted_not_retrieved"
    return "retrieval_miss"


def main():
    from llm_long_term_memory.commands.common import manifest_instances
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.evaluation.manifest import load_manifest
    from llm_long_term_memory.store import SQLiteMemoryStore

    if not (REPO / f"results/analysis/{STEM}.gate.json").exists():
        raise SystemExit("run not closed: no gate")
    manifest = load_manifest(REPO / "results/manifests/dev100.json")
    instances = {i.question_id: i for i in manifest_instances(manifest, Settings())}
    store = SQLiteMemoryStore(REPO / "stores/dev100.db", read_only=True)
    store.initialize()
    rows_out, per_question = [], defaultdict(lambda: defaultdict(list))
    try:
        for arm in ARMS:
            for rep in (1, 2, 3):
                path = REPO / f"results/raw/{STEM}-{arm}-rep{rep}.jsonl"
                for line in path.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    qid = row["question_id"]
                    instance = instances[qid]
                    gold = {
                        (s.session_id, i)
                        for s in instance.sessions
                        for i, t in enumerate(s.turns)
                        if t.has_answer
                    }
                    if row["correct"]:
                        per_question[arm][qid].append("correct")
                        continue
                    verbatim, anchored = delivered(arm, row, store, instance)
                    label = classify(
                        row, gold, verbatim, anchored, extracted(store, instance, gold)
                    )
                    per_question[arm][qid].append(label)
                    rows_out.append(
                        {
                            "arm": arm,
                            "repeat": rep,
                            "question_id": qid,
                            "type": row["question_type"],
                            "label": label,
                            "gold_turns": len(gold),
                            "gold_verbatim": len(gold & verbatim),
                            "hypothesis": row["hypothesis"],
                            "gold": row["gold"],
                            "judge_reason": row["judge_reason"],
                        }
                    )
    finally:
        store.close()

    summary = {}
    for arm in ARMS:
        questions = per_question[arm]
        wrong = {q: labels for q, labels in questions.items() if labels.count("correct") < 3}
        stable = {q for q, labels in wrong.items() if "correct" not in labels}
        main_label = {
            q: Counter(x for x in labels if x != "correct").most_common(1)[0][0]
            for q, labels in wrong.items()
        }
        summary[arm] = {
            "wrong_rows": sum(len([x for x in v if x != "correct"]) for v in questions.values()),
            "questions_ever_wrong": len(wrong),
            "questions_always_wrong": len(stable),
            "rows_by_label": dict(Counter(r["label"] for r in rows_out if r["arm"] == arm)),
            "questions_by_label": dict(Counter(main_label.values())),
            "always_wrong_by_label": dict(Counter(main_label[q] for q in stable)),
            "flaky_by_label": dict(Counter(main_label[q] for q in wrong if q not in stable)),
            "by_type_label": {
                t: dict(Counter(main_label[q] for q in wrong if instances[q].question_type == t))
                for t in sorted({instances[q].question_type for q in wrong})
            },
        }
    result = {
        "source_run": STEM,
        "evidence_class": "exposed dev100, per-item failures now read; development only",
        "provider_calls": 0,
        "summary": summary,
        "rows": rows_out,
    }
    OUT.with_suffix(".json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
