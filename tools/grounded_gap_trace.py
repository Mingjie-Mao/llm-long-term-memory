"""Trace remaining exposed train150 evidence gaps using the actual runtime, zero APIs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import sha256_file, write_report  # noqa: E402
from grounded_context_replay import StubClient  # noqa: E402


def analyse(report_path: Path) -> dict:
    from llm_long_term_memory import cli
    from llm_long_term_memory.conversation import AnswerRequest
    from llm_long_term_memory.evaluation.datasets.longmemeval import load
    from llm_long_term_memory.retrieve.excerpts import bm25_rank, fact_keyed_texts, fuse
    from llm_long_term_memory.store import external_session_id

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["store"] != "train150" or report["context_only"]:
        raise ValueError("item-level tracing is permitted only for exposed train150")
    source = REPO / "src/llm_long_term_memory/runtime/grounded_answering.py"
    if report["candidate_source_sha256"] != sha256_file(source):
        raise ValueError("candidate changed since context replay")
    qids = [r["question_id"] for r in report["rows"] if not r["candidate"]]
    stub = StubClient()
    _, settings, runner, _, _ = cli._build(
        "two_stage_raw_primary_grounded_v3",
        "configs/fallback.yaml",
        "train150",
        client_override=stub,
        read_only_store=True,
    )
    instances = {i.question_id: i for i in load("s", settings.data_dir)}
    rows = []
    try:
        for qid in qids:
            instance = instances[qid]
            answer = runner.answer_request(
                AnswerRequest(instance.question, instance.question_date, qid)
            )
            own = {
                t.id: t
                for sid in runner.store.session_ids_for_user(qid)
                for t in runner.store.turns_for_session(sid)
            }
            at = {(external_session_id(t.session_id), t.turn_index): t for t in own.values()}
            lexical = bm25_rank(
                instance.question, fact_keyed_texts(runner.store, qid, list(own.values()))
            )
            vector = runner.encoder.encode_one(instance.question)
            dense = [
                tid
                for tid, _ in runner.raw_primary_turn_index.search(
                    vector, limit=len(runner.raw_primary_turn_index)
                )
                if tid in own
            ]
            memories = {m.id: m for m in runner.store.iter_all(qid)}
            selected = [memories[mid] for mid in answer.retrieved_ids if mid in memories]
            anchor_at = {(t.session_id, t.turn_index): t.id for t in own.values()}
            anchors = list(
                dict.fromkeys(
                    anchor_at[(m.source_session_id, m.source_turn_index)]
                    for m in selected
                    if (m.source_session_id, m.source_turn_index) in anchor_at
                )
            )
            fused = fuse(lexical[:500], dense, anchors)
            audit = answer.notes["grounded_calls"][0]["evidence"]
            present = {s["source_id"] for s in audit["sources"] if s["kind"] == "raw"}
            gold = []
            for session in instance.sessions:
                for i, turn in enumerate(session.turns):
                    if not turn.has_answer:
                        continue
                    stored = at.get((session.session_id, i))
                    tid = stored.id if stored else None
                    gold.append(
                        {
                            "session": session.session_id,
                            "turn": i,
                            "source_id": tid,
                            "stored_text_matches": bool(stored and stored.content == turn.content),
                            "present": tid in present,
                            "text": turn.content,
                            "lexical_rank": lexical.index(tid) + 1 if tid in lexical else None,
                            "dense_rank": dense.index(tid) + 1 if tid in dense else None,
                            "fused_rank": fused.index(tid) + 1 if tid in fused else None,
                            "body_tokens_estimate": int(len(turn.content) / 4.6),
                        }
                    )
            rows.append(
                {
                    "question_id": qid,
                    "question": instance.question,
                    "question_date": instance.question_date,
                    "reference_answer": instance.answer,
                    "gold_sources": gold,
                    "selected_memory_texts": [m.content for m in selected],
                    "context_tokens": answer.context_tokens,
                    "context_audit": audit,
                    "classification": (
                        "Ranking/budget gap in full marked turns; "
                        "semantic sufficiency not automatically established."
                    ),
                }
            )
    finally:
        runner.store.close()
    return {
        "experiment_class": "development-gap-trace",
        "provider_requests": 0,
        "input_sha256": sha256_file(report_path),
        "source_sha256": sha256_file(source),
        "generator_sha256": sha256_file(Path(__file__)),
        "rows": rows,
    }


def render(result):
    lines = [
        "# Remaining train150 marked-turn gaps",
        "",
        "Zero provider calls; not an answer accuracy result.",
        "",
        "| Question | Missing marked turns | Fused ranks | Estimated body tokens |",
        "|---|---:|---|---|",
    ]
    for row in result["rows"]:
        missing = [s for s in row["gold_sources"] if not s["present"]]
        lines.append(
            f"| {row['question_id']} | {len(missing)} | "
            f"{[s['fused_rank'] for s in missing]} | "
            f"{[s['body_tokens_estimate'] for s in missing]} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.json_out, args.md_out):
        if path.exists() or any(p in {"frozen", "sealed"} for p in path.resolve().parts):
            raise SystemExit("refusing overwrite or immutable evidence")
    result = analyse(REPO / "results/analysis/grounded-context-v3.train150.json")
    print(write_report(result, render, json_out=args.json_out, md_out=args.md_out))


if __name__ == "__main__":
    main()
