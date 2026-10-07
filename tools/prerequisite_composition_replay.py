"""Replay exact v3 page selections with proposed raw-before-context assembly; no calls."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
from analysis_io import rows_by_question, sha256_file  # noqa: E402


def measure():
    from llm_long_term_memory import cli
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.conversation import AnswerRequest
    from llm_long_term_memory.evaluation.datasets.longmemeval import load
    from llm_long_term_memory.runtime.evidence_pages import PageSelection
    from llm_long_term_memory.runtime.grounded_answering import (
        GroundedAnswererV15,
        GroundedAnswererV17,
        GroundedAnswererV18,
    )
    from llm_long_term_memory.store import external_session_id

    saved_path = REPO / "results/raw/prerequisite-reader-v18-v3.readers.jsonl"
    saved = rows_by_question(saved_path)
    instances = {i.question_id: i for i in load("s", Settings().data_dir) if i.question_id in saved}

    class Client:
        def generate(self, **kw):
            if kw["schema"] is PageSelection:
                old = self.pages.pop(0)
                context = kw["prompt"].split("\n\nQuestion date:")[0]
                assert (
                    hashlib.sha256(context.encode()).hexdigest()
                    == old["evidence"]["context_sha256"]
                )
                text = json.dumps(old["selection"])
            else:
                text = kw["schema"](
                    scope_complete=True, status="answer", answer="stub"
                ).model_dump_json()
            return SimpleNamespace(text=text, input_tokens=0, output_tokens=0, api_latency_ms=0)

    client = Client()
    _, _, runner, _, _ = cli._build(
        "two_stage_raw_primary_grounded_v18",
        "configs/fallback.yaml",
        "train150",
        client_override=client,
        read_only_store=True,
    )
    engine = runner.grounded
    original_context = GroundedAnswererV18.hydrate_review_context
    rows = []
    try:
        for qid, instance in instances.items():
            client.pages = list(saved[qid]["answer"]["notes"]["archive_review"]["pages"])

            def extend(ledger, request, memories, query_vector):
                if not client.pages:
                    return GroundedAnswererV18.extend_ledger(
                        engine, ledger, request, memories, query_vector
                    )
                GroundedAnswererV15.extend_ledger(engine, ledger, request, memories, query_vector)
                baseline = list(ledger.sources)

                def hydrate(replacement, request):
                    from dataclasses import replace

                    added = 0
                    for s in baseline:
                        if s.kind == "raw":
                            added += replacement.add(
                                replace(s, id=f"E{len(replacement.sources) + 1}")
                            )
                    if re.search(r"\bwhere\b", request.question, re.I):
                        added += original_context(engine, replacement, request)
                    return added

                engine.hydrate_review_context = hydrate
                return GroundedAnswererV17.extend_ledger(
                    engine, ledger, request, memories, query_vector
                )

            engine.extend_ledger = extend
            answer = runner.answer_request(
                AnswerRequest(instance.question, instance.question_date, qid)
            )
            assert not client.pages
            sources = answer.notes["grounded_calls"][0]["evidence"]["sources"]
            found = {
                (external_session_id(s["session_id"]), s["turn_index"], s["text_sha256"])
                for s in sources
                if s["kind"] == "raw"
            }
            gold = {
                (s.session_id, n, hashlib.sha256(t.content.encode()).hexdigest())
                for s in instance.sessions
                for n, t in enumerate(s.turns)
                if t.has_answer
            }
            rows.append(
                {
                    "question_id": qid,
                    "complete": bool(gold) and gold <= found,
                    "context_tokens": answer.context_tokens,
                    "missing": [
                        {"session_id": sid, "turn_index": n} for sid, n, _ in sorted(gold - found)
                    ],
                }
            )
    finally:
        runner.store.close()
    return {
        "provider_calls": 0,
        "experiment_class": "development-offline-context-assembly",
        "rows": rows,
        "input_sha256": sha256_file(saved_path),
        "generator_sha256": sha256_file(Path(__file__)),
        "limitation": (
            "Same saved page selection; hypothetical assembly only, not new answers or QA accuracy."
        ),
    }


if __name__ == "__main__":
    output = REPO / "results/analysis/prerequisite-composition-floor-v1.json"
    if output.exists():
        raise ValueError("namespace exists")
    result = measure()
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["rows"]))
