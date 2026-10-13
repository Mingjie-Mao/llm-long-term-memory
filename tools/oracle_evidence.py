"""Upper bound for an evidence selector: answer from the gold turns alone.

    python tools/oracle_evidence.py

For every answer-layer failure of the t4 baseline (research/t4-first-loss-notes.json,
first_loss = answer, abstention questions excluded: they have no gold turn), t4's own
first request is rebuilt with zero calls and its 4,000 tokens of retrieved turns are
replaced by only the turns LongMemEval marks `has_answer`, rendered exactly as t4
renders excerpts. Memories, hydration, rules, schema and limits stay t4's; no fallback.
3 samples per question, paired with the same question's 3 baseline samples.

v1 (results/analysis/oracle-evidence-v1.md) also dropped the memory context; the
answerer then answered "no_evidence" on 32 of 66 samples ("there are no memories of
..."), so it understated the bound and is kept only as a record of that mistake.

If even perfect evidence leaves the answers wrong, a learned evidence selector has
little to gain; that is the question this answers. Oracle evidence uses gold labels, so
this is a diagnostic, never a deployable arm. Development evidence.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from t4_baseline_audit import load_runs  # noqa: E402

from llm_long_term_memory.answering import (  # noqa: E402
    AnswerVerdict,
    NotedAnswerVerdict,
    asks_about_time,
    asks_for_aggregate,
    count_time_system,
)
from llm_long_term_memory.ingest.extract import _parse_date  # noqa: E402
from llm_long_term_memory.retrieve.excerpts import ArchiveExcerpts  # noqa: E402
from llm_long_term_memory.runtime.context import ANSWER_TEMPLATE  # noqa: E402
from llm_long_term_memory.store import SQLiteMemoryStore, scoped_session_id  # noqa: E402

OUT = REPO / "results/analysis/oracle-evidence-v2"
SAMPLES = 3


def gold_block(store, question: dict) -> str:
    qid = question["question_id"]
    turns, dates = [], {}
    for sid, session in zip(
        question["haystack_session_ids"], question["haystack_sessions"], strict=True
    ):
        wanted = {i for i, t in enumerate(session) if t.get("has_answer")}
        if not wanted:
            continue
        scoped = scoped_session_id(qid, sid)
        stored = store.get_session(scoped)
        dates[scoped] = stored.started_at if stored else None
        turns += [t for t in store.turns_for_session(scoped) if t.turn_index in wanted]
    return ArchiveExcerpts(
        turns=turns,
        session_dates=dates,
        asked_on=_parse_date(question["question_date"]),
        time_notes=True,
    ).render()


class _Recorder:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)

        class Completion:
            text = '{"status": "answer", "answer": "ok"}'
            input_tokens = output_tokens = 1
            api_latency_ms = 0.0

        return Completion()


EXCERPTS = "Excerpts from the user's earlier conversations"
DATE_LINE = "\n\nToday's date is "


def t4_request_with(split: str, question: dict, block: str) -> dict:
    """t4's own first request for this question, with its retrieved excerpts replaced by
    `block`. Memories, hydrated sentences, rules, schema and limits stay t4's: only the
    choice of verbatim turns changes (v2 of this diagnostic; v1 also dropped the
    memories, and the answerer then reported having no memories at all)."""
    from llm_long_term_memory import cli
    from llm_long_term_memory.conversation import AnswerRequest

    recorder = _Recorder()
    _, _, runner, _, _ = cli._build(
        "two_stage_raw_primary_t4",
        "configs/fallback.yaml",
        split,
        client_override=recorder,
        read_only_store=True,
    )
    runner.answer_request(
        AnswerRequest(
            question=question["question"],
            asked_on=question["question_date"],
            user_id=question["question_id"],
        )
    )
    request = dict(recorder.calls[0])
    prompt = request["prompt"]
    start, end = prompt.index(EXCERPTS), prompt.index(DATE_LINE)
    request["prompt"] = prompt[:start] + block + prompt[end:]
    request.pop("est_input_tokens", None)
    return request


def answer_v2(client, request: dict, schema) -> str:
    completion = client.generate(**request)
    try:
        verdict = schema.model_validate_json(completion.text)
    except ValueError:
        return "I do not know."
    text = (verdict.answer or "").strip()
    return text if text and not text.startswith(("{", "```")) else "I do not know."


def answer(client, model: str, question: dict, block: str) -> str:
    q = question["question"]
    counting, timing = asks_for_aggregate(q), asks_about_time(q)
    noted = counting or timing
    schema = NotedAnswerVerdict if noted else AnswerVerdict
    completion = client.generate(
        role="answerer",
        model=model,
        prompt=ANSWER_TEMPLATE.format(context=block, date=question["question_date"], question=q),
        system=count_time_system(counting, timing),
        temperature=0.0,
        max_output_tokens=1024 if noted else 512,
        schema=schema,
    )
    try:
        verdict = schema.model_validate_json(completion.text)
    except ValueError:
        return "I do not know."
    text = (verdict.answer or "").strip()
    return text if text and not text.startswith(("{", "```")) else "I do not know."


def main(samples: tuple[int, ...] = (1, 2, 3)) -> dict:
    from llm_long_term_memory import cli

    questions = {
        q["question_id"]: q
        for q in json.loads((REPO / "data/longmemeval_s_cleaned.json").read_text(encoding="utf-8"))
    }
    notes = json.loads((REPO / "research/t4-first-loss-notes.json").read_text(encoding="utf-8"))
    targets = [
        qid
        for qid, n in notes.items()
        if not qid.startswith("_") and n["first_loss"] == "answer" and not qid.endswith("_abs")
    ]
    baseline = {split: load_runs(split) for split in ("dev100", "heldout100")}
    rows = []
    for n in samples:
        _, _, runner, judge, _ = cli._build(
            "full_context",
            "configs/fallback.yaml",
            response_cache=str(REPO / "stores/cache/oracle-evidence-v2.db"),
            sample=f"rep{n}",
        )
        for qid in targets:
            split = "dev100" if qid in baseline["dev100"][0] else "heldout100"
            store = SQLiteMemoryStore(REPO / "stores" / f"{split}.db", read_only=True)
            try:
                block = gold_block(store, questions[qid])
            finally:
                store.close()
            request = t4_request_with(split, questions[qid], block)
            text = answer_v2(runner.client, request, request["schema"])
            q = questions[qid]
            verdict = judge.grade(
                question=q["question"],
                gold=str(q["answer"]),
                hypothesis=text,
                is_abstention=False,
                question_type=q["question_type"],
            )
            rows.append(
                {"question_id": qid, "sample": n, "answer": text, "correct": verdict.correct}
            )
    report = {"targets": len(targets), "questions": []}
    for qid in targets:
        split = "dev100" if qid in baseline["dev100"][0] else "heldout100"
        base = sum(run[qid]["correct"] in (True, "True") for run in baseline[split])
        oracle = sum(r["correct"] for r in rows if r["question_id"] == qid)
        report["questions"].append(
            {
                "question_id": qid,
                "split": split,
                "failure": notes[qid]["failure"],
                "t4_correct_of_3": base,
                "oracle_correct_of_3": oracle,
                "oracle_answers": [r["answer"] for r in rows if r["question_id"] == qid],
            }
        )
    return report


def render(report: dict) -> str:
    qs = report["questions"]
    base = sum(q["t4_correct_of_3"] for q in qs)
    oracle = sum(q["oracle_correct_of_3"] for q in qs)
    by = Counter()
    for q in qs:
        by[(q["failure"], "n")] += 1
        by[(q["failure"], "t4")] += q["t4_correct_of_3"]
        by[(q["failure"], "oracle")] += q["oracle_correct_of_3"]
    lines = [
        "# Answering with only the gold turns as excerpts (selector upper bound, v2)",
        "",
        "> Generated by `tools/oracle_evidence.py`. Diagnostic only: the evidence comes from",
        "> gold labels. t4's request with its retrieved turns replaced by the gold turns.",
        "> Development evidence on exposed questions.",
        "",
        f"Answer-layer failures (abstention excluded): {len(qs)} questions, 3 samples each.",
        f"t4 with its full context: {base}/{3 * len(qs)} correct. Gold turns only:"
        f" {oracle}/{3 * len(qs)}.",
        "",
        "| failure | questions | t4 | gold turns only |",
        "|---|---:|---:|---:|",
    ]
    for failure in sorted({q["failure"] for q in qs}):
        n = by[(failure, "n")]
        lines.append(
            f"| {failure} | {n} | {by[(failure, 't4')]}/{3 * n} |"
            f" {by[(failure, 'oracle')]}/{3 * n} |"
        )
    lines += ["", "| question | failure | t4 | gold only |", "|---|---|---:|---:|"]
    for q in sorted(qs, key=lambda q: q["oracle_correct_of_3"] - q["t4_correct_of_3"]):
        lines.append(
            f"| `{q['question_id']}` | {q['failure']} | {q['t4_correct_of_3']} |"
            f" {q['oracle_correct_of_3']} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    # `python tools/oracle_evidence.py 2` fills one sample's cache; no argument runs all
    # three and writes the report (cached samples cost nothing).
    if len(sys.argv) > 1:
        main(tuple(int(a) for a in sys.argv[1:]))
        raise SystemExit(0)
    report = main()
    OUT.with_suffix(".json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    OUT.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(render(report))
