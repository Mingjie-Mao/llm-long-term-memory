"""Stage A of research/ROADMAP.md: the t4 baseline, the regression set, and where each
wrong answer's evidence stood. Zero model calls; read-only on the stores.

Run: `python tools/t4_baseline_audit.py`

Writes `results/analysis/t4-baseline-v1.json` (baseline inventory, regression set and
one record per question any baseline run got wrong) and `.md` (the summary and the
first-loss table, with the hand-written notes from `research/t4-first-loss-notes.json`).

Baseline: the six t4 runs already on disk — heldout100 (`two_stage_raw_primary_t4`,
registered t3/t4 comparison) and dev100 (the unmodified t4 control arm of
`t4-scoped-regression-dev100-v3`, graded in its `.grades.jsonl`). Nothing is re-run.

Evidence delivery is measured on turn identity, not on text: the excerpt selection is
replayed with t4's exact parameters, checked against the turn count each run recorded,
and a gold turn (LongMemEval `has_answer`) counts as delivered when it is among those
excerpts or among the turns the run's raw fallback recovered. Evidence that reached the
answer only through a structured memory is not counted here; the notes record it.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import read_jsonl, sha256_file  # noqa: E402

from llm_long_term_memory.ingest.extract import _parse_date  # noqa: E402
from llm_long_term_memory.retrieve.excerpts import archive_excerpts  # noqa: E402
from llm_long_term_memory.store import SQLiteMemoryStore  # noqa: E402
from llm_long_term_memory.store.session_keys import external_session_id  # noqa: E402

OUT = REPO / "results/analysis/t4-baseline-v1"
NOTES = REPO / "research/t4-first-loss-notes.json"
RAW = REPO / "results/raw"

BASELINE = {
    "heldout100": {
        "store": "heldout100",
        "runs": [
            {"rows": f"two_stage_raw_primary_t4.heldout100-t3t4-rep{r}.jsonl", "grades": None}
            for r in (1, 2, 3)
        ],
    },
    "dev100": {
        "store": "dev100",
        "runs": [
            {
                "rows": f"t4-scoped-regression-dev100-v3-control-rep{r}.jsonl",
                "grades": f"t4-scoped-regression-dev100-v3-control-rep{r}.grades.jsonl",
            }
            for r in (1, 2, 3)
        ],
    },
}

# t4's excerpt parameters (cli.py `two_stage_raw_primary_t4`, runtime ANSWER_PROFILES).
T4_EXCERPTS = {"time_notes": True, "prefer_user": True, "time_window": True}
T4_TOKENS = 4000


def _truthy(value) -> bool:
    return value is True or value == "True"


def load_runs(split: str) -> list[dict[str, dict]]:
    """One {question_id: row} per run, `correct` taken from the run's own grades."""
    runs = []
    for spec in BASELINE[split]["runs"]:
        rows = {r["question_id"]: r for r in read_jsonl(RAW / spec["rows"])}
        if spec["grades"]:
            for g in read_jsonl(RAW / spec["grades"]):
                rows[g["question_id"]]["correct"] = g["verdict"]["correct"]
                rows[g["question_id"]]["judge_reason"] = g["verdict"]["reason"]
        runs.append(rows)
    return runs


def gold_turns(question: dict) -> set[tuple[str, int]]:
    return {
        (sid, i)
        for sid, session in zip(
            question["haystack_session_ids"], question["haystack_sessions"], strict=True
        )
        for i, turn in enumerate(session)
        if turn.get("has_answer")
    }


def delivery(store, question: dict, runs: list[dict]) -> dict:
    """Which gold turns reached the context, per run."""
    qid = question["question_id"]
    excerpts = archive_excerpts(
        store,
        qid,
        question["question"],
        T4_TOKENS,
        asked_on=_parse_date(question["question_date"]),
        **T4_EXCERPTS,
    )
    recorded = {int(run[qid]["notes"]["raw_primary_turns"]) for run in runs}
    replay_matches = recorded == {len(excerpts.turns)}
    packed = {(external_session_id(t.session_id), t.turn_index) for t in excerpts.turns}
    gold = gold_turns(question)
    per_run = []
    for run in runs:
        fallback = set()
        for ref in run[qid]["notes"].get("fallback_turns") or []:
            sid, _, index = ref.rpartition(":")
            fallback.add((sid, int(index)))
        per_run.append(len(gold & (packed | fallback)))
    return {
        "gold_turns": len(gold),
        "delivered_by_excerpts": len(gold & packed),
        "delivered_per_run": per_run,
        "missing_turns": sorted(f"{s}:{i}" for s, i in gold - packed),
        "replay_matches_recorded_turns": replay_matches,
    }


def auto_layer(record: dict) -> str:
    """The first loss the numbers alone can establish; the notes refine it."""
    if record["is_abstention"]:
        return "abstention"
    d = record["delivery"]
    if d["gold_turns"] and min(d["delivered_per_run"]) < d["gold_turns"]:
        return "evidence_incomplete"
    return "evidence_complete"


def build() -> dict:
    questions = {
        q["question_id"]: q
        for q in json.loads((REPO / "data/longmemeval_s_cleaned.json").read_text(encoding="utf-8"))
    }
    notes = json.loads(NOTES.read_text(encoding="utf-8")) if NOTES.exists() else {}
    report: dict = {"baseline": {}, "regression_set": [], "wrong": []}
    for split, spec in BASELINE.items():
        runs = load_runs(split)
        qids = sorted(runs[0])
        if any(sorted(run) != qids for run in runs):
            raise ValueError(f"{split}: runs do not cover the same questions")
        report["baseline"][split] = {
            "store": spec["store"],
            "runs": [
                {
                    "rows": spec_run["rows"],
                    "rows_sha256": sha256_file(RAW / spec_run["rows"]),
                    "grades": spec_run["grades"],
                    "grades_sha256": sha256_file(RAW / spec_run["grades"])
                    if spec_run["grades"]
                    else None,
                    "correct": sum(_truthy(row["correct"]) for row in run.values()),
                }
                for spec_run, run in zip(spec["runs"], runs, strict=True)
            ],
        }
        report["regression_set"].extend(qids)
        store = SQLiteMemoryStore(REPO / "stores" / f"{spec['store']}.db", read_only=True)
        try:
            for qid in qids:
                right = [_truthy(run[qid]["correct"]) for run in runs]
                if all(right):
                    continue
                q = questions[qid]
                record = {
                    "question_id": qid,
                    "split": split,
                    "question_type": q["question_type"],
                    "is_abstention": qid.endswith("_abs"),
                    "correct_of_3": sum(right),
                    "question": q["question"],
                    "gold": str(q["answer"]),
                    "hypotheses": [run[qid]["hypothesis"] for run in runs],
                    "judge_reasons": [run[qid]["judge_reason"] for run in runs],
                    "delivery": delivery(store, q, runs),
                }
                record["auto_layer"] = auto_layer(record)
                note = notes.get(qid, {})
                record["first_loss"] = note.get("first_loss")
                record["failure"] = note.get("failure")
                record["contributing"] = note.get("contributing")
                record["ambiguous"] = bool(note.get("ambiguous"))
                record["note"] = note.get("note")
                report["wrong"].append(record)
        finally:
            store.close()
    report["ambiguous"] = sorted(r["question_id"] for r in report["wrong"] if r["ambiguous"])
    labelled = {r["question_id"] for r in report["wrong"]}
    report["unlabelled"] = sorted(q for q in labelled if not notes.get(q, {}).get("first_loss"))
    report["stale_notes"] = sorted(q for q in notes if not q.startswith("_") and q not in labelled)
    return report


def render(report: dict) -> str:
    lines = [
        "# t4 baseline, regression set and first losses (stage A)",
        "",
        "> Generated by `tools/t4_baseline_audit.py`; zero model calls. Exposed",
        "> development splits. First-loss labels are hand-written in",
        "> `research/t4-first-loss-notes.json` (drafted by the assistant from the saved",
        "> answers and contexts; to be confirmed by the maintainer).",
        "",
        "## Baseline (existing runs, not re-run)",
        "",
        "| split | run | correct |",
        "|---|---|---:|",
    ]
    for split, b in report["baseline"].items():
        for run in b["runs"]:
            lines.append(f"| {split} | `{run['rows']}` | {run['correct']} |")
    wrong = report["wrong"]
    rows_wrong = sum(3 - r["correct_of_3"] for r in wrong)
    replay_ok = sum(r["delivery"]["replay_matches_recorded_turns"] for r in wrong)
    lines += [
        "",
        f"Regression set: {len(report['regression_set'])} questions (dev100 + heldout100). "
        f"Questions wrong in at least one run: {len(wrong)}; wrong rows: {rows_wrong} of 600. "
        f"Excerpt replay matches the recorded turn count on {replay_ok}/{len(wrong)}.",
        "",
        "## Wrong rows by first loss",
        "",
        "| first loss | questions | wrong rows |",
        "|---|---:|---:|",
    ]
    by_layer: Counter = Counter()
    by_layer_rows: Counter = Counter()
    for r in wrong:
        layer = r["first_loss"] or f"unlabelled ({r['auto_layer']})"
        by_layer[layer] += 1
        by_layer_rows[layer] += 3 - r["correct_of_3"]
    for layer, n in by_layer.most_common():
        lines.append(f"| {layer} | {n} | {by_layer_rows[layer]} |")
    lines += [
        "",
        "## Answer-side failures (unambiguous items)",
        "",
        "| failure | questions | wrong rows | memory pointed the wrong way |",
        "|---|---:|---:|---|",
    ]
    by_failure: dict[str, list] = {}
    for r in wrong:
        if r["first_loss"] == "answer" and not r["ambiguous"]:
            by_failure.setdefault(r["failure"], []).append(r)
    for failure, items in sorted(by_failure.items(), key=lambda kv: -len(kv[1])):
        contributing = ", ".join(
            f"`{r['question_id']}` ({r['contributing']})" for r in items if r["contributing"]
        )
        lines.append(
            f"| {failure} | {len(items)} | {sum(3 - r['correct_of_3'] for r in items)} |"
            f" {contributing} |"
        )
    lines += [
        "",
        f"Ambiguous items, reported separately and never counted as improvements: "
        f"{', '.join(f'`{q}`' for q in report['ambiguous']) or 'none'}.",
        "",
        "## Per question",
        "",
        "| question | split | type | correct/3 | gold turns delivered | first loss | failure |",
        "|---|---|---|---:|---|---|---|",
    ]
    for r in sorted(wrong, key=lambda r: (r["first_loss"] or "", r["question_id"])):
        d = r["delivery"]
        delivered = (
            "n/a"
            if not d["gold_turns"]
            else f"{min(d['delivered_per_run'])}-{max(d['delivered_per_run'])}/{d['gold_turns']}"
        )
        lines.append(
            f"| `{r['question_id']}` | {r['split']} | {r['question_type']} | {r['correct_of_3']} |"
            f" {delivered} | {r['first_loss'] or r['auto_layer']} | {r['failure'] or ''} |"
        )
    return "\n".join(lines) + "\n"


def main() -> dict:
    report = build()
    OUT.with_suffix(".json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    OUT.with_suffix(".md").write_text(render(report), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(render(main()))
