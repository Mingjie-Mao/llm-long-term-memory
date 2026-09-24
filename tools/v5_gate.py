"""Judge the v5.0 arms against the gates registered before they ran.

`results/prereg-v5-reasoning48.md` fixes six conditions. Two of them cannot be read off
the rows and are reconstructed here instead of assumed:

* **the mechanism fired** — the rows record `parallel_raw_turns`, so this one is read,
  but a question whose selected memories name no source session cannot fire and is
  excluded rather than counted as a failure;
* **the sentences arrived** — the three failures the offline gate attributed to
  `local_raw` are the reason the arm exists, so the turn each of them needs is looked
  for in the evidence the run actually recovered. An arm that never carried those
  sentences would be a void run, not a negative one, and the prereg says so.

Gate 2 reads the hand attribution in `tools/v5_offline_gate.py`: a question the offline
gate called `reader` had its evidence in front of the answerer already, so losing one
is the mechanism disturbing an answer it was not addressing.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import rows_by_question, write_report  # noqa: E402

from llm_long_term_memory.stats import paired_outcomes  # noqa: E402

BASELINE = REPO / "results/raw/two_stage_v2c.reasoning48-v2c8.jsonl"
STORE = REPO / "stores/v5-reasoning48.db"

# The three the arm was registered for, and the sentence each one needs. Taken from
# `results/analysis/v5-offline-gate.md`, where they are the `local_raw` attributions.
TARGETS = {
    "gpt4_7a0daae1": "received my new tennis racket today",
    "a4996e51": "50 hours per week",
    "58ef2f1c": "valentine's day",
}


def _reader_attributions() -> set[str]:
    path = REPO / "tools/v5_offline_gate.py"
    spec = importlib.util.spec_from_file_location("v5_offline_gate_for_gate", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {qid for qid, (cause, _) in module.AUDIT.items() if cause == "reader"}


def _recovered_text(rows: dict[str, dict]) -> dict[str, str]:
    """Rebuild each question's source-local window from what the run recorded.

    `recover_local_detail` is deterministic given the store, the question and the
    selected memories, and the budget it ran with is in the row. Reconstructing beats
    trusting: the arm's whole claim is that a particular sentence reached the reader.
    """
    from llm_long_term_memory.evaluation.datasets.longmemeval import load
    from llm_long_term_memory.retrieve.fallback import RawFallback
    from llm_long_term_memory.store import SQLiteMemoryStore

    instances = {item.question_id: item for item in load("s", REPO / "data")}
    store = SQLiteMemoryStore(STORE, read_only=True)
    store.initialize()
    try:
        text = {}
        for qid, row in rows.items():
            if qid not in TARGETS:
                continue
            instance = instances[qid]
            selected = [
                memory
                for item in row["notes"]["retrieval"]
                if (memory := store.get(item["memory_id"])) is not None
            ]
            windows = row["notes"].get("parallel_raw_windows") or 0
            budget = row["notes"].get("parallel_raw_turns") or windows
            evidence = RawFallback(store, max_turns=max(windows, budget)).recover_local_detail(
                instance.store_namespace, instance.question, selected
            )
            text[qid] = evidence.render(max_chars=2400).lower() if evidence.used else ""
        return text
    finally:
        store.close()


def analyse(candidate_path: Path) -> dict:
    baseline = rows_by_question(BASELINE)
    candidate = rows_by_question(candidate_path)
    ids = [qid for qid in baseline if qid in candidate]
    if len(ids) != len(baseline):
        raise SystemExit(
            f"the candidate is incomplete: {len(candidate)} of {len(baseline)} questions"
        )

    paired = paired_outcomes(
        {q: bool(baseline[q]["correct"]) for q in ids},
        {q: bool(candidate[q]["correct"]) for q in ids},
        ids,
    )

    # A question whose memories name no source session has nothing source-local to
    # recover, so it is not a failure of the mechanism to have fired on it.
    never_fired = [
        q
        for q in ids
        if not (candidate[q]["notes"].get("parallel_raw_turns") or 0)
        and candidate[q]["notes"].get("parallel_raw_level") not in (None, "none")
    ]
    fired = sum(1 for q in ids if (candidate[q]["notes"].get("parallel_raw_turns") or 0) > 0)

    reader_ids = _reader_attributions()
    reader_losses = [q for q in paired["losses"] if q in reader_ids]

    recovered = _recovered_text(candidate)
    target_status = {
        qid: {
            "needle": needle,
            "arrived": needle in recovered.get(qid, ""),
            "correct": bool(candidate[qid]["correct"]),
            "was_correct": bool(baseline[qid]["correct"]),
        }
        for qid, needle in TARGETS.items()
    }

    def median(values: list[int]) -> float:
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return float(ordered[middle])
        return (ordered[middle - 1] + ordered[middle]) / 2

    base_context = median([baseline[q]["context_tokens"] for q in ids])
    cand_context = median([candidate[q]["context_tokens"] for q in ids])

    gates = {
        "net_positive": paired["net"] > 0,
        "no_reader_attributed_regression": not reader_losses,
        "mechanism_fired": not never_fired,
        "registered_sentences_arrived": all(t["arrived"] for t in target_status.values()),
        "median_context_at_most_2000": cand_context <= 2000,
    }
    return {
        "experiment": candidate_path.stem,
        "class": "development",
        "baseline_rows": str(BASELINE.relative_to(REPO)),
        "candidate_rows": str(candidate_path.relative_to(REPO)),
        "questions": len(ids),
        "baseline_correct": paired["old_correct"],
        "candidate_correct": paired["new_correct"],
        "wins": paired["wins"],
        "losses": paired["losses"],
        "ties": paired["ties"],
        "net": paired["net"],
        "exact_mcnemar_p": paired["exact_mcnemar_p"],
        "mechanism_fired_on": fired,
        "mechanism_should_have_fired_but_did_not": never_fired,
        "reader_attributed_losses": reader_losses,
        "targets": target_status,
        "median_context_baseline": base_context,
        "median_context_candidate": cand_context,
        "gates": gates,
        "decision": "PASS" if all(gates.values()) else "STOP",
    }


def render(result: dict) -> str:
    lines = [
        f"# {result['experiment']}",
        "",
        "> **DEVELOPMENT RESULT.** These 48 questions have been read in full. The baseline",
        "> is the committed v2c run rather than a fresh one, so the arms were measured on",
        "> different days. Directional only; no significance is claimed.",
        "",
        f"Decision: **{result['decision']}**",
        "",
        f"- baseline: **{result['baseline_correct']}/{result['questions']}**",
        f"- candidate: **{result['candidate_correct']}/{result['questions']}**",
        f"- paired: **{len(result['wins'])} wins / {len(result['losses'])} losses / "
        f"{result['ties']} ties**, net **{result['net']:+d}**",
        f"- exact McNemar p = {result['exact_mcnemar_p']:.4f} (shape only — one run per arm)",
        f"- median context: {result['median_context_baseline']:.0f} -> "
        f"{result['median_context_candidate']:.0f} tokens",
        f"- the mechanism recovered turns on "
        f"**{result['mechanism_fired_on']}/{result['questions']}** questions",
        "",
        "## Registered gates",
        "",
    ]
    lines += [
        f"- {'PASS' if passed else 'FAIL'} — `{name}`" for name, passed in result["gates"].items()
    ]
    lines += [
        "",
        "## The three failures this arm was registered for",
        "",
        "| question | sentence | reached the reader | v2c | v5 |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| `{qid}` | {t['needle']} | {'yes' if t['arrived'] else '**no**'} | "
        f"{'correct' if t['was_correct'] else 'wrong'} | "
        f"{'correct' if t['correct'] else 'wrong'} |"
        for qid, t in result["targets"].items()
    ]
    if result["wins"]:
        lines += ["", f"Wins: {', '.join(f'`{q}`' for q in result['wins'])}"]
    if result["losses"]:
        lines += ["", f"Losses: {', '.join(f'`{q}`' for q in result['losses'])}"]
    if result["reader_attributed_losses"]:
        lines += [
            "",
            "Losses on questions the offline gate attributed to the reader — evidence the "
            "answerer already had, so the mechanism disturbed an answer it was not "
            f"addressing: {', '.join(f'`{q}`' for q in result['reader_attributed_losses'])}",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rows", type=Path)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args()
    # Resolved against the repository, so the report records one path whether the
    # caller passed a relative or an absolute one.
    args.rows = args.rows if args.rows.is_absolute() else (REPO / args.rows).resolve()
    stem = args.rows.stem
    json_out = args.json_out or REPO / f"results/analysis/{stem}.json"
    md_out = args.md_out or REPO / f"results/analysis/{stem}.md"
    result = analyse(args.rows)
    print(write_report(result, render, json_out=json_out, md_out=md_out), end="")
    return 0 if result["decision"] == "PASS" else 1


if __name__ == "__main__":
    # `sys.exit`, not a bare call: a gate that exits 0 on STOP is one that anything
    # reading the status rather than the prose walks straight through.
    sys.exit(main())
