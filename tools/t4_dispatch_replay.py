"""Strict zero-call baseline/known-repair replay acceptance for scoped dispatch."""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
from analysis_io import rows_by_question, sha256_file  # noqa: E402
from run_grounded_paired import combine  # noqa: E402
from run_recursive_summary_cost import instances_for  # noqa: E402
from t4_event_count_acceptance import GAP_GOLD  # noqa: E402
from t4_regression_resume import freeze as parent_freeze  # noqa: E402
from t4_regression_resume import verify as verify_parent  # noqa: E402
from t4_savings_regression import immutable_json  # noqa: E402
from t4_savings_replay import (  # noqa: E402
    ReplayMismatch,
    SavedReplyClient,
    cached_grade,
    grade_request_sha,
    row_sha,
)
from t4_scoped_dispatch import VERSION, ScopedSavingsRepairRunner, applicable_repair  # noqa: E402
from t4_scoped_selection_acceptance import CapturingClient  # noqa: E402
from t4_selection_preflight import OfflineClient  # noqa: E402

STEM = "t4-scoped-dispatch-replay-v1"
ANALYSIS = REPO / "results/analysis"
INVENTORY = ANALYSIS / f"{STEM}.execution.json"
GATE = ANALYSIS / f"{STEM}.gate.json"
DEV = "t4-scoped-regression-dev100-v3"
BASE_TRAIN = "t4-selection-train21-v1"
REPAIR_TRAIN = "t4-scoped-selection-train21-v4"


def freeze():
    parent = parent_freeze()
    verify_parent(parent)
    files = dict(parent["files"])
    paths = [
        Path(__file__),
        REPO / "tools/t4_scoped_dispatch.py",
        REPO / "tests/test_t4_scoped_dispatch.py",
        REPO / "tests/test_t4_dispatch_replay.py",
        REPO / f"results/prereg-{STEM}.md",
        ANALYSIS / f"{DEV}.gate.json",
        ANALYSIS / "t4-scoped-dev100-postmortem-v1.json",
        REPO / "stores/train150.db",
        REPO / "stores/train150-index.npy",
        REPO / "stores/train150-index.ids.json",
    ]
    for stem in (BASE_TRAIN, REPAIR_TRAIN):
        paths += [ANALYSIS / f"{stem}.execution.json"]
        paths += [
            REPO / f"results/raw/{stem}.{s}"
            for s in ("readers.jsonl", "grades.jsonl", "gap-grades.jsonl")
        ]
    for rep in (1, 2, 3):
        paths += [
            REPO / f"results/raw/{DEV}-control-rep{rep}.{s}"
            for s in ("readers.jsonl", "grades.jsonl")
        ]
    files.update({str(p.relative_to(REPO)): sha256_file(p) for p in paths})
    snapshot = {
        "files": files,
        "parent": parent,
        "candidate_prompt_version": VERSION,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "python": platform.python_version(),
        "provider_calls": 0,
        "experiment_class": "exposed-development-strict-request-replay",
    }
    immutable_json(INVENTORY, snapshot)
    verify(snapshot)
    return snapshot, sha256_file(INVENTORY)


def verify(snapshot):
    verify_parent(snapshot["parent"])
    for rel, digest in snapshot["files"].items():
        if sha256_file(REPO / rel) != digest:
            raise ValueError(f"dispatch replay source changed: {rel}")


def source(stem):
    base = REPO / f"results/raw/{stem}"
    readers, grades = [
        rows_by_question(base.with_suffix("." + s)) for s in ("readers.jsonl", "grades.jsonl")
    ]
    xp = base.with_suffix(".gap-grades.jsonl")
    return readers, grades, rows_by_question(xp) if xp.exists() else {}


def replay_one(runner, instance, saved, grade, identity, gap=None):
    from llm_long_term_memory.conversation import AnswerRequest

    replay = SavedReplyClient(OfflineClient(), saved)
    runner.client = CapturingClient(replay)
    request = AnswerRequest(instance.question, instance.question_date, instance.store_namespace)
    try:
        answer = runner.answer_request(request)
        replay.verify_output(answer, saved)
        if answer.context_tokens != saved["answer"]["context_tokens"]:
            raise ReplayMismatch("actual supplied context accounting changed")
        current = answer.notes.get("selection_repair") or {}
        previous = saved["answer"]["notes"].get("selection_repair") or {}
        if current.get("calculation") and row_sha(current) != row_sha(previous):
            raise ReplayMismatch("active source/calculation derivation changed")
        proof = {
            "source_reader_sha256": row_sha(saved),
            "judge_prompt_version": grade["judge_prompt_version"],
            "judge_request_sha256": {"original": grade_request_sha(instance, answer.text)},
        }
        if gap:
            proof["judge_request_sha256"]["gap"] = grade_request_sha(
                replace(instance, answer=GAP_GOLD[instance.question_id]), answer.text
            )
        answer.notes["exact_request_replay"] = proof
        row = {
            "question_id": instance.question_id,
            "cohort": "dispatch-replay",
            "inventory_sha256": identity,
            "answer_prompt_version": VERSION,
            "answer": asdict(answer),
        }
        original = cached_grade(instance, row, saved, grade, identity)
        faithful = (
            cached_grade(
                replace(instance, answer=GAP_GOLD[instance.question_id]),
                row,
                saved,
                gap,
                identity,
                "gap",
            )
            if gap
            else None
        )
        return {
            "question_id": instance.question_id,
            "matched": True,
            "source_reader_sha256": row_sha(saved),
            "source_grade_sha256": row_sha(grade),
            "source_gap_grade_sha256": row_sha(gap) if gap else None,
            "request_sha256": [row_sha(c["request"]) for c in replay.receipts],
            "generations_replayed": replay.position,
            "answer_sha256": row_sha(answer.text),
            "historical_correct": original["verdict"]["correct"],
            "historical_faithful_gap": faithful["verdict"]["correct"] if faithful else None,
            "context_tokens": answer.context_tokens,
            "active_mechanism": (current.get("calculation") or {}).get("mechanism"),
        }
    except ReplayMismatch as exc:
        return {"question_id": instance.question_id, "matched": False, "error": str(exc)}


def decide(dev, train):
    per_rep = [{r["question_id"] for r in dev if r.get("rep") == rep} for rep in (1, 2, 3)]
    complete = (
        len(dev) == 300
        and all(len(ids) == 100 and ids == per_rep[0] for ids in per_rep)
        and len({(r["question_id"], r.get("rep")) for r in dev}) == 300
        and len(train) == len({r["question_id"] for r in train}) == 21
    )
    matched = complete and all(r.get("matched") is True for r in [*dev, *train])
    supported = [r for r in train if r["question_id"] not in GAP_GOLD]
    gaps = [r for r in train if r["question_id"] in GAP_GOLD]
    supported_ok = (
        matched and len(supported) == 18 and all(r["historical_correct"] is True for r in supported)
    )
    gaps_ok = matched and len(gaps) == 3 and all(r["historical_faithful_gap"] is True for r in gaps)
    median = statistics.median(r["context_tokens"] for r in train) if matched else None
    return {
        "complete": complete,
        "all_requests_outputs_and_bindings_match": matched,
        "supported_known_cases": {
            "correct": sum(r.get("historical_correct") is True for r in supported),
            "total": 18,
        },
        "faithful_gaps": {
            "correct": sum(r.get("historical_faithful_gap") is True for r in gaps),
            "total": 3,
        },
        "median_train_context": median,
        "pass": bool(matched and supported_ok and gaps_ok and median <= 6500),
    }


def main():
    from llm_long_term_memory import cli
    from llm_long_term_memory.conversation import AnswerRequest

    snapshot, identity = freeze()
    dev, train = [], []
    for split in ("dev100", "train150"):
        _, _, base, _, _ = cli._build(
            "two_stage_raw_primary_t4",
            "configs/fallback.yaml",
            split,
            client_override=OfflineClient(),
            read_only_store=True,
        )
        runner = ScopedSavingsRepairRunner.from_t4(base)
        try:
            if split == "dev100":
                for rep in (1, 2, 3):
                    readers, grades, _ = source(f"{DEV}-control-rep{rep}")
                    for i in instances_for(split):
                        combine(
                            i,
                            readers[i.question_id],
                            grades[i.question_id],
                            readers[i.question_id]["inventory_sha256"],
                            "control",
                        )
                        r = replay_one(
                            runner, i, readers[i.question_id], grades[i.question_id], identity
                        )
                        r.update(rep=rep, source=f"{DEV}-control-rep{rep}")
                        dev.append(r)
            else:
                sources = {s: source(s) for s in (BASE_TRAIN, REPAIR_TRAIN)}
                for i in instances_for(split):
                    active = applicable_repair(
                        runner, AnswerRequest(i.question, i.question_date, i.store_namespace)
                    )
                    stem = REPAIR_TRAIN if active else BASE_TRAIN
                    readers, grades, gaps = sources[stem]
                    r = replay_one(
                        runner,
                        i,
                        readers[i.question_id],
                        grades[i.question_id],
                        identity,
                        gaps.get(i.question_id),
                    )
                    r.update(source=stem, applicability=active)
                    train.append(r)
        finally:
            base.store.close()
    verify(snapshot)
    result = {
        "provider_calls": 0,
        "fresh_reader_rows": 0,
        "fresh_grade_rows": 0,
        "experiment_class": "exposed-development-structural-and-historical-replay-acceptance",
        "inventory_sha256": identity,
        "candidate_prompt_version": VERSION,
        "dev100_replayed_rows": len(dev),
        "train21_replayed_rows": len(train),
        "decision": decide(dev, train),
        "pass": decide(dev, train)["pass"],
        "dev": dev,
        "train": train,
        "limitations": "Strict logical-request/output preservation, not new model samples "
        "or a fresh population accuracy measurement. The previous fresh dev gate remains failed.",
    }
    immutable_json(GATE, result)
    print(
        json.dumps(
            {
                "provider_calls": 0,
                "dev100_replays": len(dev),
                "train21_replays": len(train),
                "decision": result["decision"],
            }
        )
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
