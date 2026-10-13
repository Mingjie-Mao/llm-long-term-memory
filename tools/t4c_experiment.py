"""Stage C of research/ROADMAP.md: t4c (t4 + code-side counts, intervals and order).

    python tools/t4c_experiment.py check    # zero calls: request equality + manifests
    python tools/t4c_experiment.py report   # after the live runs: paired comparison

`check` builds both arms exactly as `lltm eval run` does (`cli._build`, configs/fallback
.yaml, the original dev100/heldout100 stores) with a recording client, answers all 200
regression questions, and requires every question `compute_mode` does not select to send
t4's first request byte for byte. It writes the triggered question ids per split to
`results/manifests/t4c-triggered-<split>.json` for the live runs.

Live runs (development evidence; only triggered questions are re-asked, 3 samples):

    lltm eval run two_stage_raw_primary_t4c -c configs/fallback.yaml --store-name dev100 \\
        --questions results/manifests/t4c-triggered-dev100.json --label t4c-v1-dev100-rep1 \\
        --cache stores/cache/t4c-v1.db --sample rep1

`report` pairs each triggered question's 3 new answers with the same question's 3 t4
baseline answers. Untriggered questions are not re-run: their requests are identical, so
their baseline rows stand for both arms, and that is stated rather than resampled.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import read_jsonl  # noqa: E402
from t4_baseline_audit import BASELINE, load_runs  # noqa: E402

from llm_long_term_memory.answer_compute import compute_mode  # noqa: E402

OUT = REPO / "results/analysis/t4c-v1"
MANIFESTS = REPO / "results/manifests"
VARIANT = "two_stage_raw_primary_t4c"
LABEL = "t4c-v1-{split}-rep{n}"


class Recorder:
    """Records requests; answers every call with a plain verdict so no fallback runs."""

    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, **request):
        self.calls.append(request)

        class Completion:
            text = '{"status": "answer", "answer": "ok"}'
            input_tokens = output_tokens = 1
            api_latency_ms = 0.0

        return Completion()


def check() -> dict:
    from llm_long_term_memory import cli
    from llm_long_term_memory.conversation import AnswerRequest

    questions = {
        q["question_id"]: q
        for q in json.loads((REPO / "data/longmemeval_s_cleaned.json").read_text(encoding="utf-8"))
    }
    result = {"splits": {}}
    for split, spec in BASELINE.items():
        ids = json.loads((MANIFESTS / f"{split}.json").read_text(encoding="utf-8"))["question_ids"]
        arms = {}
        for variant in ("two_stage_raw_primary_t4", VARIANT):
            recorder = Recorder()
            _, _, runner, _, _ = cli._build(
                variant,
                "configs/fallback.yaml",
                spec["store"],
                client_override=recorder,
                read_only_store=True,
            )
            arms[variant] = (runner, recorder)
        triggered, identical, differs = [], 0, []
        modes = Counter()
        for qid in ids:
            q = questions[qid]
            request = AnswerRequest(
                question=q["question"], asked_on=q["question_date"], user_id=qid
            )
            first = {}
            for variant, (runner, recorder) in arms.items():
                recorder.calls.clear()
                runner.answer_request(request)
                first[variant] = recorder.calls[0]
            mode = compute_mode(q["question"])
            modes[mode or "untouched"] += 1
            a, b = first["two_stage_raw_primary_t4"], first[VARIANT]
            if mode:
                triggered.append(qid)
                if a["prompt"] != b["prompt"]:
                    differs.append(qid)  # only system, schema and limit may change
            elif a == b:
                identical += 1
            else:
                differs.append(qid)
        manifest = {
            "name": f"t4c-triggered-{split}",
            "variant": "s",
            "seed": 0,
            "n": len(triggered),
            "exposure": "regression",
            "note": (
                f"Questions of the {split} regression set that compute_mode selects "
                "(stage C, research/ROADMAP.md). Derived by tools/t4c_experiment.py check; "
                "exposed development questions."
            ),
            "question_ids": triggered,
        }
        (MANIFESTS / f"t4c-triggered-{split}.json").write_text(
            json.dumps(manifest, indent=1) + "\n", encoding="utf-8"
        )
        result["splits"][split] = {
            "questions": len(ids),
            "modes": dict(modes),
            "untouched_identical": identical,
            "violations": differs,
        }
    return result


def _rows(split: str, n: int) -> dict[str, dict]:
    path = REPO / "results/raw" / f"{VARIANT}.{LABEL.format(split=split, n=n)}.jsonl"
    rows = read_jsonl(path) if path.exists() else []
    return {r["question_id"]: r for r in rows}


def report() -> dict:
    """Pair each triggered question's samples with the t4 baseline's, for two t4c arms:
    v1 as run (overrides on), and record-only (overridden rows replaced by their
    cache-replayed, regraded no-override answer)."""
    questions = {
        q["question_id"]: q
        for q in json.loads((REPO / "data/longmemeval_s_cleaned.json").read_text(encoding="utf-8"))
    }
    notes = json.loads((REPO / "research/t4-first-loss-notes.json").read_text(encoding="utf-8"))
    out = {"splits": {}, "questions": []}
    for split in BASELINE:
        ids = json.loads((MANIFESTS / f"t4c-triggered-{split}.json").read_text(encoding="utf-8"))[
            "question_ids"
        ]
        base = load_runs(split)
        new = [_rows(split, n) for n in (1, 2, 3)]
        regraded = []
        for n in (1, 2, 3):
            path = REPO / "results/raw/t4c-v1-noov" / f"{split}-rep{n}.jsonl"
            regraded.append(
                {r["question_id"]: r for r in read_jsonl(path)} if path.exists() else {}
            )
        complete = [qid for qid in ids if all(qid in run for run in new)]
        s = {"triggered": len(ids), "complete": len(complete), "base": 0, "new": 0, "noov": 0}
        s["noov_missing"] = 0
        for qid in complete:
            b = sum(bool(run[qid]["correct"] in (True, "True")) for run in base)
            c = sum(bool(run[qid]["correct"]) for run in new)
            noov, overrides = 0, 0
            for run, again in zip(new, regraded, strict=True):
                if (run[qid]["notes"].get("synthesis_computation") or {}).get("overrode"):
                    overrides += 1
                    if qid in again:
                        noov += bool(again[qid]["correct"])
                    else:
                        s["noov_missing"] += 1
                else:
                    noov += bool(run[qid]["correct"])
            s["base"] += b
            s["new"] += c
            s["noov"] += noov
            out["questions"].append(
                {
                    "question_id": qid,
                    "split": split,
                    "type": questions[qid]["question_type"],
                    "mode": compute_mode(questions[qid]["question"]),
                    "base_correct_of_3": b,
                    "new_correct_of_3": c,
                    "noov_correct_of_3": noov,
                    "overrides": overrides,
                    "ambiguous": bool(notes.get(qid, {}).get("ambiguous")),
                    "baseline_failure": notes.get(qid, {}).get("failure"),
                }
            )
        out["splits"][split] = s
    return out


class CacheOnly:
    """A provider that is never reached: every answer must come from the response cache."""

    def generate(self, **request):
        raise RuntimeError("cache miss: the replay would have called the answerer")


# Regrades of overridden rows only: partial, so kept out of results/raw/*.jsonl, which
# tools/resolution.py reads as complete repeat runs.
NOOV_DIR = "results/raw/t4c-v1-noov"


def replay_no_override(only_split: str | None = None, only_rep: int | None = None) -> dict:
    """Re-derive every overridden v1 row with overrides off, and grade only those.

    The answerer is never called: the runner is built on the response cache of the v1
    runs with a provider that raises on any miss. Only the judge is called, through its
    own cache, so an interrupted replay resumes without regrading.
    """
    from llm_long_term_memory import cli
    from llm_long_term_memory.conversation import AnswerRequest
    from llm_long_term_memory.llm.cache import CachingClient

    questions = {
        q["question_id"]: q
        for q in json.loads((REPO / "data/longmemeval_s_cleaned.json").read_text(encoding="utf-8"))
    }
    summary = {}
    for split, spec in BASELINE.items():
        for n in (1, 2, 3):
            if (only_split and split != only_split) or (only_rep and n != only_rep):
                continue
            rows = _rows(split, n)
            out = REPO / "results/raw/t4c-v1-noov" / f"{split}-rep{n}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            done = {r["question_id"] for r in read_jsonl(out)} if out.exists() else set()
            cached = CachingClient(
                CacheOnly(),
                REPO / "stores/cache/t4c-v1.db",
                sample=f"rep{n}",
                scope={"store": spec["store"], "config": "configs/fallback.yaml"},
            )
            _, _, runner, _, _ = cli._build(
                VARIANT,
                "configs/fallback.yaml",
                spec["store"],
                client_override=cached,
                read_only_store=True,
            )
            runner.compute_override = False
            _, _, _, judge, _ = cli._build(
                "full_context",
                "configs/fallback.yaml",
                response_cache=str(REPO / "stores/cache/t4c-v1-noov-judge.db"),
                sample=f"rep{n}",
            )
            regraded = 0
            with out.open("a", encoding="utf-8") as sink:
                for qid, row in rows.items():
                    if qid in done:
                        continue
                    if not (row["notes"].get("synthesis_computation") or {}).get("overrode"):
                        continue
                    q = questions[qid]
                    answer = runner.answer_request(
                        AnswerRequest(
                            question=q["question"], asked_on=q["question_date"], user_id=qid
                        )
                    )
                    verdict = judge.grade(
                        question=q["question"],
                        gold=str(q["answer"]),
                        hypothesis=answer.text,
                        is_abstention=qid.endswith("_abs"),
                        question_type=q["question_type"],
                    )
                    sink.write(
                        json.dumps(
                            {
                                "question_id": qid,
                                "hypothesis": answer.text,
                                "v1_hypothesis": row["hypothesis"],
                                "correct": verdict.correct,
                                "judge_reason": verdict.reason,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    sink.flush()
                    regraded += 1
            summary[f"{split}-rep{n}"] = regraded
    return summary


V2 = "two_stage_raw_primary_t4c2"
V2_OUT = REPO / "results/analysis/t4c-v2-replay"


def check_v2() -> dict:
    """Zero calls: t4c2 sends v1's count request where it triggers, t4's elsewhere."""
    from llm_long_term_memory import cli
    from llm_long_term_memory.answer_compute import count_mode_v2
    from llm_long_term_memory.conversation import AnswerRequest

    questions = {
        q["question_id"]: q
        for q in json.loads((REPO / "data/longmemeval_s_cleaned.json").read_text(encoding="utf-8"))
    }
    result = {}
    for split, spec in BASELINE.items():
        ids = json.loads((MANIFESTS / f"{split}.json").read_text(encoding="utf-8"))["question_ids"]
        arms = {}
        for variant in ("two_stage_raw_primary_t4", VARIANT, V2):
            recorder = Recorder()
            _, _, runner, _, _ = cli._build(
                variant,
                "configs/fallback.yaml",
                spec["store"],
                client_override=recorder,
                read_only_store=True,
            )
            arms[variant] = (runner, recorder)
        triggered, violations = [], []
        for qid in ids:
            q = questions[qid]
            request = AnswerRequest(
                question=q["question"], asked_on=q["question_date"], user_id=qid
            )
            first = {}
            for variant, (runner, recorder) in arms.items():
                recorder.calls.clear()
                runner.answer_request(request)
                first[variant] = recorder.calls[0]
            reference = VARIANT if count_mode_v2(q["question"]) else "two_stage_raw_primary_t4"
            if count_mode_v2(q["question"]):
                triggered.append(qid)
            if first[V2] != first[reference]:
                violations.append(qid)
        result[split] = {"triggered": triggered, "violations": violations}
    return result


def replay_v2(check: dict) -> dict:
    """Zero calls. Triggered: v1's samples with the record-only answer (v1's grade where
    nothing was overridden, the regrade where something was). Elsewhere t4's request, so
    t4's samples stand for both arms."""
    notes = json.loads((REPO / "research/t4-first-loss-notes.json").read_text(encoding="utf-8"))
    out = {"splits": {}, "questions": []}
    for split in BASELINE:
        base = load_runs(split)
        v1 = [_rows(split, n) for n in (1, 2, 3)]
        regraded = []
        for n in (1, 2, 3):
            path = REPO / "results/raw/t4c-v1-noov" / f"{split}-rep{n}.jsonl"
            regraded.append({r["question_id"]: r for r in read_jsonl(path)})
        s = {
            "triggered": 0,
            "base": 0,
            "v2": 0,
            "base_unamb": 0,
            "v2_unamb": 0,
            "regression_rows": 0,
        }
        for qid in check[split]["triggered"]:
            b = sum(bool(run[qid]["correct"] in (True, "True")) for run in base)
            r = 0
            for row, again in zip(v1, regraded, strict=True):
                if (row[qid]["notes"].get("synthesis_computation") or {}).get("overrode"):
                    r += bool(again[qid]["correct"])
                else:
                    r += bool(row[qid]["correct"])
            amb = bool(notes.get(qid, {}).get("ambiguous"))
            s["triggered"] += 1
            s["base"] += b
            s["v2"] += r
            if not amb:
                s["base_unamb"] += b
                s["v2_unamb"] += r
            out["questions"].append(
                {"question_id": qid, "split": split, "t4": b, "t4c2": r, "ambiguous": amb}
            )
        s["regression_rows"] = 3 * len(base[0])
        out["splits"][split] = s
    return out


def render_v2(check: dict, rep: dict) -> str:
    lines = [
        "# t4c v2: plain counts, computation recorded but never substituted (zero calls)",
        "",
        "> Generated by `tools/t4c_experiment.py v2`. Development evidence on exposed",
        "> questions. A replay, not new samples: every t4c2 request is byte-identical to a",
        "> request already sampled (t4c v1's count request where it triggers, t4's elsewhere),",
        "> so those samples are reused. The trigger was narrowed after reading v1's results.",
        "",
        "| split | triggered | violations | t4 (triggered) | t4c2 (triggered) |"
        " unambiguous t4 → t4c2 |",
        "|---|---:|---|---:|---:|---|",
    ]
    for split, s in rep["splits"].items():
        lines.append(
            f"| {split} | {s['triggered']} | {check[split]['violations'] or 'none'} |"
            f" {s['base']}/{3 * s['triggered']} | {s['v2']}/{3 * s['triggered']} |"
            f" {s['base_unamb']} → {s['v2_unamb']} |"
        )
    lines += ["", "| question | split | t4 | t4c2 | ambiguous |", "|---|---|---:|---:|---|"]
    for q in rep["questions"]:
        if q["t4"] != q["t4c2"]:
            lines.append(
                f"| `{q['question_id']}` | {q['split']} | {q['t4']} | {q['t4c2']} |"
                f" {'yes' if q['ambiguous'] else ''} |"
            )
    return "\n".join(lines) + "\n"


def render(check_result: dict | None, rep: dict | None) -> str:
    lines = [
        "# t4c: code-side counts, intervals and order (stage C, v1)",
        "",
        "> Generated by `tools/t4c_experiment.py`. Development evidence on exposed",
        "> questions; only the questions `compute_mode` selects were re-asked (3 samples).",
        "> Record-only reuses v1's cached answers with overrides off and regrades only the",
        "> rows an override had changed; it is a replay, not new samples.",
        "",
    ]
    if check_result:
        lines += [
            "## Zero-call request check",
            "",
            "| split | questions | modes | untouched = t4 | violations |",
            "|---|---:|---|---:|---|",
        ]
        for split, s in check_result["splits"].items():
            untouched = s["modes"].get("untouched", 0)
            lines.append(
                f"| {split} | {s['questions']} | {s['modes']} |"
                f" {s['untouched_identical']}/{untouched} | {s['violations'] or 'none'} |"
            )
        lines.append("")
    if not rep:
        return "\n".join(lines) + "\n"
    lines += [
        "## Triggered questions, correct of 3 samples per question",
        "",
        "| split | triggered | t4 | t4c v1 (override) | t4c record-only | regrades missing |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, s in rep["splits"].items():
        lines.append(
            f"| {split} | {s['triggered']} | {s['base']} | {s['new']} | {s['noov']} |"
            f" {s['noov_missing']} |"
        )
    by_mode: dict[str, list[int]] = {}
    for q in rep["questions"]:
        if q["ambiguous"]:
            continue
        m = by_mode.setdefault(q["mode"], [0, 0, 0, 0])
        m[0] += 1
        m[1] += q["base_correct_of_3"]
        m[2] += q["new_correct_of_3"]
        m[3] += q["noov_correct_of_3"]
    lines += [
        "",
        "| mode (unambiguous) | questions | t4 | t4c v1 (override) | t4c record-only |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, (n, b, c, r) in sorted(by_mode.items()):
        lines.append(f"| {mode} | {n} | {b}/{3 * n} | {c}/{3 * n} | {r}/{3 * n} |")
    moved = [
        q
        for q in rep["questions"]
        if len({q["base_correct_of_3"], q["new_correct_of_3"], q["noov_correct_of_3"]}) > 1
    ]
    lines += [
        "",
        "| question | mode | t4 | v1 override | record-only | overrides | baseline failure |"
        " ambiguous |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for q in sorted(moved, key=lambda q: q["noov_correct_of_3"] - q["base_correct_of_3"]):
        lines.append(
            f"| `{q['question_id']}` | {q['mode']} | {q['base_correct_of_3']} |"
            f" {q['new_correct_of_3']} | {q['noov_correct_of_3']} | {q['overrides']} |"
            f" {q['baseline_failure'] or ''} | {'yes' if q['ambiguous'] else ''} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> None:
    check_path = OUT.parent / f"{OUT.name}.check.json"
    if argv[:1] == ["check"]:
        result = check()
        check_path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        print(render(result, None))
        if any(s["violations"] for s in result["splits"].values()):
            raise SystemExit("untouched questions changed their request")
    elif argv[:1] == ["replay-no-override"]:
        split, rep = (argv[1], int(argv[2])) if len(argv) == 3 else (None, None)
        print(json.dumps(replay_no_override(split, rep), indent=1))
    elif argv[:1] == ["v2"]:
        v2_check = check_v2()
        rep = replay_v2(v2_check)
        V2_OUT.with_suffix(".json").write_text(
            json.dumps({"check": v2_check, "replay": rep}, indent=1) + "\n", encoding="utf-8"
        )
        V2_OUT.with_suffix(".md").write_text(render_v2(v2_check, rep), encoding="utf-8")
        print(render_v2(v2_check, rep))
        if any(c["violations"] for c in v2_check.values()):
            raise SystemExit("t4c2 sent a request that was never sampled")
    elif argv[:1] == ["report"]:
        rep = report()
        check_result = json.loads(check_path.read_text(encoding="utf-8"))
        OUT.with_suffix(".json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
        OUT.with_suffix(".md").write_text(render(check_result, rep), encoding="utf-8")
        print(render(check_result, rep))
    else:
        raise SystemExit("usage: t4c_experiment.py check | replay-no-override | report | v2")


if __name__ == "__main__":
    main(sys.argv[1:])
