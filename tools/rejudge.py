"""Re-grade answers already bought, so judge noise can be measured without buying answers.

Pass B of the registered noise design (`results/prereg-beam-v1.md`). The answers are fixed;
only the judge runs again. That isolates the evaluator's own variance from the answerer's,
and it costs one judge call per question instead of a whole answering pass — 440 requests
against 1,030 on BEAM's development half.

Why it has to be isolated: at 2-3 points of difference, "is this a real improvement or the
evaluator moving?" is the whole question, and a single end-to-end repeat cannot answer it.
`v4.0-flat` and `v4.0-flat2` are the same configuration on the same store with
byte-identical retrieval, and 38 of 142 verdicts differed between them — but that pair
mixes answerer sampling with judge sampling and cannot say which produced the 38.

Every record carries what a later reader needs to tell one run from another: the answer it
graded, the judge's model and prompt version, a hash of the exact prompt text, temperature,
seed where the provider exposes one, the raw reply, and the parsed per-item grades. Scores
are recomputed from the grades offline, so a change to aggregation never costs a call.

    python3 tools/rejudge.py --rows <answers.jsonl> --pass-label b1 --dry-run
    python3 tools/rejudge.py --rows <answers.jsonl> --pass-label b1 --execute

`--execute` refuses to start unless `BEAM_PREREG` names the registration, and it never
writes into the file it reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

PREREG = "results/prereg-beam-v1.md"


def prompt_fingerprint(question: str, rubric, hypothesis: str, ordering: bool) -> str:
    from llm_long_term_memory.evaluation.beam_judge import render_prompt

    text = render_prompt(question, rubric, hypothesis, ordering)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def answer_id(row: dict) -> str:
    """Identifies the answer, not the question: the same question answered twice must not
    look like one answer graded twice."""
    digest = hashlib.sha256((row.get("hypothesis") or "").encode("utf-8")).hexdigest()[:16]
    return f"{row['question_id']}@{digest}"


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                seen.add(json.loads(line)["answer_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


_PASS = ""
"""The pass label, read by the dry-run judge so repeated passes differ."""


def build_judge(model: str, *, execute: bool):
    """The real judge, or a fake one that varies its grades a little.

    A dry-run judge that always returns the same grades rehearses the plumbing and
    measures a variance of exactly zero, which is the one answer that cannot be right.
    So the fake perturbs its threshold by the pass label, and the rehearsal can show the
    tool detecting disagreement rather than only running.
    """
    from llm_long_term_memory.evaluation.beam_judge import BeamRubricJudge

    if execute:
        from llm_long_term_memory.config import Settings
        from llm_long_term_memory.llm import Limits, QuotaManager, UsageTracker
        from llm_long_term_memory.llm.client import GeminiClient

        settings = Settings()
        quota = QuotaManager(
            state_dir=settings.store_dir / "quota", default=Limits(rpm=10, tpm=250_000, rpd=500)
        )
        usage = UsageTracker()
        client = GeminiClient(settings.require_api_key(), quota=quota, usage=usage)
        return BeamRubricJudge(client, model=model), usage

    import beam_dry_run

    # The boundaries move with the pass label, so a rehearsed second pass disagrees with
    # the first the way a real judge does, and the variance arithmetic is exercised
    # rather than merely run.
    shift = (int(hashlib.sha256(_PASS.encode("utf-8")).hexdigest()[:4], 16) % 7) / 100
    fake = beam_dry_run.FakeProvider(
        full_credit_at=0.5 + shift - 0.03, half_credit_at=0.2 + shift - 0.03
    )
    return BeamRubricJudge(fake, model=model), None


def summarise(paths: list[Path]) -> dict:
    """How far two judging passes of the same answers disagree.

    Discordance is counted on the binary verdict, because that is what a paired test moves
    on; the mean absolute difference of the continuous score is reported beside it, because
    the primary metric is continuous and a verdict that never flips can still drift.
    """
    passes: dict[str, dict[str, dict]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if "refused" in record:
                continue
            passes.setdefault(record["pass"], {})[record["answer_id"]] = record
    names = sorted(passes)
    if len(names) < 2:
        return {"passes": names, "note": "two passes are needed to measure disagreement"}
    left, right = passes[names[0]], passes[names[1]]
    shared = sorted(set(left) & set(right))
    flips = [a for a in shared if left[a]["correct"] != right[a]["correct"]]
    drift = [abs(left[a]["score"] - right[a]["score"]) for a in shared]
    by_ability: dict[str, dict] = {}
    for answer in shared:
        ability = left[answer]["question_type"]
        row = by_ability.setdefault(ability, {"n": 0, "flips": 0})
        row["n"] += 1
        row["flips"] += int(left[answer]["correct"] != right[answer]["correct"])
    return {
        "passes": names,
        "answers_compared": len(shared),
        "verdict_discordance": len(flips) / len(shared) if shared else None,
        "mean_absolute_score_difference": sum(drift) / len(drift) if drift else None,
        "max_absolute_score_difference": max(drift) if drift else None,
        "by_ability": {
            ability: {**row, "discordance": row["flips"] / row["n"]}
            for ability, row in sorted(by_ability.items())
        },
        "reading": (
            "This is the judge alone: the answers were fixed and only the grading ran again. "
            "Any candidate difference smaller than this is the evaluator, not the system."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", help="a finished run's answers; read only")
    parser.add_argument("--pass-label", help="names this judging pass, e.g. b1")
    parser.add_argument("--half", default="dev", choices=("dev", "test"))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model", default=None, help="judge model; defaults to the config's")
    parser.add_argument("--config", default="configs/v2.yaml")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--execute", action="store_true", help="spend quota; needs BEAM_PREREG")
    parser.add_argument(
        "--summarise",
        nargs="*",
        help="re-judging files to compare instead of grading; two passes measure the judge",
    )
    args = parser.parse_args()

    if args.summarise is not None:
        payload = summarise([Path(p) for p in args.summarise])
        print(json.dumps(payload, indent=2))
        return 0

    if args.execute and os.environ.get("BEAM_PREREG") != PREREG:
        print(f"STOP: --execute needs BEAM_PREREG={PREREG}")
        return 2

    from llm_long_term_memory.config import ExperimentConfig
    from llm_long_term_memory.evaluation.beam_judge import (
        BEAM_JUDGE_PROMPT_VERSION,
        InvalidJudgement,
    )
    from llm_long_term_memory.evaluation.beam_report import load_rows
    from llm_long_term_memory.evaluation.datasets import beam

    if not args.rows or not args.pass_label:
        print("STOP: --rows and --pass-label are required unless --summarise is given")
        return 2
    rows_path = Path(args.rows)
    model = args.model or ExperimentConfig.from_yaml(args.config).models.judge
    out = Path(args.out) if args.out else rows_path.with_suffix(f".rejudge-{args.pass_label}.jsonl")
    if out.resolve() == rows_path.resolve():
        print("STOP: the re-judging pass may not overwrite the answers it reads")
        return 2

    rows = load_rows(rows_path)
    instances = {i.question_id: i for i in beam.load(args.half, Path(args.data_dir))}
    global _PASS
    _PASS = args.pass_label
    judge, usage = build_judge(model, execute=args.execute)
    already = done_ids(out)
    todo = [row for row in rows if answer_id(row) not in already]
    print(
        f"{len(rows)} answer(s), {len(already)} already graded, {len(todo)} to grade\n"
        f"judge={model} version={BEAM_JUDGE_PROMPT_VERSION} pass={args.pass_label} "
        f"{'EXECUTING' if args.execute else 'dry run, zero provider calls'}\n-> {out}"
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    refused = 0
    with out.open("a", encoding="utf-8") as sink:
        for row in todo:
            instance = instances.get(row["question_id"])
            if instance is None or not instance.rubric:
                continue
            ordering = row["question_type"] == "event_ordering"
            try:
                verdict = judge.grade(
                    question=instance.question,
                    gold=instance.answer,
                    hypothesis=row.get("hypothesis") or "",
                    is_abstention=instance.is_abstention,
                    question_type=row["question_type"],
                    rubric=instance.rubric,
                )
            except InvalidJudgement as exc:
                # Recorded, not repaired, and not retried silently: an off-rubric reply is
                # a property of this pass and averaging over it would hide the variance
                # this run exists to measure.
                refused += 1
                sink.write(
                    json.dumps(
                        {
                            "answer_id": answer_id(row),
                            "question_id": row["question_id"],
                            "pass": args.pass_label,
                            "refused": str(exc),
                        }
                    )
                    + "\n"
                )
                sink.flush()
                continue
            sink.write(
                json.dumps(
                    {
                        "answer_id": answer_id(row),
                        "question_id": row["question_id"],
                        "question_type": row["question_type"],
                        "pass": args.pass_label,
                        "judge_model": model,
                        "judge_prompt_version": BEAM_JUDGE_PROMPT_VERSION,
                        "prompt_sha256_16": prompt_fingerprint(
                            instance.question,
                            instance.rubric,
                            row.get("hypothesis") or "",
                            ordering,
                        ),
                        "temperature": args.temperature,
                        "seed": args.seed,
                        "raw_judge_output": verdict.details,
                        "grades": verdict.details.get("grades"),
                        "score": verdict.score,
                        "correct": verdict.correct,
                        "original_score": ((row.get("notes") or {}).get("judge") or {}).get(
                            "score"
                        ),
                    }
                )
                + "\n"
            )
            sink.flush()
    if usage is not None:
        usage.save(out.with_suffix(".usage.json"), merge=True)
    print(f"done; {refused} reply/replies refused as off-rubric")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
