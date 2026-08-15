"""Evaluation driver.

Results are appended to JSONL as each question completes, and a resumed run skips
question_ids already present. This is not defensive programming for its own sake:
D5/D6 establish that a run can outlive its daily quota, so "stop and continue
tomorrow" is the normal path, not the error path.

Hitting the daily cap therefore returns a partial `RunReport` rather than raising.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from llm_long_term_memory.evaluation.datasets.longmemeval import Instance
from llm_long_term_memory.evaluation.judge import JUDGE_PROMPT_VERSION, Judge
from llm_long_term_memory.evaluation.runners.base import ANSWER_PROMPT_VERSION, Runner
from llm_long_term_memory.llm.client import DailyQuotaExhausted
from llm_long_term_memory.llm.usage import UsageTracker
from llm_long_term_memory.locking import AlreadyRunning, exclusive


@dataclass(slots=True)
class QuestionResult:
    question_id: str
    question_type: str
    is_abstention: bool
    correct: bool
    hypothesis: str
    gold: str
    judge_reason: str
    context_tokens: int
    prompt_tokens: int
    output_tokens: int
    latency_ms: float
    evidence_recalled: bool | None = None
    source_session_recalled: bool | None = None
    # Provenance for the number itself. `None` means the row predates version
    # stamping (before 2026-08-14) — which is information, not a gap to fill in.
    answer_prompt_version: str | None = None
    judge_prompt_version: str | None = None
    extractor_version: str | None = None
    store_fingerprint: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TypeBreakdown:
    question_type: str
    n: int
    correct: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0


@dataclass
class RunReport:
    variant: str
    results: list[QuestionResult] = field(default_factory=list)
    completed: bool = True
    stopped_reason: str | None = None

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def accuracy(self) -> float:
        return sum(r.correct for r in self.results) / self.n if self.n else 0.0

    @property
    def median_context_tokens(self) -> float:
        if not self.results:
            return 0.0
        vals = sorted(r.context_tokens for r in self.results)
        return vals[len(vals) // 2]

    @property
    def p95_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        vals = sorted(r.latency_ms for r in self.results)
        return vals[min(len(vals) - 1, round(0.95 * (len(vals) - 1)))]

    @property
    def source_session_recall(self) -> float | None:
        """Share of questions whose source session has a selected memory.

        This is deliberately not called answer-support recall: a structured memory
        can point at the correct session while having already dropped the needed
        number, date, or duration during extraction.
        """
        vals = [
            r.source_session_recalled
            if r.source_session_recalled is not None
            else r.evidence_recalled
            for r in self.results
            if r.source_session_recalled is not None or r.evidence_recalled is not None
        ]
        return sum(vals) / len(vals) if vals else None

    @property
    def evidence_recall(self) -> float | None:
        """Compatibility alias for result artifacts written before the rename."""
        return self.source_session_recall

    def by_type(self) -> dict[str, TypeBreakdown]:
        out: dict[str, TypeBreakdown] = {}
        for r in self.results:
            b = out.setdefault(r.question_type, TypeBreakdown(r.question_type, 0, 0))
            b.n += 1
            b.correct += int(r.correct)
        return dict(sorted(out.items()))


def _load_done(path: Path) -> dict[str, QuestionResult]:
    if not path.exists():
        return {}
    done: dict[str, QuestionResult] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            done[json.loads(line)["question_id"]] = QuestionResult(**json.loads(line))
        except (json.JSONDecodeError, TypeError, KeyError):
            continue  # a half-written final line from a killed process
    return done


def run_eval(
    runner: Runner,
    judge: Judge,
    instances: list[Instance],
    out_path: str | Path,
    usage: UsageTracker | None = None,
    resume: bool = True,
    on_progress=None,
) -> RunReport:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with _exclusive(path):
        return _run_eval_locked(runner, judge, instances, path, usage, resume, on_progress)


class StoreChanged(RuntimeError):
    """The artifact being resumed was produced from different data."""


def _check_resumable(done: dict[str, QuestionResult], runner, path: Path) -> None:
    """Refuse to append answers from one store onto answers from another.

    Resume keys on question_id alone, which is right for the case it was built for
    (a run that stopped on quota, continued the next day against the same store) and
    silently wrong for the case that was about to happen here: a 31-question pilot
    run against a 63%-ingested store, then the formal 50-question run after ingest
    finished. Nineteen questions would be answered from the complete store and
    thirty-one reused from the partial one, reported as a single number, with the
    partial rows systematically *worse* because their evidence was missing.

    That is the same defect the `--fresh` truncation already guards against for
    prompts and judges. The store is the third input, and the largest.
    """
    current = getattr(runner, "store_fingerprint", None)
    if current is None or not done:
        return  # a runner that builds its own index per question has no store to pin
    unstamped = sum(1 for r in done.values() if not r.store_fingerprint)
    stale = {r.store_fingerprint for r in done.values() if r.store_fingerprint} - {current}
    if not stale and not unstamped:
        return
    # Rows written before this field existed are refused too. Unknown provenance is
    # not the same as matching provenance, and the artifact this was written for —
    # the A2 pilot — is entirely unstamped, so treating None as "probably fine" would
    # have let through the exact merge it exists to stop.
    origin = ", ".join(sorted(stale)) if stale else f"{unstamped} rows of unknown store"
    raise StoreChanged(
        f"{path} holds {len(done)} answers from {origin}, but this run uses {current}. "
        f"Resuming would merge answers from different stores into one number. "
        f"Write this run to a different --out, or pass --fresh to replace the file."
    )


def _run_eval_locked(runner, judge, instances, path, usage, resume, on_progress) -> RunReport:
    done = _load_done(path) if resume else {}
    if resume:
        _check_resumable(done, runner, path)
    report = RunReport(variant=runner.name, results=list(done.values()))

    if not resume and path.exists():
        # `--fresh` has to truncate, not just skip the resume lookup. Appending to a
        # file that already held a previous run silently merges two runs into one
        # report — and the reason for re-running is usually that something changed
        # (a judge, a prompt), so the merged rows are graded under different rules.
        # That is precisely the flaw D1 rejects LoCoMo for, committed against
        # ourselves. Observed as an n of 56 and 62 on a 50-question subset.
        path.unlink()

    with path.open("a", encoding="utf-8") as sink:
        for inst in instances:
            if inst.question_id in done:
                continue
            try:
                runner.prepare(inst)
                answer = runner.answer(inst)
                verdict = judge.grade(
                    question=inst.question,
                    gold=inst.answer,
                    hypothesis=answer.text,
                    is_abstention=inst.is_abstention,
                    question_type=inst.question_type,
                )
            except DailyQuotaExhausted as exc:
                report.completed = False
                report.stopped_reason = str(exc)
                break

            result = QuestionResult(
                question_id=inst.question_id,
                question_type=inst.question_type,
                is_abstention=inst.is_abstention,
                correct=verdict.correct,
                hypothesis=answer.text,
                gold=inst.answer,
                judge_reason=verdict.reason,
                context_tokens=answer.context_tokens,
                prompt_tokens=answer.prompt_tokens,
                output_tokens=answer.output_tokens,
                latency_ms=answer.latency_ms,
                evidence_recalled=answer.notes.get("evidence_recalled"),
                source_session_recalled=answer.notes.get("source_session_recalled"),
                answer_prompt_version=ANSWER_PROMPT_VERSION,
                judge_prompt_version=JUDGE_PROMPT_VERSION,
                # From the store, not from the checked-out code: it describes the
                # data being evaluated, which an older store will not share.
                extractor_version=getattr(runner, "extractor_version", None),
                store_fingerprint=getattr(runner, "store_fingerprint", None),
                notes=answer.notes,
            )
            report.results.append(result)
            sink.write(json.dumps(asdict(result)) + "\n")
            sink.flush()  # a crash must not lose completed work

            if on_progress:
                on_progress(result, report)

    if usage:
        usage.save(path.with_suffix(".usage.json"), merge=resume)

    _verify_artifact_matches(path, report)
    return report


# The lock lives in llm_long_term_memory.locking because ingest needs the same one:
# a duplicate ingest corrupts usage accounting, which is harder to notice than a
# corrupted artifact. Re-exported under the old name so callers do not have to care.
RunAlreadyInProgress = AlreadyRunning


@contextmanager
def _exclusive(path: Path):
    """Refuse to start when another run owns this output file.

    `_verify_artifact_matches` detects a corrupted artifact after the fact; this
    prevents it. Two evaluations sharing an output path interleave their appends
    and, if either was started with `--fresh`, one truncates the other's completed
    work — observed as a 50-question file dropping to 34 and a second file being
    deleted outright mid-run.
    """
    with exclusive(path, what="evaluation"):
        yield


class ArtifactMismatch(RuntimeError):
    """The JSONL on disk disagrees with the report returned in memory."""


def _verify_artifact_matches(path: Path, report: RunReport) -> None:
    """Fail loudly if the file and the report have diverged.

    The published table is regenerated from these files, so the file is the record
    and the console summary is only a convenience. When the two disagree, every
    number downstream is untrustworthy — and it disagrees silently, which is worse:
    a console line reporting 50 questions at 56% was backed by a file holding 30.
    The cause there was two eval processes racing on one path, but the class of
    failure (a concurrent writer, a truncation, a lost write) recurs, so the check
    is permanent rather than a one-off fix.
    """
    on_disk = _load_done(path)
    if len(on_disk) == report.n:
        return
    raise ArtifactMismatch(
        f"{path} holds {len(on_disk)} results but the run reported {report.n}. "
        "Another process is probably writing to the same file. The file is the "
        "record; discard this run's console output and re-run alone."
    )
