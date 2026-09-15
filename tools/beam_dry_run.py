"""Rehearse the whole BEAM path with a fake provider, before any quota is spent.

    python3 tools/beam_dry_run.py --conversations 2

Runs the real CLI entry points — `lltm ingest run` and `lltm eval run`, called as
functions — with `GeminiClient` replaced by a fake that answers from the prompt it was
given. Nothing is sent anywhere. What is being tested is not accuracy; it is that

    BEAM export -> ingestion -> store -> retrieval -> answer -> judge -> rows -> report

survives end to end, and that **the rows carry the fields the conclusion depends on**. A
run that completes and writes rows without `notes.judge.grades` is a full-price run with
no conclusion in it (the report's 花配额之前 section, steps 3 and 4).

The canned replies are shaped for the policy under test, which is the rule this file
exists to obey. A reply that always says `status: "answer"` never enters the raw-source
fallback, so the fallback's second call — the difference between 1.0 and ~1.34 requests
per question, and therefore between one quota day and two — would not be rehearsed at
all. So a fixed share of answers ask for source, and the judge's grades are computed from
whether the answer actually contains the rubric item's words, so that scores vary with
retrieval instead of being a constant the aggregation cannot be read through.

Everything is written under `--work` (inside the ignored `stores/`), never into the
results directory a real run resumes from. The committed artifact is the summary at
`results/analysis/beam-dry-run.json`, which quotes no BEAM text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

HALF = "dev"
"""Never the final half. Exporting it needs `--final-run`, and rehearsing on it would
spend the one set this project has left."""

CHARS_PER_TOKEN = 4.6
"""D5's measured ratio. Token figures here are estimates from prompt length, not provider
counts — a dry run has no provider to count them."""

SESSION_BLOCK = re.compile(r"^=== Session index (\d+) \| date: (.+) ===$", re.M)
NUMBERED = re.compile(r"^(\d+)\.\s+(.*)$", re.M)
WORD = re.compile(r"[a-z0-9]{5,}")


# --------------------------------------------------------------------------- fake


@dataclass
class FakeProvider:
    """Stands in for `GeminiClient`, answering from the prompt's own text.

    Its replies are derived rather than fixed so that the pipeline downstream has real
    content to work on: facts really come from the sessions, so retrieval really has
    something to find, and the judge's grades really depend on the answer.
    """

    facts_per_session: int = 6
    need_source_percent: int = 25
    # Where the canned judge puts its grade boundaries. Exposed so a second judging pass
    # can differ from the first: a rehearsal whose repeat is byte-identical measures a
    # variance of exactly zero, which is the one answer that cannot be right.
    full_credit_at: float = 0.5
    half_credit_at: float = 0.2
    calls: Counter = field(default_factory=Counter)
    by_kind: Counter = field(default_factory=Counter)
    prompt_chars: Counter = field(default_factory=Counter)

    def generate(self, *, role, model, prompt, system=None, schema=None, **_):
        from llm_long_term_memory.llm.client import Completion

        kind, text = self._reply(role, prompt, schema)
        self.calls[role] += 1
        self.by_kind[kind] += 1
        self.prompt_chars[kind] += len(prompt)
        return Completion(
            text=text,
            model=model,
            input_tokens=round(len(prompt) / CHARS_PER_TOKEN),
            output_tokens=round(len(text) / CHARS_PER_TOKEN),
            thinking_tokens=0,
            attempts=1,
            api_latency_ms=0.0,
        )

    # -- dispatch ----------------------------------------------------------------

    def _reply(self, role: str, prompt: str, schema) -> tuple[str, str]:
        name = getattr(schema, "__name__", "")
        if name == "FactsResult":
            return "extract_stage_a", self._facts(prompt)
        if name == "KeyingResult":
            return "extract_stage_b", self._keying(prompt)
        if name == "DedupDecision":
            return "dedup_adjudication", self._dedup(prompt)
        if name == "RubricVerdict":
            return "judge", self._grades(prompt)
        if role == "answerer":
            kind = "answer" if schema is not None else "answer_from_raw_source"
            return kind, self._answer(prompt, structured=schema is not None)
        raise NotImplementedError(
            f"the dry run has no canned reply for role={role!r} schema={name!r}; "
            "a call it cannot shape is a call it cannot rehearse"
        )

    # -- ingestion ---------------------------------------------------------------

    @staticmethod
    def _sessions(prompt: str) -> list[tuple[int, str]]:
        marks = list(SESSION_BLOCK.finditer(prompt))
        blocks = []
        for position, mark in enumerate(marks):
            end = marks[position + 1].start() if position + 1 < len(marks) else len(prompt)
            blocks.append((int(mark.group(1)), prompt[mark.end() : end]))
        return blocks

    def _facts(self, prompt: str) -> str:
        sessions = []
        for index, body in self._sessions(prompt):
            facts = []
            for line in body.splitlines():
                if not line.startswith("user: "):
                    continue
                for sentence in re.split(r"(?<=[.!?])\s+", line[len("user: ") :]):
                    sentence = sentence.strip()
                    if len(sentence) < 25:
                        continue
                    facts.append(
                        {
                            "content": f"The user said: {sentence[:220]}",
                            "source_role": "user",
                            "subject": "user",
                            "scope": "profile",
                        }
                    )
                    if len(facts) >= self.facts_per_session:
                        break
                if len(facts) >= self.facts_per_session:
                    break
            sessions.append({"session_index": index, "facts": facts})
        return json.dumps({"sessions": sessions})

    @staticmethod
    def _keying(prompt: str) -> str:
        facts = []
        for number, content in NUMBERED.findall(prompt.split("## Facts", 1)[-1]):
            words = WORD.findall(content.lower())
            facts.append(
                {
                    "index": int(number),
                    "temporal_key": words[0] if words else "states",
                    "update_op": "coexists",
                    "object": words[1] if len(words) > 1 else "",
                }
            )
        return json.dumps({"facts": facts})

    @staticmethod
    def _dedup(prompt: str) -> str:
        old = prompt.split("Existing memory:", 1)[-1].split("\n", 1)[0].strip()
        new = prompt.split("New candidate:", 1)[-1].split("\n", 1)[0].strip()
        if old.casefold() == new.casefold():
            verdict, reason = "DUPLICATE", "identical text"
        elif old.split()[:5] == new.split()[:5]:
            verdict, reason = "UPDATE", "same opening, different value"
        else:
            verdict, reason = "DISTINCT", "different facts"
        return json.dumps({"verdict": verdict, "reason": reason})

    # -- answering ---------------------------------------------------------------

    @staticmethod
    def _context_and_question(prompt: str) -> tuple[str, str]:
        head, _, question = prompt.rpartition("\nQuestion: ")
        return head, question.strip()

    def _answer(self, prompt: str, structured: bool) -> str:
        context, question = self._context_and_question(prompt)
        lines = [
            line.strip("- ").strip() for line in context.splitlines() if len(line.strip()) > 30
        ]
        body = " ".join(lines[-3:])[:600] or "Nothing in memory covers that."
        if not structured:
            # The fallback's second pass carries no schema and must return prose. A dry
            # run that returned JSON here would rehearse the raw-structure leak instead
            # of the fallback.
            return f"{body}"
        digest = int(hashlib.sha1(question.encode("utf-8")).hexdigest()[:8], 16)
        if digest % 100 < self.need_source_percent:
            return json.dumps(
                {
                    "status": "need_source",
                    "answer": "",
                    "reason": "the exact figure was not preserved in the memories",
                    "source_query": " ".join(WORD.findall(question.lower())[:6]),
                }
            )
        return json.dumps({"status": "answer", "answer": body, "reason": "", "source_query": ""})

    # -- judging -----------------------------------------------------------------

    def _grades(self, prompt: str) -> str:
        answer = prompt.split("<<<ANSWER", 1)[-1].split("ANSWER>>>", 1)[0]
        items_block = prompt.split("Rubric items:", 1)[-1].split("Grade every rubric item", 1)[0]
        ordering = "`position`" in prompt
        seen = set(WORD.findall(answer.lower()))
        grades = []
        rank = 0
        for number, item in NUMBERED.findall(items_block):
            words = set(WORD.findall(item.lower()))
            hit = len(words & seen) / len(words) if words else 0.0
            score = (
                1.0 if hit >= self.full_credit_at else (0.5 if hit >= self.half_credit_at else 0.0)
            )
            position = None
            if ordering and score > 0:
                rank += 1
                position = rank
            grades.append(
                {
                    "item": int(number),
                    "score": score,
                    "reason": f"{len(words & seen)} of {len(words)} key words present",
                    "position": position,
                }
            )
        return json.dumps({"grades": grades})


@dataclass
class Interrupted(FakeProvider):
    """A fake that fails part-way, so the two failures a long paid run actually meets are
    rehearsed rather than assumed.

    `quota`: the daily budget runs out mid-run. This is the normal path, not the error
    path — dev60 and test100 both crossed a day boundary — and the harness answers it by
    returning a partial report.

    `judge`: the judge returns grades that do not cover the rubric exactly once. The judge
    refuses to repair those, on purpose, so the run stops; what has to be true is that the
    rows already bought survive it and that resuming does not buy them again.
    """

    mode: str = "quota"
    fail_at: int = 12
    fired: bool = False

    def generate(self, *, role, model, prompt, system=None, schema=None, **kwargs):
        from llm_long_term_memory.llm.client import Completion, DailyQuotaExhausted
        from llm_long_term_memory.llm.rate_limiter import Wait

        judging = getattr(schema, "__name__", "") == "RubricVerdict"
        due = not self.fired and self.calls[role] >= self.fail_at
        if due and self.mode == "quota" and role == "answerer":
            self.fired = True
            raise DailyQuotaExhausted(model, Wait(seconds=3600.0, reason="rpd"))
        if due and self.mode == "judge" and judging:
            self.fired = True
            self.calls["judge"] += 1
            # Off the 1.0 / 0.5 / 0.0 scale, and only one grade. A rubric of two items
            # or more is already refused by the item numbers; a one-item rubric — the
            # median for five of the ten abilities — is refused by the score, which is
            # why the drill does not rely on the count alone.
            return Completion(
                text=json.dumps({"grades": [{"item": 1, "score": 0.42, "reason": "off scale"}]}),
                model=model,
                input_tokens=0,
                output_tokens=0,
                thinking_tokens=0,
                attempts=1,
            )
        return super().generate(
            role=role, model=model, prompt=prompt, system=system, schema=schema, **kwargs
        )


# --------------------------------------------------------------------------- run


def manifest_for(conversations: int, work: Path, data_dir: Path) -> tuple[Path, list]:
    """A `beam-dev` manifest holding the first `conversations` conversations, whole.

    Whole conversations, because a conversation is the unit of analysis and a store: half
    a conversation would rehearse a store no real run ever builds.
    """
    from llm_long_term_memory.evaluation.datasets import beam
    from llm_long_term_memory.evaluation.manifest import Manifest

    instances = beam.load(HALF, data_dir)
    by_scale: dict[str, list[str]] = {}
    for instance in instances:
        scale = instance.namespace.split("-")[1]
        if instance.namespace not in by_scale.setdefault(scale, []):
            by_scale[scale].append(instance.namespace)
    # One from each scale before a second from either. A 500K conversation holds three to
    # four times the sessions of a 100K one, and fourteen of the development half's
    # twenty-two are 500K, so a rehearsal that only ever saw 100K would be rehearsing the
    # smaller two thirds of the work.
    ordered: list[str] = []
    for position in range(max(len(names) for names in by_scale.values())):
        for scale in sorted(by_scale):
            if position < len(by_scale[scale]):
                ordered.append(by_scale[scale][position])
    chosen = set(ordered[:conversations])
    picked = [instance for instance in instances if instance.namespace in chosen]
    path = work / "manifests" / f"beam-{HALF}.json"
    Manifest(
        name=f"beam-{HALF}",
        variant="beam",
        seed=0,
        question_ids=tuple(instance.question_id for instance in picked),
        note=f"dry run over {len(chosen)} conversation(s); zero provider calls",
    ).save(path)
    return path, picked


def check_rows(rows: list[dict], instances: list, store, fingerprint: str) -> list[str]:
    """What the run wrote, field by field. Each returned string is a defect."""
    from llm_long_term_memory.evaluation.beam_judge import BEAM_JUDGE_PROMPT_VERSION

    problems = []
    by_id = {instance.question_id: instance for instance in instances}
    answered = {row["question_id"] for row in rows}
    missing = sorted(set(by_id) - answered)
    if missing:
        problems.append(f"{len(missing)} question(s) produced no row, first {missing[0]}")

    owned: dict[str, set[str]] = {}
    for namespace in {instance.namespace for instance in instances}:
        owned[namespace] = {memory.id for memory in store.iter_all(namespace)}

    for row in rows:
        qid = row["question_id"]
        instance = by_id.get(qid)
        if instance is None:
            problems.append(f"{qid}: a row for a question the manifest does not name")
            continue
        judge = (row.get("notes") or {}).get("judge") or {}
        grades = judge.get("grades") or []
        if len(grades) != len(instance.rubric):
            problems.append(
                f"{qid}: {len(grades)} grade(s) for {len(instance.rubric)} rubric item(s)"
            )
        if judge.get("version") != BEAM_JUDGE_PROMPT_VERSION:
            problems.append(f"{qid}: graded by {judge.get('version')!r}")
        if judge.get("score") is None:
            problems.append(f"{qid}: no question score on the row")
        if row["question_type"] == "event_ordering" and judge.get("order_tau_norm", "") == "":
            problems.append(f"{qid}: an event-ordering row with no order metric")
        for column in ("store_fingerprint", "extractor_version", "answer_prompt_version"):
            if not row.get(column):
                problems.append(f"{qid}: {column} is empty, so the row names no provenance")
        if row.get("store_fingerprint") not in (None, fingerprint):
            problems.append(f"{qid}: answered from {row['store_fingerprint']}, not {fingerprint}")
        if row.get("source_session_recalled") is None and instance.answer_session_ids:
            problems.append(f"{qid}: source-session recall unrecorded on a question with evidence")
        foreign = [
            hit["memory_id"]
            for hit in (row.get("notes") or {}).get("retrieval") or []
            if hit["memory_id"] not in owned.get(instance.namespace, set())
        ]
        if foreign:
            problems.append(
                f"{qid}: retrieved {len(foreign)} memory/memories from another conversation"
            )
    return problems


def _drill(cli, client_module, settings, args, manifest, questions: int, mode: str) -> dict:
    """One interruption, then a resume. Returns the drill's record and its defects."""
    from llm_long_term_memory.evaluation.beam_judge import InvalidJudgement
    from llm_long_term_memory.evaluation.harness import _load_done

    problems: list[str] = []
    label = f"beam-{HALF}-dry-{mode}-stop"
    raw = settings.results_dir / "raw"
    path = raw / f"{args.variant}.{label}.jsonl"

    def run(fresh: bool) -> None:
        cli.eval_run(
            variant=args.variant,
            config=args.config,
            limit=None,
            questions=str(manifest),
            store_name="beam-dry-run",
            fresh=fresh,
            top_k=None,
            rerank=None,
            label=label,
        )

    broken = Interrupted(mode=mode, fail_at=max(2, questions // 3))
    client_module.GeminiClient = lambda *a, **k: broken  # type: ignore[assignment]
    stopped = None
    try:
        run(fresh=True)
    except InvalidJudgement as exc:
        stopped = type(exc).__name__
    partial = len(_load_done(path))
    if not broken.fired:
        problems.append(f"{mode} drill: the interruption never fired")
    if partial >= questions:
        problems.append(f"{mode} drill: the run finished despite being interrupted")
    if mode == "judge" and stopped is None:
        problems.append("judge drill: an off-rubric grade set did not stop the run")
    if mode == "quota" and stopped is not None:
        problems.append(f"quota drill: a quota stop raised {stopped} instead of returning")
    if not (raw / f"{args.variant}.{label}.usage.json").is_file():
        problems.append(f"{mode} drill: the interruption lost its usage accounting")

    clean = FakeProvider()
    client_module.GeminiClient = lambda *a, **k: clean  # type: ignore[assignment]
    run(fresh=False)
    finished = len(_load_done(path))
    if finished != questions:
        problems.append(f"{mode} drill: resume left {questions - finished} question(s) short")
    rebought = clean.calls["judge"] - (questions - partial)
    if rebought > 0:
        problems.append(f"{mode} drill: resume re-answered {rebought} question(s)")
    return {
        "problems": problems,
        "record": {
            "stopped_after_rows": partial,
            "stop": stopped or "partial report returned",
            "rows_after_resume": finished,
            "questions_re_answered_by_resume": max(0, rebought),
        },
    }


def interruption_drills(cli, client_module, settings, args, manifest, questions: int) -> dict:
    """Stop the run the two ways a long paid run actually stops, then resume it.

    Every paid run this project has made was interrupted — by quota, by DNS, by a 503 —
    and the property that matters is not that it never stops but that stopping costs the
    question in flight and nothing more. That is checked here rather than believed.
    """
    problems: list[str] = []
    drills = {}
    for mode in ("quota", "judge"):
        outcome = _drill(cli, client_module, settings, args, manifest, questions, mode)
        problems.extend(outcome["problems"])
        drills[mode] = outcome["record"]
    return {"problems": problems, "drills": drills}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", type=int, default=2)
    parser.add_argument("--config", default="configs/v2.yaml")
    parser.add_argument("--variant", default="two_stage_fallback")
    parser.add_argument("--work", default="stores/beam-dry-run")
    parser.add_argument("--keep", action="store_true", help="do not clear the work directory")
    args = parser.parse_args()

    work = (REPO / args.work).resolve()
    if work.exists() and not args.keep:
        shutil.rmtree(work)
    (work / "stores").mkdir(parents=True, exist_ok=True)
    (work / "results").mkdir(parents=True, exist_ok=True)

    # A dry run must never write where a real run resumes from, and must never need a
    # key. Set before anything constructs Settings.
    os.environ["LLTM_STORE_DIR"] = str(work / "stores")
    os.environ["LLTM_RESULTS_DIR"] = str(work / "results")
    os.environ.setdefault("GEMINI_API_KEY", "dry-run-no-request-is-sent")

    from llm_long_term_memory import cli
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.evaluation import beam_report
    from llm_long_term_memory.evaluation.harness import _load_done
    from llm_long_term_memory.ingest import namespace_batch_count, namespaced_sessions
    from llm_long_term_memory.ingest.dedup import Deduplicator
    from llm_long_term_memory.llm import client as client_module

    settings = Settings()
    fake = FakeProvider()
    client_module.GeminiClient = lambda *a, **k: fake  # type: ignore[assignment]

    manifest, instances = manifest_for(args.conversations, work, settings.data_dir)
    store_name = "beam-dry-run"
    label = f"beam-{HALF}-dry"

    # Deduplication searches the shared index with no namespace condition, while
    # retrieval filters on `memory.user_id == namespace`. Whether that reaches across
    # conversations is a property of the data, so it is watched rather than assumed —
    # and on BEAM it matters more than it did on LongMemEval, because the split keeps
    # conversations generated from one seed on the same side, so a 100K conversation and
    # its 500K twin are ingested into the same index.
    crossings: list[tuple[str, str]] = []
    adjudicate = Deduplicator.adjudicate

    def watched(self, old, new):
        if old.user_id != new.user_id:
            crossings.append((old.user_id, new.user_id))
        return adjudicate(self, old, new)

    Deduplicator.adjudicate = watched  # type: ignore[method-assign]

    cli.ingest_run(
        config=args.config,
        limit=None,
        sessions=None,
        fresh=True,
        store_name=store_name,
        questions=str(manifest),
    )
    ingest_calls = dict(fake.by_kind)
    Deduplicator.adjudicate = adjudicate  # type: ignore[method-assign]

    cli.eval_run(
        variant=args.variant,
        config=args.config,
        limit=None,
        questions=str(manifest),
        store_name=store_name,
        fresh=True,
        top_k=None,
        rerank=None,
        label=label,
    )

    rows_path = settings.results_dir / "raw" / f"{args.variant}.{label}.jsonl"
    rows = beam_report.load_rows(rows_path)

    from llm_long_term_memory.store import SQLiteMemoryStore

    store = SQLiteMemoryStore(settings.store_dir / f"{store_name}.db", read_only=True)
    store.initialize()
    fingerprint = next(iter({row.get("store_fingerprint") for row in rows}), None)
    problems = check_rows(rows, instances, store, fingerprint)
    if crossings:
        pairs = sorted({tuple(sorted(pair)) for pair in crossings})
        problems.append(
            f"deduplication adjudicated {len(crossings)} candidate(s) against memories of "
            f"another conversation ({len(pairs)} conversation pair(s)); a DUPLICATE verdict "
            "there deletes a fact the other conversation never contained "
            "(results/analysis/dedup-namespace-leak.json)"
        )
    memories = {ns: len(list(store.iter_all(ns))) for ns in sorted(store.user_ids())}
    store.close()

    # The fact-coverage analyzer is rehearsed here, on fake rows, for the same reason
    # everything else is: it has to work the day the first real rows land, not a week
    # later. Its numbers mean nothing (the memories are this file's invention); its
    # plumbing is what is being checked.
    import beam_fact_coverage

    coverage = beam_fact_coverage.analyse(
        rows_path, settings.store_dir / f"{store_name}.db", HALF, settings.data_dir
    )
    report = beam_report.report(rows, fact_coverage=coverage)
    if not report["mandatory_secondary_complete"]:
        problems.append(
            "the registered evidence layers came out empty: "
            + ", ".join(report["mandatory_secondary_missing"])
        )
    ladders = [q for q in coverage["per_question"].values() if q["required_facts"]]
    if not ladders:
        problems.append("no question produced a required-fact ladder; the analyzer read nothing")

    # Resume has to be a no-op, or an interrupted paid run re-answers what it already
    # bought. Counting rows is not enough — a re-answered question overwrites its own row
    # and the count stays put — so the requests are counted too.
    before = len(rows)
    calls_before_resume = sum(fake.calls.values())
    cli.eval_run(
        variant=args.variant,
        config=args.config,
        limit=None,
        questions=str(manifest),
        store_name=store_name,
        fresh=False,
        top_k=None,
        rerank=None,
        label=label,
    )
    after = len(_load_done(rows_path))
    calls_after_resume = sum(fake.calls.values())
    if after != before:
        problems.append(f"resume changed the row count from {before} to {after}")
    if calls_after_resume != calls_before_resume:
        problems.append(
            f"resume spent {calls_after_resume - calls_before_resume} more request(s) on "
            "questions already answered"
        )

    drills = interruption_drills(cli, client_module, settings, args, manifest, len(instances))
    problems.extend(drills["problems"])

    # A second arm, so the paired comparison the registration is decided by is rehearsed
    # too, not only the single-arm report. The fake asks for raw source more often here,
    # which is one changed behaviour and therefore a difference the test should see.
    second = FakeProvider(need_source_percent=60)
    client_module.GeminiClient = lambda *a, **k: second  # type: ignore[assignment]
    second_label = f"{label}-arm-b"
    cli.eval_run(
        variant=args.variant,
        config=args.config,
        limit=None,
        questions=str(manifest),
        store_name=store_name,
        fresh=True,
        top_k=None,
        rerank=None,
        label=second_label,
    )
    rows_b = beam_report.load_rows(
        settings.results_dir / "raw" / f"{args.variant}.{second_label}.jsonl"
    )
    comparison = beam_report.compare(rows, rows_b)
    if comparison["conversations_compared"] != args.conversations:
        problems.append(
            f"the paired comparison kept {comparison['conversations_compared']} of "
            f"{args.conversations} conversation(s)"
        )
    identical = beam_report.compare(rows, rows)
    if identical["mean_difference"] != 0.0 or identical["p_value"] is None:
        problems.append("comparing an arm with itself did not come out at zero difference")

    # The full development half's extraction cost, counted rather than extrapolated.
    from llm_long_term_memory.config import ExperimentConfig
    from llm_long_term_memory.evaluation.datasets import beam as beam_dataset
    from llm_long_term_memory.ingest.pipeline import fit_batch_size

    cfg = ExperimentConfig.from_yaml(args.config)
    every = beam_dataset.load(HALF, settings.data_dir)
    per_request = fit_batch_size(
        cfg.ingest.sessions_per_request, cfg.quota.tpm, tokens_per_session=2_560
    )
    all_sessions = namespaced_sessions(every)
    batches = namespace_batch_count(all_sessions, per_request)
    stages = 2 if cfg.ingest.two_stage else 1
    rehearsed_sessions = len(namespaced_sessions(instances))
    answerer_calls = fake.calls["answerer"]
    rehearsed_rate = answerer_calls / max(1, len(instances))
    # Stage A's prompt is the conversation text itself, so its size is a property of the
    # data and can be counted exactly for the whole half. Stage B's prompt is the facts
    # Stage A returned, which in a rehearsal are this file's invention, so it is not
    # projected here.
    stage_a_chars = sum(
        len(turn.role) + len(turn.content) + 2
        for _, session in all_sessions
        for turn in session.turns
    )
    payload = {
        "name": "beam-dry-run",
        "half": HALF,
        "provider_calls": 0,
        "note": "every reply was produced locally from the prompt; nothing was sent",
        "rehearsal": {
            "conversations": args.conversations,
            "questions": len(instances),
            "sessions": rehearsed_sessions,
            "memories_by_conversation": memories,
            "cross_namespace_adjudications": len(crossings),
            "calls_by_kind": dict(fake.by_kind),
            "calls_during_ingest": ingest_calls,
            "estimated_input_tokens": {
                kind: round(chars / CHARS_PER_TOKEN) for kind, chars in fake.prompt_chars.items()
            },
            "answerer_calls_per_question": round(answerer_calls / max(1, len(instances)), 3),
            "resume_added_rows": after - before,
            "requests_spent_by_resume": calls_after_resume - calls_before_resume,
            "interruption_drills": drills["drills"],
            "paired_comparison": {
                key: value for key, value in comparison.items() if key != "by_conversation"
            },
        },
        "development_half": {
            "conversations": len({instance.namespace for instance in every}),
            "questions": len(every),
            "session_chunks": len(all_sessions),
            "sessions_per_request": per_request,
            "batches": batches,
            "extraction_requests": batches * stages,
            "projected_judge_requests": len(every),
            "stage_a_input_tokens": round(stage_a_chars / CHARS_PER_TOKEN),
            "projected_answerer_requests": {
                "at_longmemeval_fallback_rate_1.34": round(len(every) * 1.34),
                "at_this_rehearsal_rate": round(len(every) * rehearsed_rate),
            },
            "basis": "Extraction is counted from the real batching and is exact. The "
            "answerer figure is not: how often the answerer asks for raw source is a "
            "property of the model, and a dry run has no model. 1.34 requests per question "
            "is LongMemEval's measured rate (the report's 花配额之前 section); the rehearsal's own "
            "rate comes from this file's declared need_source share and measures nothing. "
            "The first BEAM answering run replaces both with a measurement.",
        },
        "row_checks": {
            "rows": len(rows),
            "problems": problems,
            "passed": not problems,
        },
        "fact_coverage_rehearsal": {
            "questions_with_a_required_fact_ladder": len(ladders),
            "layers": report["evidence_layers"],
            "first_loss": coverage["first_loss"],
            "note": "shapes only; the memories these rows were answered from are canned",
        },
        "aggregate_shape": {
            "primary_questions": report["primary"]["questions"],
            "primary_conversations": report["primary"]["conversations"],
            "abilities_present": sorted(report["by_ability"]),
            "event_order_tau_norm_computed": report["event_order_tau_norm"] is not None,
        },
    }
    out = REPO / "results/analysis/beam-dry-run.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print()
    print(json.dumps({k: v for k, v in payload.items() if k != "rehearsal"}, indent=2))
    print(f"\nwritten: {out.relative_to(REPO)}")
    if problems:
        print(f"\nSTOP: {len(problems)} defect(s) in what the run wrote")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
