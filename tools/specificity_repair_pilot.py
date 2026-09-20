"""Run the preregistered conditional specificity-repair development pilot.

The baseline and candidate share one batch-8 two-stage extraction. The candidate adds
grounded facts only for sessions whose user assertions contain specifics absent from the
baseline. Every successful call is checkpointed before the next one starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

CHECKPOINT = REPO / "results/raw/specificity-repair-pilot.checkpoint.json"
USAGE = REPO / "results/raw/specificity-repair-pilot.usage.json"
RESULT = REPO / "results/analysis/specificity-repair-pilot.json"
REPORT = REPO / "results/analysis/specificity-repair-pilot.md"
CONFIG = REPO / "configs/v2.yaml"
OFFSET = 60
SESSIONS = 60
BATCH = 8


def _cohort() -> list:
    from llm_long_term_memory.config import ExperimentConfig
    from llm_long_term_memory.evaluation.datasets import longmemeval as lme

    cfg = ExperimentConfig.from_yaml(CONFIG)
    every = lme.load(cfg.dataset_variant, REPO / "data")
    _, holdout = lme.split_dev_test(every)
    sessions = [session for instance in holdout for session in instance.sessions]
    cohort = sessions[OFFSET : OFFSET + SESSIONS]
    if len(cohort) != SESSIONS:
        raise RuntimeError(f"wanted {SESSIONS} sessions at offset {OFFSET}, found {len(cohort)}")
    return cohort


def _cohort_digest(sessions: list) -> str:
    from llm_long_term_memory.ingest.fidelity import user_assertions

    digest = hashlib.sha256()
    for session in sessions:
        digest.update(session.session_id.encode())
        digest.update(b"\x00")
        digest.update(user_assertions(session).encode())
        digest.update(b"\x1e")
    return digest.hexdigest()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _new_checkpoint(sessions: list, model: str) -> dict:
    return {
        "experiment": "specificity-repair-pilot",
        "class": "development",
        "created_at": datetime.now().astimezone().isoformat(),
        "git_sha": _git_sha(),
        "config": str(CONFIG.relative_to(REPO)),
        "model": model,
        "offset": OFFSET,
        "sessions": SESSIONS,
        "batch": BATCH,
        "session_ids": [session.session_id for session in sessions],
        "cohort_sha256": _cohort_digest(sessions),
        "baseline": {},
        "repairs": {},
    }


def _load_checkpoint(sessions: list, model: str) -> dict:
    if CHECKPOINT.exists():
        data = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        expected = {
            "model": model,
            "offset": OFFSET,
            "sessions": SESSIONS,
            "batch": BATCH,
            "session_ids": [session.session_id for session in sessions],
            "cohort_sha256": _cohort_digest(sessions),
        }
        moved = [key for key, value in expected.items() if data.get(key) != value]
        if moved:
            raise RuntimeError(
                "checkpoint does not match this registered cohort: " + ", ".join(moved)
            )
        return data
    data = _new_checkpoint(sessions, model)
    _save_checkpoint(data)
    return data


def _save_checkpoint(data: dict) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    temporary = CHECKPOINT.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(CHECKPOINT)


def _memory_dict(memory) -> dict:
    fields = (
        "id",
        "user_id",
        "type",
        "content",
        "token_count",
        "subject",
        "predicate",
        "object",
        "source_session_id",
        "source_turn_index",
        "source_char_start",
        "source_char_end",
    )
    return {field: getattr(memory, field) for field in fields}


def _memory(payload: dict):
    from llm_long_term_memory.store import Memory

    return Memory(**payload)


def _grounded_memory(anchored, session_id: str):
    from llm_long_term_memory.store import Memory

    fact = anchored.fact
    stable_id = hashlib.sha1(
        f"{session_id}|{anchored.turn_index}|{fact.verbatim_span}|{fact.content}".encode()
    ).hexdigest()[:16]
    return Memory(
        id=f"repair_{stable_id}",
        user_id="fidelity",
        type="semantic",
        content=fact.content,
        token_count=max(1, len(fact.content.split())),
        subject=fact.subject,
        predicate=fact.attribute or None,
        object=(f"{fact.value} {fact.unit}".strip() or None) if fact.value else None,
        source_session_id=session_id,
        source_turn_index=anchored.turn_index,
        source_char_start=anchored.char_start,
        source_char_end=anchored.char_end,
    )


def _missing(session, memories: list) -> set[str]:
    from llm_long_term_memory.ingest.fidelity import _extract_facets, user_assertions

    stated = _extract_facets(user_assertions(session))
    blob = " || ".join(f"{memory.content} {memory.object or ''}" for memory in memories).lower()
    return {value for values in stated.values() for value in values if value not in blob}


def _build_client():
    from llm_long_term_memory.config import ExperimentConfig, Settings
    from llm_long_term_memory.llm import Limits, QuotaManager, UsageTracker
    from llm_long_term_memory.llm.client import GeminiClient

    cfg = ExperimentConfig.from_yaml(CONFIG)
    settings = Settings()
    quota = QuotaManager(
        state_dir=settings.store_dir / "quota",
        default=Limits(rpm=cfg.quota.rpm, tpm=cfg.quota.tpm, rpd=cfg.quota.rpd),
    )
    quota.load_learned()
    usage = UsageTracker()
    client = GeminiClient(settings.require_api_key(), quota=quota, usage=usage)
    return cfg, client, usage


def _flush_usage(usage) -> None:
    if not usage.records:
        return
    usage.save(USAGE, merge=True)
    usage.records.clear()


def run() -> int:
    from llm_long_term_memory.ingest import TwoStageExtractor
    from llm_long_term_memory.ingest.grounded import GroundedExtractor, normalise
    from llm_long_term_memory.llm.client import DailyQuotaExhausted

    sessions = _cohort()
    cfg, client, usage = _build_client()
    checkpoint = _load_checkpoint(sessions, cfg.models.extractor)
    baseline = checkpoint["baseline"]
    repairs = checkpoint["repairs"]

    base_extractor = TwoStageExtractor(client, cfg.models.extractor)
    grounded_extractor = GroundedExtractor(client, cfg.models.extractor)

    try:
        for start in range(0, len(sessions), BATCH):
            chunk = sessions[start : start + BATCH]
            pending = [session for session in chunk if session.session_id not in baseline]
            if not pending:
                continue
            outcome = base_extractor.extract(pending)
            by_session = {session.session_id: [] for session in pending}
            for memory in outcome.memories:
                if memory.source_session_id in by_session:
                    by_session[memory.source_session_id].append(_memory_dict(memory))
            baseline.update(by_session)
            _save_checkpoint(checkpoint)
            _flush_usage(usage)
            print(
                f"baseline {min(start + BATCH, len(sessions))}/{len(sessions)}: "
                f"+{len(outcome.memories)} memories"
            )

        for index, session in enumerate(sessions, start=1):
            if session.session_id in repairs:
                continue
            base_memories = [_memory(item) for item in baseline[session.session_id]]
            missing = sorted(_missing(session, base_memories))
            if not missing:
                repairs[session.session_id] = {
                    "called": False,
                    "missing": [],
                    "memories": [],
                    "attempted": 0,
                    "rejected": 0,
                }
                _save_checkpoint(checkpoint)
                continue

            report = grounded_extractor.extract(
                [(turn.role, turn.content) for turn in session.turns],
                session.date,
            )
            accepted = []
            for anchored in report.anchored:
                fact = anchored.fact
                if session.turns[anchored.turn_index].role != "user":
                    continue
                content = normalise(fact.content)
                span = normalise(fact.verbatim_span)
                if any(value in content and value in span for value in missing):
                    accepted.append(_memory_dict(_grounded_memory(anchored, session.session_id)))
            repairs[session.session_id] = {
                "called": True,
                "missing": missing,
                "memories": accepted,
                "attempted": len(report.anchored) + len(report.rejected),
                "rejected": len(report.rejected),
            }
            _save_checkpoint(checkpoint)
            _flush_usage(usage)
            print(
                f"repair {index}/{len(sessions)} {session.session_id}: "
                f"missing={len(missing)} accepted={len(accepted)}"
            )
    except DailyQuotaExhausted as exc:
        _flush_usage(usage)
        print(f"paused on quota: {exc}")
        return 2
    except BaseException:
        _flush_usage(usage)
        raise

    _flush_usage(usage)
    return score(checkpoint, sessions)


def _specificity_outcomes(pairs: list) -> list[tuple[str, str, str, bool]]:
    from llm_long_term_memory.ingest.fidelity import _extract_facets, user_assertions

    rows = []
    for session, memories in pairs:
        blob = " || ".join(f"{memory.content} {memory.object or ''}" for memory in memories).lower()
        for facet, values in _extract_facets(user_assertions(session)).items():
            for value in values:
                rows.append((session.session_id, facet, value, value in blob))
    return rows


def score(checkpoint: dict | None = None, sessions: list | None = None) -> int:
    from llm_long_term_memory.ingest.fidelity import score_sessions

    sessions = sessions or _cohort()
    if checkpoint is None:
        if not CHECKPOINT.exists():
            raise RuntimeError(f"missing checkpoint: {CHECKPOINT}")
        checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if len(checkpoint["baseline"]) != len(sessions) or len(checkpoint["repairs"]) != len(sessions):
        raise RuntimeError("pilot is incomplete; run without `score` to resume")

    baseline_pairs = []
    candidate_pairs = []
    for session in sessions:
        base = [_memory(item) for item in checkpoint["baseline"][session.session_id]]
        repaired = [_memory(item) for item in checkpoint["repairs"][session.session_id]["memories"]]
        baseline_pairs.append((session, base))
        candidate_pairs.append((session, [*base, *repaired]))

    baseline_report = score_sessions(baseline_pairs)
    candidate_report = score_sessions(candidate_pairs)
    baseline_outcomes = _specificity_outcomes(baseline_pairs)
    candidate_outcomes = _specificity_outcomes(candidate_pairs)
    assert [row[:3] for row in baseline_outcomes] == [row[:3] for row in candidate_outcomes]
    gained = sum(
        not before[3] and after[3]
        for before, after in zip(baseline_outcomes, candidate_outcomes, strict=True)
    )
    lost = sum(
        before[3] and not after[3]
        for before, after in zip(baseline_outcomes, candidate_outcomes, strict=True)
    )
    repair_calls = sum(row["called"] for row in checkpoint["repairs"].values())
    added_memories = sum(len(row["memories"]) for row in checkpoint["repairs"].values())
    delta = candidate_report.overall - baseline_report.overall
    gates = {
        "fidelity_gain_at_least_10pp": delta >= 0.10,
        "at_least_10_gains": gained >= 10,
        "zero_losses": lost == 0,
        "repair_calls_at_most_half": repair_calls <= SESSIONS / 2,
        "added_memories_at_most_one_per_session": added_memories <= SESSIONS,
        "cohort_complete": len(checkpoint["repairs"]) == SESSIONS,
    }
    decision = "PASS_TO_QA_GATE" if all(gates.values()) else "STOP"
    payload = {
        "experiment": checkpoint["experiment"],
        "class": "development",
        "decision": decision,
        "git_sha": checkpoint["git_sha"],
        "cohort_sha256": checkpoint["cohort_sha256"],
        "sessions": SESSIONS,
        "offset": OFFSET,
        "model": checkpoint["model"],
        "baseline": {
            "overall": baseline_report.overall,
            "memories": baseline_report.memories,
            "per_facet": {name: asdict(facet) for name, facet in baseline_report.per_facet.items()},
        },
        "candidate": {
            "overall": candidate_report.overall,
            "memories": candidate_report.memories,
            "per_facet": {
                name: asdict(facet) for name, facet in candidate_report.per_facet.items()
            },
        },
        "delta": delta,
        "gained": gained,
        "lost": lost,
        "repair_calls": repair_calls,
        "added_memories": added_memories,
        "gates": gates,
        "usage_path": str(USAGE.relative_to(REPO)),
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(
        "# Specificity repair pilot\n\n"
        "> **DEVELOPMENT RESULT**: source-text fidelity mechanism experiment, not QA "
        "or unseen final evidence.\n\n"
        f"Decision: **{decision}**\n\n"
        f"- baseline: {baseline_report.overall:.1%}\n"
        f"- candidate: {candidate_report.overall:.1%}\n"
        f"- delta: {delta:+.1%}\n"
        f"- paired specifics: {gained} gains / {lost} losses\n"
        f"- repair calls: {repair_calls}/{SESSIONS}\n"
        f"- added memories: {added_memories} ({added_memories / SESSIONS:.2f}/session)\n\n"
        "## Registered gates\n\n"
        + "\n".join(
            f"- {'PASS' if passed else 'FAIL'} — `{name}`" for name, passed in gates.items()
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    return 0 if decision == "PASS_TO_QA_GATE" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "score"), nargs="?", default="run")
    args = parser.parse_args()
    return score() if args.command == "score" else run()


if __name__ == "__main__":
    raise SystemExit(main())
