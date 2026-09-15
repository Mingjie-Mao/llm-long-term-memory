"""The registered BEAM numbers, computed from saved rows.

Every figure here is derived from the per-item grades already on each result row, so a
change to how they are aggregated costs no judge call — the same rule that made the count
probes' verdicts a derived column after a paid run was read the wrong way once
(`results/gate-protocol.md`).

Three levels, in this order, because the middle one is what makes the test honest:

    question      the mean of its own rubric items, so a 12-item summarization question
                  and a 1-item knowledge_update question weigh the same
    conversation  the mean of its questions; twenty questions share one store, so they
                  are one observation, not twenty
    arm           the mean over conversations

`configs/beam-eval.json` names which abilities the primary covers. It is read rather than
hard-coded so that the declaration and the report cannot drift apart.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import fmean

from llm_long_term_memory.evaluation.beam_judge import CORRECT_AT, rescore
from llm_long_term_memory.evaluation.clustered import sign_flip_p_value

DECLARATION = Path("configs/beam-eval.json")

_DIRECTIVE = {"instruction_following", "preference_following"}
"""Abilities whose rubric grades a standing directive rather than a located fact."""


class MissingGrades(ValueError):
    """A row carries no rubric grades, so nothing can be recomputed from it."""


def conversation_of(question_id: str, question_type: str) -> str:
    """`beam-100K-10-event_ordering-1` -> `beam-100K-10`, the store it was answered from."""
    marker = f"-{question_type}-"
    head, sep, _ = question_id.rpartition(marker)
    if not sep:
        raise ValueError(f"{question_id!r} does not name ability {question_type!r}")
    return head


def load_rows(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def scored(rows: Iterable[dict], correct_at: float = CORRECT_AT) -> list[dict]:
    """One record per question: conversation, ability, score, verdict, order metric.

    Scores are recomputed from the grades rather than read off `correct`, so a row written
    under one aggregation rule reports under the current one.
    """
    out = []
    for row in rows:
        details = (row.get("notes") or {}).get("judge")
        if not details or not details.get("grades"):
            raise MissingGrades(f"{row.get('question_id')!r} carries no rubric grades")
        ability = row["question_type"]
        derived = rescore(details, ability, correct_at)
        out.append(
            {
                "question_id": row["question_id"],
                "conversation": conversation_of(row["question_id"], ability),
                "ability": ability,
                "score": derived["score"],
                "correct": derived["correct"],
                "order_tau_norm": derived.get("order_tau_norm"),
                "source_session_recalled": row.get("source_session_recalled"),
                "context_tokens": row.get("context_tokens"),
            }
        )
    return out


def _by_conversation(records: Sequence[dict], key: str) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for record in records:
        value = record[key]
        if value is not None:
            groups.setdefault(record["conversation"], []).append(float(value))
    return {name: fmean(values) for name, values in sorted(groups.items())}


def _stratum(records: Sequence[dict], abilities: Sequence[str] | None) -> list[dict]:
    if abilities is None:
        return list(records)
    wanted = set(abilities)
    return [record for record in records if record["ability"] in wanted]


def summarise(records: Sequence[dict], abilities: Sequence[str] | None = None) -> dict:
    """Mean rubric score and binary accuracy over one stratum, by conversation."""
    chosen = _stratum(records, abilities)
    if not chosen:
        return {"questions": 0, "conversations": 0, "mean_score": None, "binary_accuracy": None}
    scores = _by_conversation(chosen, "score")
    verdicts = _by_conversation([{**r, "correct": float(r["correct"])} for r in chosen], "correct")
    recall = _by_conversation(chosen, "source_session_recalled")
    return {
        "questions": len(chosen),
        "conversations": len(scores),
        "mean_score": fmean(scores.values()),
        "binary_accuracy": fmean(verdicts.values()),
        "source_session_recall": fmean(recall.values()) if recall else None,
        "by_conversation": scores,
    }


def declaration(path: str | Path = DECLARATION) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def report(
    rows: Iterable[dict],
    declared: dict | None = None,
    fact_coverage: dict | None = None,
) -> dict:
    """Every figure `configs/beam-eval.json` registers, for one arm.

    `fact_coverage` is `tools/beam_fact_coverage.py`'s per-question output. Layers 1 and 2
    of the registered diagnostic ride on the rows; layers 3 and 4 need the store and the
    benchmark text, so they are passed in. Without them the report says so rather than
    quietly omitting them: they are mandatory secondary metrics, and a metric that is only
    reported when someone remembers is not mandatory.
    """
    declared = declared or declaration()
    rows = list(rows)
    records = scored(rows, declared["judge"]["correct_at"])
    primary = list(declared["primary"]["abilities"])
    separate = list(declared["reported_separately"]["abilities"])
    groups = declared.get("ability_groups") or {}
    facts = list(groups.get("fact") or [a for a in primary if a not in _DIRECTIVE])
    directive = list(groups.get("standing_directive") or sorted(_DIRECTIVE))
    every = sorted({record["ability"] for record in records})
    # Every composition the primary could have taken, reported whatever it says. The
    # line between a remembered preference and a remembered instruction is a judgement
    # call, so the result has to show what the other calls would have given rather than
    # asking a reader to trust this one.
    compositions = {
        "fact_abilities_only": facts,
        "primary_as_registered": primary,
        "primary_plus_instruction_following": sorted(set(primary) | set(directive)),
        "all_abilities": every,
    }

    order = [r["order_tau_norm"] for r in records if r["order_tau_norm"] is not None]
    from llm_long_term_memory.evaluation.beam_coverage import layers

    diagnostic = layers(rows, (fact_coverage or {}).get("per_question", {}))
    missing = [name for name, row in diagnostic.items() if row["value"] is None]
    return {
        "primary": summarise(records, primary),
        "all_abilities": summarise(records, None),
        "fact_abilities": summarise(records, facts),
        "compositions": {
            name: summarise(records, abilities) for name, abilities in compositions.items()
        },
        "reported_separately": {ability: summarise(records, [ability]) for ability in separate},
        "by_ability": {ability: summarise(records, [ability]) for ability in every},
        "event_order_tau_norm": fmean(order) if order else None,
        # The four registered layers: any source session, all source sessions,
        # required-fact coverage, answer utilisation.
        "evidence_layers": diagnostic,
        "mandatory_secondary_complete": not missing,
        "mandatory_secondary_missing": missing,
    }


def compare(
    rows_a: Iterable[dict],
    rows_b: Iterable[dict],
    declared: dict | None = None,
    abilities: Sequence[str] | None = None,
    seed: int = 0,
) -> dict:
    """Paired difference b - a, tested by sign flips over conversations.

    Only conversations both arms answered are compared, and every question of a compared
    conversation has to be present in both: a conversation half-answered in one arm would
    contribute a mean over a different question set, which is not a paired difference.
    """
    declared = declared or declaration()
    abilities = list(abilities or declared["primary"]["abilities"])
    left = {r["question_id"]: r for r in _stratum(scored(rows_a), abilities)}
    right = {r["question_id"]: r for r in _stratum(scored(rows_b), abilities)}
    shared_questions = set(left) & set(right)
    conversations: dict[str, list[float]] = {}
    for question_id in sorted(shared_questions):
        conversations.setdefault(left[question_id]["conversation"], []).append(
            right[question_id]["score"] - left[question_id]["score"]
        )
    dropped = sorted(
        {left[q]["conversation"] for q in set(left) - shared_questions}
        | {right[q]["conversation"] for q in set(right) - shared_questions}
    )
    complete = {name: fmean(diffs) for name, diffs in conversations.items() if name not in dropped}
    nets = list(complete.values())
    return {
        "abilities": abilities,
        "conversations_compared": len(complete),
        "conversations_dropped_incomplete": dropped,
        "questions_compared": len(shared_questions),
        "mean_difference": fmean(nets) if nets else None,
        "conversations_favouring_b": sum(1 for net in nets if net > 0),
        "conversations_favouring_a": sum(1 for net in nets if net < 0),
        "conversations_tied": sum(1 for net in nets if net == 0),
        "p_value": sign_flip_p_value(nets, seed=seed) if nets else None,
        "by_conversation": complete,
    }
