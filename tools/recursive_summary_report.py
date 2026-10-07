"""Zero-call gates and resource accounting for recursive navigation experiments."""

from __future__ import annotations

import hashlib
import math
import statistics
from collections import defaultdict
from decimal import Decimal
from statistics import NormalDist

from analysis_io import sha256_file


def wilson(successes, n, confidence=0.975):
    if not 0 <= successes <= n or n <= 0:
        raise ValueError("invalid paired sample")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p, z2 = successes / n, z * z
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return max(0, center - half), min(1, center + half)


def paired_accuracy(runs, types):
    if set(runs) != {"control", "candidate"} or any(len(v) != 3 for v in runs.values()):
        raise ValueError("three fresh repeats per arm required")
    for repeats in runs.values():
        for rows in repeats:
            if set(rows) != set(types) or any(
                type(r["correct"]) is not bool for r in rows.values()
            ):
                raise ValueError("incomplete paired outcomes")
    deltas = {
        q: (
            sum(r[q]["correct"] for r in runs["candidate"])
            - sum(r[q]["correct"] for r in runs["control"])
        )
        / 3
        for q in types
    }
    control = {q: sum(r[q]["correct"] for r in runs["control"]) >= 2 for q in types}
    candidate = {q: sum(r[q]["correct"] for r in runs["candidate"]) >= 2 for q in types}
    wins = sum(candidate[q] and not control[q] for q in types)
    losses = sum(control[q] and not candidate[q] for q in types)
    wl, wu = wilson(wins, len(types))
    ll, lu = wilson(losses, len(types))
    lower, upper = wl - lu, wu - ll
    net = sum(deltas.values())
    return {
        "mean_correct": {
            a: sum(sum(r[q]["correct"] for q in types) for r in reps) / 3
            for a, reps in runs.items()
        },
        "per_run_correct": {
            a: [sum(r[q]["correct"] for q in types) for r in reps] for a, reps in runs.items()
        },
        "mean_net": net,
        "by_type_mean_net": {
            t: sum(d for q, d in deltas.items() if types[q] == t)
            for t in sorted(set(types.values()))
        },
        "majority_paired": {
            "questions": len(types),
            "wins": wins,
            "losses": losses,
            "difference_ci95": [lower, upper],
            "method": "Bonferroni 97.5% Wilson win/loss bounds",
        },
        "statistical_no_loss_supported": lower >= 0 and net >= 0,
    }


def model_workload(records, role=None):
    out = defaultdict(lambda: {"calls": 0, "failures": 0, "input_tokens": 0, "output_tokens": 0})
    for r in records:
        if role is not None and r.role != role:
            continue
        row = out[r.model]
        row["calls"] += 1
        row["failures"] += not r.ok
        row["input_tokens"] += r.input_tokens
        row["output_tokens"] += r.output_tokens
    return dict(out)


def resource_total(workload):
    return sum(r["input_tokens"] + r["output_tokens"] for r in workload.values())


def amortization(build, update, control_queries, summary_queries):
    """One history per user; totals are workload tokens, never currency."""
    if set(control_queries) != set(summary_queries) or set(build) != set(summary_queries):
        raise ValueError("per-user resource scopes differ")
    users = {}
    for user in sorted(build):
        baseline, candidate = control_queries[user], summary_queries[user]
        saving = baseline - candidate
        users[user] = {
            "build_workload_tokens": build[user],
            "update_workload_tokens": update.get(user, 0),
            "mean_control_query_workload_tokens": baseline,
            "mean_summary_query_workload_tokens": candidate,
            "break_even_queries": math.ceil((build[user] + update.get(user, 0)) / saving)
            if saving > 0
            else None,
            "scenarios": {
                str(q): {
                    "control_workload_tokens": q * baseline,
                    "summary_workload_tokens": build[user] + q * candidate,
                    "summary_with_one_update_workload_tokens": build[user]
                    + update.get(user, 0)
                    + q * candidate,
                }
                for q in (1, 3, 10, 25)
            },
        }
    return {
        "units": "workload tokens across fixed models, not financial cost",
        "users": users,
        "same_question_repeats_are_not_query_diversity": True,
    }


def raw_reference_coverage(instance, reader):
    from llm_long_term_memory.store import external_session_id

    gold = {
        (s.session_id, n, hashlib.sha256(t.content.encode()).hexdigest())
        for s in instance.sessions
        for n, t in enumerate(s.turns)
        if t.has_answer
    }
    calls = reader["answer"]["notes"].get("grounded_calls", [])
    sources = calls[-1]["evidence"]["sources"] if calls else []
    found = {
        (external_session_id(s["session_id"]), s["turn_index"], s["text_sha256"])
        for s in sources
        if s["kind"] == "raw"
    }
    # Abstention questions may have no positive reference turn. That is not a
    # retrieval loss; absence is still graded by the original judge.
    return gold <= found if gold else instance.is_abstention


def faithful_gap(instance, reader):
    from llm_long_term_memory.store import external_session_id

    notes = reader["answer"]["notes"]
    gap = notes.get("evidence_gap") or {}
    if gap.get("status") != "needs_clarification" or not gap.get("missing_fields"):
        return False
    original = {
        (s.session_id, n): t.content for s in instance.sessions for n, t in enumerate(s.turns)
    }
    references = gap.get("sources", [])
    if not references or not all(
        hashlib.sha256(
            original.get((external_session_id(s["session_id"]), s["turn_index"]), "").encode()
        ).hexdigest()
        == s["sha256"]
        for s in references
    ):
        return False
    if instance.question_id == "gpt4_731e37d7":
        calls = notes.get("grounded_calls", [])
        calc = calls[-1].get("calculation", {}) if calls else {}
        sources = {s["id"]: s for s in calls[-1]["evidence"]["sources"]} if calls else {}
        payments = calc.get("payments", [])
        valid_quotes = bool(payments)
        for payment in payments:
            source = sources.get(payment["source"])
            if not source or source["kind"] != "raw" or source["role"] != "user":
                valid_quotes = False
                break
            quote = payment["quote"]
            text = original.get(
                (external_session_id(source["session_id"]), source["turn_index"]), ""
            )
            if quote not in text or payment["unit"] + payment["amount"] not in quote:
                valid_quotes = False
        return (
            valid_quotes
            and sorted(p["amount"] for p in payments) == ["20", "200", "500"]
            and str(calc.get("reported_total")) == "720"
            and sum(Decimal(p["amount"]) for p in payments) == Decimal("720")
            and calc.get("mechanism") == "scoped_payment_conflict"
            and not calc.get("scope_complete", True)
            and set(gap["missing_fields"]) == {"event_date", "event_year", "report_period"}
            and "cannot confirm" in reader["answer"]["text"]
        )
    expected = {
        "f9e8c073": "period_identity_unspecified",
        "gpt4_2f8be40d": "scope_incomplete",
    }
    fields = {
        "f9e8c073": {"report_period", "overlap_between_reports"},
        "gpt4_2f8be40d": {"event_identity", "event_date", "report_period"},
    }
    return gap.get("cause") == expected.get(instance.question_id) and set(
        gap["missing_fields"]
    ) == fields.get(instance.question_id)


def pilot_gate(instances, readers, grades, gaps):
    if set(readers) != set(instances) or set(grades) != set(instances):
        return {"pass": False, "status": "INCOMPLETE"}
    rows = []
    for q, instance in instances.items():
        reader = readers[q]
        supported = q not in gaps
        coverage = raw_reference_coverage(instance, reader)
        correct = grades[q]["verdict"]["correct"]
        faithful = not supported and faithful_gap(instance, reader)
        context_ok = reader["answer"]["context_tokens"] <= 6000
        rows.append(
            {
                "question_id": q,
                "supported": supported,
                "judged_correct": correct,
                "gold_raw_final_context_complete": coverage,
                "faithful_gap": faithful,
                "context_bound": context_ok,
                "pass": coverage and context_ok and (correct if supported else faithful),
            }
        )
    return {
        "pass": all(r["pass"] for r in rows),
        "status": "COMPLETE",
        "rows": rows,
        "original_correct": sum(r["judged_correct"] for r in rows),
        "supported_correct": sum(r["judged_correct"] for r in rows if r["supported"]),
        "supported_total": sum(r["supported"] for r in rows),
    }


def latency_summary(readers):
    values = sorted(r["answer"]["notes"]["end_to_end_wall_ms"] for r in readers.values())
    return {
        "p50_ms": statistics.median(values),
        "p95_ms": values[min(len(values) - 1, math.ceil(len(values) * 0.95) - 1)],
    }


def file_inputs(paths):
    return {str(p): sha256_file(p) for p in paths}
