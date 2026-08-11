"""Turn run outputs into the README table.

Reads the JSONL each run appends to, so the table can be regenerated at any time
from committed artifacts — including for runs that stopped early on quota.
"""

from __future__ import annotations

import json
from pathlib import Path

from .harness import QuestionResult, RunReport

# Column order follows LongMemEval's ability taxonomy, with the two abilities a
# memory system is uniquely responsible for placed where they are easy to compare.
TYPE_COLUMNS = [
    ("single-session-user", "SS-user"),
    ("single-session-assistant", "SS-asst"),
    ("single-session-preference", "SS-pref"),
    ("multi-session", "Multi-sess"),
    ("temporal-reasoning", "Temporal"),
    ("knowledge-update", "Know-update"),
]


def load_report(path: str | Path, variant: str | None = None) -> RunReport:
    p = Path(path)
    results = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            results.append(QuestionResult(**json.loads(line)))
        except (json.JSONDecodeError, TypeError, KeyError):
            # A process killed mid-write leaves a partial final line. The resume
            # path already tolerates this; the reporting path has to as well, or a
            # run that was interrupted — the normal case on a daily quota — cannot
            # be reported on at all.
            continue
    return RunReport(variant=variant or p.stem, results=results)


def _pct(x: float | None) -> str:
    return f"{x * 100:.1f}%" if x is not None else "—"


def render_table(reports: list[RunReport]) -> str:
    header = (
        ["Variant", "n", "Accuracy"]
        + [label for _, label in TYPE_COLUMNS]
        + ["Abstention", "Evid. recall", "Ctx tokens", "p95 latency"]
    )
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    for rep in reports:
        by_type = rep.by_type()
        abst = [r for r in rep.results if r.is_abstention]
        row = [
            f"`{rep.variant}`" + ("" if rep.completed else " ⚠️"),
            str(rep.n),
            f"**{_pct(rep.accuracy)}**",
        ]
        for key, _ in TYPE_COLUMNS:
            b = by_type.get(key)
            row.append(_pct(b.accuracy) if b else "—")
        row.append(_pct(sum(r.correct for r in abst) / len(abst)) if abst else "—")
        row.append(_pct(rep.evidence_recall))
        row.append(f"{rep.median_context_tokens:,.0f}")
        row.append(f"{rep.p95_latency_ms / 1000:.1f}s")
        lines.append("| " + " | ".join(row) + " |")

    if any(not r.completed for r in reports):
        lines.append("")
        lines.append("⚠️ = run stopped early on daily quota; numbers are partial.")
    return "\n".join(lines)


def render_summary(rep: RunReport) -> str:
    out = [
        f"### `{rep.variant}`",
        "",
        f"- questions: **{rep.n}**",
        f"- accuracy: **{_pct(rep.accuracy)}**",
        f"- median context tokens: **{rep.median_context_tokens:,.0f}**",
        f"- p95 latency: **{rep.p95_latency_ms / 1000:.1f}s**",
    ]
    if rep.evidence_recall is not None:
        out.append(f"- evidence recall: **{_pct(rep.evidence_recall)}**")
    if not rep.completed:
        out.append(f"- ⚠️ stopped early: {rep.stopped_reason}")
    out += ["", "| question type | n | accuracy |", "|---|---|---|"]
    for key, b in rep.by_type().items():
        out.append(f"| {key} | {b.n} | {_pct(b.accuracy)} |")
    return "\n".join(out)
