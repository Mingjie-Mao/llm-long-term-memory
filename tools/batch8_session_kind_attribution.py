"""Why three batch-8 readings disagree: the kind of session, not the cohort. No calls.

    python tools/batch8_session_kind_attribution.py

`batch8_cohort_attribution.py` showed the 64.9%, 44.8% and 47.5% readings come from
nearly disjoint sessions, that facet mix does not explain the gap, and left cohort
versus post-extraction processing unseparated. This splits the same three
reconstructions by the *source* of each session in LongMemEval's haystack:

* `evidence`  — `answer_*`, the sessions that carry a question's answer;
* `simulated` — hex ids, LongMemEval's simulated user-assistant chats;
* `sharegpt`  — `sharegpt_*`, real task conversations used as filler;
* `ultrachat` — `ultrachat_*`, likewise.

**Exploratory and post hoc.** The split was chosen after looking at one per-kind
table, and it is labelled so. It is reported because the result is large and the
mechanism was then checked by reading sessions, not because it was registered.

Post-extraction processing is addressed separately and directly: the gate16 baseline
store's own ingest record shows how many extracted memories deduplication dropped.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import read_json, write_report  # noqa: E402

KINDS = ("simulated", "evidence", "ultrachat", "sharegpt")
TASK_KINDS = ("sharegpt",)


def session_kind(session_id: str) -> str:
    if session_id.startswith("answer_"):
        return "evidence"
    match = re.match(r"(sharegpt|ultrachat)_", session_id)
    return match.group(1) if match else "simulated"


def _score(pairs: list[tuple]) -> dict:
    from llm_long_term_memory.ingest.fidelity import _extract_facets, user_assertions

    stated = kept = 0
    for session, memories in pairs:
        blob = " || ".join(f"{m.content} {m.object or ''}" for m in memories).lower()
        for values in _extract_facets(user_assertions(session)).values():
            for value in values:
                stated += 1
                kept += value in blob
    return {
        "sessions": len(pairs),
        "specifics": stated,
        "kept": kept,
        "recall": kept / stated if stated else None,
    }


def standardise(per_kind: dict, weights: dict[str, int]) -> float | None:
    """This cohort's per-kind recall, weighted by another composition's specifics."""
    total = kept = 0.0
    for kind, weight in weights.items():
        recall = per_kind.get(kind, {}).get("recall")
        if recall is None or not weight:
            continue
        total += weight
        kept += weight * recall
    return kept / total if total else None


def analyse() -> dict:
    from batch8_cohort_attribution import cohort_pairs

    pairs, history = cohort_pairs()
    cohorts: dict[str, dict] = {}
    for name, cohort in pairs.items():
        by_kind: dict[str, list] = defaultdict(list)
        for session, memories in cohort:
            by_kind[session_kind(session.session_id)].append((session, memories))
        overall = _score(cohort)
        personal = _score([p for k, ps in by_kind.items() if k not in TASK_KINDS for p in ps])
        cohorts[name] = {
            "overall": overall,
            "without_task_sessions": personal,
            "task_share_of_specifics": sum(
                _score(by_kind[k])["specifics"] for k in TASK_KINDS if k in by_kind
            )
            / max(1, overall["specifics"]),
            "by_kind": {k: _score(by_kind[k]) for k in KINDS if k in by_kind},
        }

    reference = cohorts["C_gate16_store"]["by_kind"]
    c_weights = {k: v["specifics"] for k, v in reference.items()}
    standardised = {
        name: {
            "own": row["overall"]["recall"],
            "at_C_composition": standardise(row["by_kind"], c_weights),
            "C_recall_at_this_composition": standardise(
                reference, {k: v["specifics"] for k, v in row["by_kind"].items()}
            ),
        }
        for name, row in cohorts.items()
    }

    ingest = read_json(REPO / "stores/v2b-gate16-ingest.json")
    return {
        "experiment": "batch8-session-kind-attribution-v1",
        "class": "development, exploratory, post hoc",
        "model": history["model"],
        "provider_calls": 0,
        "cohorts": cohorts,
        "standardised": standardised,
        "gate16_dedup": {
            "memories_written": ingest["memories_written"],
            "duplicates_dropped": ingest["duplicates_dropped"],
        },
    }


def _pct(value) -> str:
    return "—" if value is None else f"{value:.1%}"


def render(result: dict) -> str:
    names = list(result["cohorts"])
    labels = {
        "A_first60_extraction": "A first 60",
        "B_second60_extraction": "B second 60",
        "C_gate16_store": "C gate16 store",
    }
    lines = [
        "# Batch 8 readings by session kind",
        "",
        "> DEVELOPMENT, zero calls, **exploratory and post hoc** — the split was chosen",
        "> after seeing one per-kind table. Same reconstructions as",
        "> `batch8-cohort-attribution-v1`.",
        "",
        "## Recall by kind of session",
        "",
        "| kind | " + " | ".join(labels[n] for n in names) + " |",
        "|---|" + "---:|" * len(names),
    ]
    for kind in KINDS:
        cells = []
        for name in names:
            row = result["cohorts"][name]["by_kind"].get(kind)
            cells.append(
                "—" if row is None else f"{_pct(row['recall'])} ({row['kept']}/{row['specifics']})"
            )
        lines.append(f"| `{kind}` | " + " | ".join(cells) + " |")
    lines.append(
        "| **all** | "
        + " | ".join(f"**{_pct(result['cohorts'][n]['overall']['recall'])}**" for n in names)
        + " |"
    )
    lines.append(
        "| **without `sharegpt`** | "
        + " | ".join(
            f"**{_pct(result['cohorts'][n]['without_task_sessions']['recall'])}**" for n in names
        )
        + " |"
    )
    lines.append(
        "| `sharegpt` share of specifics | "
        + " | ".join(f"{result['cohorts'][n]['task_share_of_specifics']:.0%}" for n in names)
        + " |"
    )
    lines += [
        "",
        "## Composition, standardised",
        "",
        "| cohort | own recall | its per-kind recall at C's composition | "
        "C's per-kind recall at its composition |",
        "|---|---:|---:|---:|",
    ]
    for name in names:
        row = result["standardised"][name]
        lines.append(
            f"| {labels[name]} | {_pct(row['own'])} | {_pct(row['at_C_composition'])} | "
            f"{_pct(row['C_recall_at_this_composition'])} |"
        )
    dedup = result["gate16_dedup"]
    lines += [
        "",
        "## Post-extraction processing",
        "",
        f"The gate16 baseline store's ingest record: {dedup['duplicates_dropped']} memories "
        f"dropped as duplicates against {dedup['memories_written']:,} written. Deduplication "
        "cannot account for a gap of this size in C.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO / "results/analysis/batch8-session-kind-attribution-v1.json",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=REPO / "results/analysis/batch8-session-kind-attribution-v1.md",
    )
    args = parser.parse_args()
    result = analyse()
    print(write_report(result, render, json_out=args.json_out, md_out=args.md_out), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
