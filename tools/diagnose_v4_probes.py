"""Close out the two unconcluded v4 runs, and measure what the probe instrument can resolve.

Zero provider calls. Everything here is arithmetic over rows that were already paid for.

Two questions, in this order, because the second one decides whether the first one means
anything.

**How much do the saved runs disagree?** The variant labels, store fingerprints and
retrieval match. Historical code/configuration identity is unknown, so this cannot prove
pure provider sampling. Report both categorical verdict changes and binary correctness
changes; only the latter move the accuracy difference.

**Where do count answers actually fail?** The answerer reports its own item list alongside
its number, so a wrong count can be split into layers that need different fixes:
the arithmetic (is the number the length of the list it named?), the reading (were facts
in the context never named?), and the retrieval (were they missing from the context?).
Facts are matched to the probe's SQL-derived gold memories by token overlap, which is why
the per-fact figures are reported as counts of evidence rather than as a score.

`--out` writes the machine-readable record; the printed summary is the same numbers.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Runs to read. `control` is the v3.3 answerer the v4 arms were registered against; it is
# the archived copy, so this reads the same bytes the conclusion was written from.
RUNS = {
    "control-v3.3": "results/archive/v4.0-flat/probes.v3.3-dev.jsonl",
    "v4.0-flat": "results/archive/v4.0-flat/probes.v4.0-flat.jsonl",
    "v4.0-flat2": "results/raw/probes.v4.0-flat2.jsonl",
    "v4.1-scan": "results/raw/probes.v4.1-scan.jsonl",
}
# The pair that shares a configuration. Their disagreement is the noise floor.
REPLICATE_PAIR = ("v4.0-flat", "v4.0-flat2")

# Words that carry no identity, so that "my new rose bush" matches "rose bush" and a
# question's own phrasing does not match everything.
# fmt: off
_STOP = frozenset([
    "a", "an", "the", "of", "and", "or", "to", "in", "on", "at", "for", "my", "i",
    "user", "s", "is", "was", "were", "has", "have", "had", "some", "new", "their",
])
# fmt: on
_WORD = re.compile(r"[a-z0-9]+")
# Half the item's content words must appear in the gold fact. Deliberately blunt: this
# decides how a *diagnosis* is bucketed, never whether an answer was graded correct.
_OVERLAP = 0.5


def _tokens(text: str | None) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOP and len(w) > 2}


def _as_int(value: object) -> int | None:
    """Rows store `parsed` as a string, because the grader compares normalised text."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _load(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as handle:
        return {row["probe_id"]: row for row in map(json.loads, handle)}


def _correct(rows: dict[str, dict], kind: str | None = None) -> tuple[int, int]:
    subset = [r for r in rows.values() if kind is None or r["kind"] == kind]
    return sum(1 for r in subset if r["verdict"] == "correct"), len(subset)


def _paired(left: dict[str, dict], right: dict[str, dict]) -> dict:
    """Right's wins and losses against left, over the probes both answered."""
    shared = sorted(set(left) & set(right))
    wins = [k for k in shared if right[k]["verdict"] == "correct" != left[k]["verdict"]]
    losses = [k for k in shared if left[k]["verdict"] == "correct" != right[k]["verdict"]]
    return {
        "n": len(shared),
        "wins": len(wins),
        "losses": len(losses),
        "net": len(wins) - len(losses),
        "correctness_changed": len(wins) + len(losses),
        "verdict_changed": sum(1 for k in shared if left[k]["verdict"] != right[k]["verdict"]),
        "retrieval_identical": sum(
            1 for k in shared if left[k]["retrieved_ids"] == right[k]["retrieved_ids"]
        ),
    }


def _gold_texts(store: Path) -> dict[str, dict[str, str]]:
    with sqlite3.connect(f"file:{store}?mode=ro", uri=True) as con:
        return {
            row[0]: {"content": row[1], "object": row[2]}
            for row in con.execute("select id, content, object from memories")
        }


def _match(item: str, gold_ids: list[str], memories: dict[str, dict[str, str]]) -> str | None:
    """The gold fact an enumerated item names, or None if it names none of them."""
    item_tokens = _tokens(item)
    if not item_tokens:
        return None
    best, score = None, 0.0
    for gold_id in gold_ids:
        memory = memories.get(gold_id)
        if memory is None:
            continue
        for field in ("object", "content"):
            gold_tokens = _tokens(memory[field])
            if not gold_tokens:
                continue
            overlap = len(item_tokens & gold_tokens) / len(item_tokens)
            if overlap > score:
                best, score = gold_id, overlap
    return best if score >= _OVERLAP else None


def _diagnose_counts(
    rows: dict[str, dict], probes: dict[str, dict], memories: dict[str, dict[str, str]]
) -> dict:
    layers: dict[str, int] = {
        "correct": 0,
        "abstained_no_number": 0,
        "retrieval_gap": 0,
        "reading_failure": 0,
    }
    direction = {"under": 0, "over": 0}
    facts = {"in_context_never_named": 0, "absent_from_context": 0}
    spurious = duplicated = arithmetic_disagreed = 0

    for row in rows.values():
        if row["kind"] != "count":
            continue
        probe = probes[row["probe_id"]]
        gold, gold_ids = int(probe["answer"]), probe["evidence_memory_ids"]
        items = (row.get("synthesis_computation") or {}).get("items") or []
        parsed = _as_int(row.get("parsed"))
        retrieved = set(row.get("retrieved_ids") or [])

        if parsed is not None and parsed != len(items):
            arithmetic_disagreed += 1

        named = [g for g in (_match(i, gold_ids, memories) for i in items) if g]
        duplicated += len(named) - len(set(named))
        spurious += len(items) - len(named)
        for gold_id in gold_ids:
            if gold_id in set(named):
                continue
            key = "in_context_never_named" if gold_id in retrieved else "absent_from_context"
            facts[key] += 1

        if row["verdict"] == "correct":
            layers["correct"] += 1
            continue
        if parsed is None:
            layers["abstained_no_number"] += 1
            continue
        layers["retrieval_gap" if not row["context_complete"] else "reading_failure"] += 1
        direction["under" if parsed < gold else "over"] += 1

    return {
        "layers": layers,
        "direction_of_wrong_numbers": direction,
        "gold_facts_never_named": facts,
        "spurious_items_named": spurious,
        "same_fact_named_twice": duplicated,
        "number_disagreed_with_own_item_list": arithmetic_disagreed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=REPO / "stores/train150.db")
    parser.add_argument(
        "--probes", type=Path, default=REPO / "results/analysis/synthesis-probes.json"
    )
    parser.add_argument(
        "--out", type=Path, default=REPO / "results/analysis/v4-probe-diagnosis.json"
    )
    args = parser.parse_args()

    missing = [name for name, rel in RUNS.items() if not (REPO / rel).is_file()]
    if missing:
        print(f"STOP: missing probe rows for {', '.join(missing)}")
        return 2

    runs = {name: _load(REPO / rel) for name, rel in RUNS.items()}
    probes = {
        p["probe_id"]: p for p in json.loads(args.probes.read_text(encoding="utf-8"))["probes"]
    }
    memories = _gold_texts(args.store)

    kinds = sorted({r["kind"] for r in runs["v4.0-flat"].values()})
    scoreboard = {}
    for name, rows in runs.items():
        correct, total = _correct(rows)
        scoreboard[name] = {
            "correct": correct,
            "n": total,
            "share": round(correct / total, 4),
            "by_kind": {
                kind: dict(zip(("correct", "n"), _correct(rows, kind), strict=True))
                for kind in kinds
            },
            # A run whose rows carry no routing cannot be graded against a gate that is
            # stated in terms of routing, which is what v4.1's Gate 0 is.
            "rows_with_scan_route": sum(1 for r in rows.values() if r.get("scan_route")),
        }

    left, right = REPLICATE_PAIR
    noise = _paired(runs[left], runs[right])
    noise["per_kind_point_swing"] = {
        kind: round(
            _correct(runs[right], kind)[0] / _correct(runs[right], kind)[1]
            - _correct(runs[left], kind)[0] / _correct(runs[left], kind)[1],
            4,
        )
        for kind in kinds
    }

    payload = {
        "schema_version": 1,
        "not_a_benchmark": (
            "Probe ground truth is derived by SQL over stored facts and inherits extraction "
            "loss. These are instrument readings, never LongMemEval accuracy."
        ),
        "provider_calls": 0,
        "runs": {name: rel for name, rel in RUNS.items()},
        "scoreboard": scoreboard,
        "noise_floor": {
            "pair": list(REPLICATE_PAIR),
            "why_this_is_noise": (
                "Same variant label, store fingerprint and retrieval on shared probes. "
                "Historical source/configuration identity was not recorded, so disagreement "
                "cannot be attributed exclusively to sampling. Only binary correctness "
                "changes belong in the paired-sign calculation."
            ),
            **noise,
        },
        "comparisons_against_control": {
            name: _paired(runs["control-v3.3"], runs[name])
            for name in RUNS
            if name != "control-v3.3"
        },
        "count_diagnosis": {
            name: _diagnose_counts(rows, probes, memories) for name, rows in runs.items()
        },
        "pairwise_retrieval_identity": {
            f"{a} vs {b}": _paired(runs[a], runs[b])["retrieval_identical"]
            for a, b in combinations(RUNS, 2)
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("probe scoreboard (development half, 142 probes)")
    for name, entry in scoreboard.items():
        route = entry["rows_with_scan_route"]
        note = "" if route else "   [no scan_route recorded on any row]"
        print(f"  {name:14s} {entry['correct']:3d}/{entry['n']}  {entry['share']:6.1%}{note}")

    print(f"\nnoise floor — {left} vs {right}, the same configuration")
    print(f"  retrieval identical on   {noise['retrieval_identical']}/{noise['n']} probes")
    print(f"  verdicts that changed    {noise['verdict_changed']}/{noise['n']}")
    print(f"  paired                   {noise['wins']}W-{noise['losses']}L  net {noise['net']:+d}")
    for kind, swing in noise["per_kind_point_swing"].items():
        print(f"    {kind:14s} {swing:+.1%}")

    print("\neffect of each candidate against the registered control")
    for name, entry in payload["comparisons_against_control"].items():
        print(f"  {name:14s} {entry['wins']}W-{entry['losses']}L  net {entry['net']:+d}")

    print("\ncount failures by layer")
    for name, entry in payload["count_diagnosis"].items():
        layers = entry["layers"]
        print(
            f"  {name:14s} correct {layers['correct']:2d}  "
            f"reading {layers['reading_failure']:2d}  "
            f"retrieval {layers['retrieval_gap']:2d}  "
            f"abstained {layers['abstained_no_number']:2d}  "
            f"| number != own list {entry['number_disagreed_with_own_item_list']}"
            f"  under/over {entry['direction_of_wrong_numbers']['under']}"
            f"/{entry['direction_of_wrong_numbers']['over']}"
            f"  facts in context never named {entry['gold_facts_never_named']['in_context_never_named']}"  # noqa: E501
        )

    print(f"\nwrote {args.out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
