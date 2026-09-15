"""Layer-by-layer fact coverage on BEAM, and the self-check that says whether to trust it.

    python3 tools/beam_fact_coverage.py --validate          # needs no run; zero calls
    python3 tools/beam_fact_coverage.py --rows <jsonl> --store <db>

**`--validate`** grades the matcher against a ground truth that already exists. BEAM labels
which sessions hold each question's evidence, so a required fact ought to be findable in
its own labelled source sessions. The share that is, is the instrument's ceiling: nothing
read through it can be more reliable than that. It also reports what share of rubric items
the matcher refuses outright, per ability, because a coverage figure over a fifth of the
items is a different claim from one over all of them.

**The default mode** reads a finished run's rows and walks every required fact through

    source -> extracted -> retrieved -> context -> answer

reporting, per question, how many of the rubric's own facts each stage still had, and
where the first one was lost. That is the difference between "never extracted", "extracted
but not retrieved", "retrieved but dropped from the context" and "in the context and not
used" — four failures that call for four different fixes and have until now been one
number.

Zero provider calls in both modes: every figure is arithmetic over text already on disk.
No BEAM text is written to the artifacts; counts and question ids only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VALIDATION = REPO / "results/analysis/beam-fact-matcher-validation.json"
COVERAGE = REPO / "results/analysis/beam-fact-coverage.json"

HALF = "dev"


def validate(half: str = HALF, data_dir: Path | None = None) -> dict:
    """Can a required fact be found in its own labelled source sessions?"""
    from llm_long_term_memory.evaluation import beam_coverage as fc
    from llm_long_term_memory.evaluation.datasets import beam

    instances = beam.load(half, data_dir or REPO / "data")
    per_ability: dict[str, Counter] = defaultdict(Counter)
    for instance in instances:
        labelled = set(instance.answer_session_ids)
        conversation = fc.haystack(
            *(turn.content for session in instance.sessions for turn in session.turns)
        )
        gold = fc.haystack(
            *(
                turn.content
                for session in instance.sessions
                if not labelled or session.session_id in labelled
                for turn in session.turns
            )
        )
        counts = per_ability[instance.question_type]
        for raw in instance.rubric:
            item = fc.classify(raw, conversation)
            counts["items"] += 1
            if item.kind != fc.EVIDENCE:
                counts[item.kind] += 1
                continue
            counts["evidence"] += 1
            if all(token in gold for token in item.tokens):
                counts["found_in_labelled_source"] += 1
            elif any(token in gold for token in item.tokens):
                counts["partly_found"] += 1
            else:
                counts["elsewhere_in_conversation"] += 1

    totals = Counter()
    for counts in per_ability.values():
        totals.update(counts)
    evidence = totals["evidence"]
    return {
        "name": "beam-fact-matcher-validation",
        "half": half,
        "provider_calls": 0,
        "question": "is a required fact findable in the sessions BEAM labels as its source?",
        "rubric_items": totals["items"],
        "decidable": evidence + totals[fc.DERIVED],
        "decidable_share": round((evidence + totals[fc.DERIVED]) / max(1, totals["items"]), 3),
        "evidence_bearing": evidence,
        "found_in_labelled_source": totals["found_in_labelled_source"],
        "ceiling": round(totals["found_in_labelled_source"] / max(1, evidence), 3),
        "derived": totals[fc.DERIVED],
        "undecidable": totals[fc.UNDECIDABLE],
        "negative": totals[fc.NEGATIVE_ITEM],
        "reading": (
            "`ceiling` is the share of evidence-bearing items whose every required token "
            "appears in its own labelled source. Coverage figures read through this matcher "
            "cannot be more reliable than it. `decidable_share` says how much of the rubric "
            "the matcher will speak about at all; the rest is reported as undecidable and "
            "never as a loss."
        ),
        "by_ability": {
            ability: dict(sorted(counts.items())) for ability, counts in sorted(per_ability.items())
        },
    }


def analyse(rows_path: Path, store_path: Path, half: str, data_dir: Path) -> dict:
    """Walk every required fact through the pipeline, per question."""
    from llm_long_term_memory.evaluation import beam_coverage as fc
    from llm_long_term_memory.evaluation.beam_report import conversation_of, load_rows
    from llm_long_term_memory.evaluation.datasets import beam
    from llm_long_term_memory.store import SQLiteMemoryStore

    rows = load_rows(rows_path)
    by_id = {instance.question_id: instance for instance in beam.load(half, data_dir)}
    store = SQLiteMemoryStore(store_path, read_only=True)
    store.initialize()

    memories: dict[str, dict[str, str]] = {}
    for namespace in store.user_ids():
        memories[namespace] = {m.id: m.content for m in store.iter_all(namespace)}
    store.close()

    per_question: dict[str, dict] = {}
    first_loss: Counter = Counter()
    by_ability: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        instance = by_id.get(row["question_id"])
        if instance is None:
            continue
        notes = row.get("notes") or {}
        owned = memories.get(instance.namespace, {})
        labelled = set(instance.answer_session_ids)
        ranked = set(notes.get("ranked_memory_ids") or [])
        selected = {hit["memory_id"] for hit in notes.get("retrieval") or []}
        selected |= set(notes.get("hydrated_memory_ids") or [])
        conversation = fc.haystack(
            *(turn.content for session in instance.sessions for turn in session.turns)
        )
        stage_text = {
            "source": fc.haystack(
                *(
                    turn.content
                    for session in instance.sessions
                    if not labelled or session.session_id in labelled
                    for turn in session.turns
                )
            ),
            "extracted": fc.haystack(*owned.values()),
            "retrieved": fc.haystack(*(owned[i] for i in ranked if i in owned)),
            "context": fc.haystack(*(owned[i] for i in selected if i in owned)),
            "answer": fc.haystack(row.get("hypothesis") or ""),
        }
        items = fc.ladder(instance.rubric, conversation, stage_text)
        score = (notes.get("judge") or {}).get("score")
        report = fc.question_report(items, score)
        report["conversation"] = conversation_of(row["question_id"], row["question_type"])
        report["ability"] = row["question_type"]
        report["rubric_score"] = score
        per_question[row["question_id"]] = report
        if report["first_loss"]:
            first_loss[report["first_loss"]] += 1
            by_ability[row["question_type"]][report["first_loss"]] += 1
        elif report["required_facts"]:
            first_loss["nothing lost"] += 1
            by_ability[row["question_type"]]["nothing lost"] += 1

    instrument = {
        "suspected_annotation_gap": sum(
            1 for r in per_question.values() if r["suspected_annotation_gap"]
        ),
        "suspected_judge_error": sum(
            1 for r in per_question.values() if r["suspected_judge_error"]
        ),
        "note": "Not pipeline failures. A required fact absent from its own labelled "
        "source, and a rubric scored zero on an answer carrying every fact it asked for. "
        "Counted so they are not read as extraction or reasoning loss.",
    }
    decidable = sum(r["decidable_items"] for r in per_question.values())
    rubric_items = sum(len(by_id[q].rubric) for q in per_question if q in by_id)
    first = first_loss.most_common(1)
    headline = (
        "Among automatically decidable evidence-bearing rubric items "
        f"({decidable} of {rubric_items} graded items on these questions), "
        f"the most common first loss was {first[0][0]!r} ({first[0][1]} question(s))."
        if first
        else "No question carried an automatically decidable required fact."
    )
    return {
        "name": "beam-fact-coverage",
        # Any sentence quoting these numbers has to carry its denominator. The matcher
        # decides about a fifth of rubric items, so "extraction loss = x%" generalised to
        # BEAM as a whole would be false — and it is the shape of claim this project has
        # already had to retract once, when 98.3% source recall was read as proof that the
        # remaining failures were reasoning.
        "headline_phrasing": headline,
        "denominator": {
            "graded_rubric_items": rubric_items,
            "automatically_decidable": decidable,
            "share": round(decidable / rubric_items, 3) if rubric_items else None,
            "rule": "a figure from this file is quoted only as a share of automatically "
            "decidable evidence-bearing rubric items, never of BEAM questions or of all "
            "rubric items",
        },
        "instrument_errors": instrument,
        "half": half,
        "provider_calls": 0,
        "rows": str(rows_path),
        "questions": len(per_question),
        "layers": fc.layers(rows, per_question),
        "first_loss": dict(first_loss.most_common()),
        "first_loss_by_ability": {a: dict(c.most_common()) for a, c in sorted(by_ability.items())},
        "per_question": per_question,
    }


def _print_validation(payload: dict) -> None:
    print(
        f"{payload['half']} half · {payload['rubric_items']} rubric items · zero provider calls\n"
        f"decidable {payload['decidable']} ({payload['decidable_share']:.0%}), "
        f"of which {payload['evidence_bearing']} evidence-bearing "
        f"and {payload['derived']} derived\n"
        f"matcher ceiling: {payload['found_in_labelled_source']}/{payload['evidence_bearing']} "
        f"= {payload['ceiling']:.0%} found in their own labelled source\n"
    )
    print(f"{'ability':26} {'items':>6} {'evid':>5} {'found':>6} {'derived':>8} {'undec':>6}")
    for ability, counts in payload["by_ability"].items():
        print(
            f"{ability:26} {counts.get('items', 0):6d} {counts.get('evidence', 0):5d} "
            f"{counts.get('found_in_labelled_source', 0):6d} "
            f"{counts.get('derived', 0):8d} {counts.get('undecidable', 0):6d}"
        )


def _print_question(payload: dict, question_id: str) -> None:
    """The ladder for one question, which is how a single failure is read."""
    report = payload["per_question"].get(question_id)
    if report is None:
        print(f"STOP: {question_id} is not in these rows")
        return
    required = report["required_facts"]
    print(f"{question_id}  ({report['ability']})")
    print(f"  {'Gold required facts:':<26}{required:>4}")
    for stage, label in (
        ("source", "In the source turns:"),
        ("extracted", "Extracted:"),
        ("retrieved", "Retrieved:"),
        ("context", "Selected context:"),
        ("answer", "Mentioned in answer:"),
    ):
        print(f"  {label:<26}{report['reached'][stage]:>4} / {required}")
    score = report["rubric_score"]
    print(f"  {'Final rubric score:':<26}{'n/a' if score is None else f'{score:.2f}':>4}")
    if report["undecidable_items"] or report["derived_items"]:
        print(
            f"  [{report['undecidable_items']} rubric item(s) carried no distinctive token; "
            f"{report['derived_items']} name a derived answer]"
        )


def _print_coverage(payload: dict) -> None:
    print(f"{payload['questions']} questions · zero provider calls\n")
    print(f"{payload['headline_phrasing']}\n")
    for layer, row in payload["layers"].items():
        value = "n/a" if row["value"] is None else f"{row['value']:.1%}"
        print(f"  {layer:24} {value:>7}   ({row['conversations']} conversations)")
    print("\nfirst stage that lost a required fact:")
    for stage, n in payload["first_loss"].items():
        print(f"  {stage:24} {n}")
    instrument = payload["instrument_errors"]
    print(
        f"\nnot the pipeline: {instrument['suspected_annotation_gap']} question(s) want a fact "
        f"absent from their own labelled source, "
        f"{instrument['suspected_judge_error']} scored zero with every required fact present"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true", help="grade the matcher; needs no run")
    parser.add_argument("--rows", help="a finished run's JSONL")
    parser.add_argument("--store", help="the store those rows were answered from")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--question", help="print one question's ladder and stop")
    parser.add_argument("--out", help="where to write the artifact")
    args = parser.parse_args()

    if args.validate or not args.rows:
        payload = validate(data_dir=Path(args.data_dir))
        out = Path(args.out) if args.out else VALIDATION
        _print_validation(payload)
    else:
        if not args.store:
            print("STOP: --rows needs --store, the store those rows were answered from")
            return 2
        payload = analyse(Path(args.rows), Path(args.store), HALF, Path(args.data_dir))
        out = Path(args.out) if args.out else COVERAGE
        if args.question:
            _print_question(payload, args.question)
            return 0
        _print_coverage(payload)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
