"""Ask for the same fact in different words, and see whether it still comes back.

Every retrieval number this project has is measured with **the question the benchmark
wrote**. That leaves one thing untested: whether a fact is retrievable at all, or only
retrievable by the phrasing it was asked in. A store that answers "how many postcards do
I have?" and misses "what's my collection up to now?" has memorised a query, not a fact.

BEAM makes the test possible without writing new data or spending a request, because its
rubric items **name the required fact in words of their own**:

    question   "What did the doctor say about my blood pressure?"
    rubric     "LLM response should mention: 140/90"

So the rubric item is already a second phrasing of the same target. Querying with it and
asking whether the same required facts reach the context is a paraphrase test that costs
nothing.

**What this is and is not.** It tests the retrieval half only — the answerer is not
involved, so it cannot say whether the model would have used what came back. And rubric
wording is not a user's wording; it is terser and more literal, which if anything makes
retrieval *easier*. A drop here is therefore a floor on the real drop, not an estimate of
it. The stronger experiment — rewriting all 440 questions with a second model and freezing
them as a separate set — costs 440 requests and needs its own registration; this is what
can be learned before that.

    python3 tools/paraphrase_probe.py --store beam-dev-frozen \\
        --questions results/manifests/beam-dev.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

OUT = REPO / "results/analysis/paraphrase-probe.json"


def probe(runner, instances) -> dict:
    """For each question, retrieve twice — by the question, then by each rubric item."""
    from llm_long_term_memory.evaluation import beam_coverage as fc

    owned: dict[str, dict[str, str]] = {}
    rows = []
    for instance in instances:
        namespace = instance.store_namespace
        if namespace not in owned:
            owned[namespace] = {m.id: m.content for m in runner.store.iter_all(namespace)}
        conversation = fc.haystack(
            *(turn.content for session in instance.sessions for turn in session.turns)
        )
        items = [
            (number, item)
            for number, item in enumerate(instance.rubric, start=1)
            if fc.classify(item, conversation).kind == fc.EVIDENCE
        ]
        if not items:
            continue

        def context_for(text: str, namespace: str = namespace) -> str:
            hits = runner.retriever.retrieve(
                runner.encoder.encode_one(text),
                text,
                namespace,
                temporal=runner.temporal,
                limit=runner.top_k,
            )
            return fc.haystack(*(hit.memory.content for hit in hits))

        asked = context_for(instance.question)
        for number, item in items:
            classified = fc.classify(item, conversation)
            wanted = set(classified.tokens)
            # The rubric item stripped of its "LLM response should mention:" frame is the
            # paraphrase: same fact, different words, written by someone who was not
            # writing a question.
            paraphrase = fc.payload(item)
            reached_by_question = {t for t in wanted if t in asked}
            reached_by_paraphrase = {t for t in wanted if t in context_for(paraphrase)}
            rows.append(
                {
                    "question_id": instance.question_id,
                    "item": number,
                    "ability": instance.question_type,
                    "required": len(wanted),
                    "by_question": len(reached_by_question),
                    "by_paraphrase": len(reached_by_paraphrase),
                    "lost_by_rewording": sorted(reached_by_question - reached_by_paraphrase),
                    "found_only_by_rewording": sorted(reached_by_paraphrase - reached_by_question),
                }
            )
    return {"rows": rows}


def summarise(rows: list[dict]) -> dict:
    if not rows:
        return {"items": 0}
    by_ability: dict[str, list[dict]] = {}
    for row in rows:
        by_ability.setdefault(row["ability"], []).append(row)

    def rate(subset, key):
        required = sum(r["required"] for r in subset)
        return (sum(r[key] for r in subset) / required) if required else None

    return {
        "items": len(rows),
        "required_facts": sum(r["required"] for r in rows),
        "reached_by_the_question": rate(rows, "by_question"),
        "reached_by_the_rubric_wording": rate(rows, "by_paraphrase"),
        "items_where_rewording_lost_a_fact": sum(1 for r in rows if r["lost_by_rewording"]),
        "items_where_rewording_found_one": sum(1 for r in rows if r["found_only_by_rewording"]),
        "by_ability": {
            ability: {
                "items": len(subset),
                "by_question": rate(subset, "by_question"),
                "by_rubric_wording": rate(subset, "by_paraphrase"),
            }
            for ability, subset in sorted(by_ability.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--variant", default="two_stage_fallback")
    parser.add_argument("--config", default="configs/v2.yaml")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import os

    os.environ.setdefault("GEMINI_API_KEY", "probe-no-request-is-sent")

    import beam_dry_run

    from llm_long_term_memory.cli import _build, _manifest_instances
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.evaluation.manifest import load_manifest

    settings = Settings()
    manifest = load_manifest(args.questions)
    by_id = {i.question_id: i for i in _manifest_instances(manifest, settings)}
    _cfg, _s, runner, _j, _u = _build(
        args.variant,
        args.config,
        store_name=args.store,
        client_override=beam_dry_run.FakeProvider(need_source_percent=0),
        read_only_store=True,
    )
    present = set(runner.store.user_ids())
    instances = [
        by_id[q] for q in manifest.question_ids if q in by_id and by_id[q].namespace in present
    ]
    try:
        outcome = probe(runner, instances)
    finally:
        runner.store.close()

    summary = summarise(outcome["rows"])
    payload = {
        "name": "paraphrase-probe",
        "provider_calls": 0,
        "store": args.store,
        "questions_probed": len(instances),
        "measures": "retrieval only; whether the answerer would have used the fact is out "
        "of reach here",
        "limits": "rubric wording is terser and more literal than a user's, so a drop here "
        "is a floor on the real drop rather than an estimate of it",
        **summary,
    }
    out = Path(args.out) if args.out else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"{payload['questions_probed']} questions, {payload['items']} decidable rubric items")
    print("required facts reaching the context:")
    print(f"  asked with the benchmark's question   {payload['reached_by_the_question']:.1%}")
    print(f"  asked with the rubric's own wording   {payload['reached_by_the_rubric_wording']:.1%}")
    print(
        f"  items where rewording lost a fact     {payload['items_where_rewording_lost_a_fact']}"
        f"\n  items where rewording found one       {payload['items_where_rewording_found_one']}"
    )
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
