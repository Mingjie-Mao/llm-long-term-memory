"""Falsify a retrieval-side candidate before paying for it.

Retrieval is deterministic given the store. So any claim of the shape "if we ranked
differently / kept more candidates / budgeted the context differently, the evidence would
be there" can be tested by replaying it — no extraction, no answering, no judging, no
quota. A hypothesis that survives a free attempt to falsify it is worth paying for; one
that does not was going to fail anyway (the report's 花配额之前 section, step 2). That step has
already changed the answer twice: `fallback_depth_probe.py` killed a `max_turns` fix by
showing it reached 1 of 18 questions.

This generalises those one-off probes. It sweeps retrieval configurations over a store that
already exists and reports, per configuration, what reached the context:

    any source session      at least one labelled source session is represented
    all source sessions     every labelled source session is

A third layer — whether the *fact* the question needs reached the context, rather than the
session containing it — needs a benchmark that names the required fact in words of its own.
LongMemEval does not, so it is not measurable here and is not reported. The answerer is
stubbed and the judge never runs, so **the fourth layer — whether the model used what it
was given — is out of reach by construction.** Nothing in this file can say a
candidate improves answers; it can only say a candidate cannot, because the evidence still
is not there. That is the cheap half of the question and it is the half that eliminates.

    python3 tools/retrieval_replay.py --store train150 \\
        --questions results/manifests/train150.json --sweep top_k=10,20,40
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import fmean

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

OUT = REPO / "results/analysis/retrieval-replay.json"


AXES = ("top_k", "rerank", "recency", "halflife", "strength", "decay_halflife", "evict_to")
"""What may be swept. `recency` is the weight on the recency signal, `halflife` its
half-life in days, `strength` whether decayed strength multiplies the score,
`decay_halflife` the decay applied before retrieval, and `evict_to` a per-namespace
capacity cap applied before any question is asked.

`decay_halflife` and `evict_to` **write to the store**. Point them at a copy: a sweep
that mutates the store an experiment was paid for is not a measurement, it is damage."""


class StubAnswerer:
    """Answers every question with a fixed verdict, so no request is ever sent.

    Retrieval is what is being measured; the answerer only has to return something the
    runner can parse. `status: "answer"` keeps the raw-source fallback out of the picture,
    because a fallback puts turns in the context that retrieval did not choose and the
    comparison would stop being about retrieval.
    """

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, *, role, model, prompt, system=None, schema=None, **_):
        import json

        from llm_long_term_memory.llm.client import Completion

        self.calls += 1
        return Completion(
            text=json.dumps(
                {"status": "answer", "answer": "stub", "reason": "", "source_query": ""}
            ),
            model=model,
            input_tokens=0,
            output_tokens=0,
            thinking_tokens=0,
            attempts=1,
        )


def parse_sweep(values: list[str]) -> list[dict]:
    """`top_k=10,20,40 rerank=false,true` -> one configuration per combination."""
    from itertools import product

    axes: list[tuple[str, list]] = []
    for item in values:
        name, _, raw = item.partition("=")
        options: list = []
        for piece in raw.split(","):
            piece = piece.strip()
            if piece.lower() in ("true", "false"):
                options.append(piece.lower() == "true")
            else:
                try:
                    options.append(int(piece))
                except ValueError:
                    options.append(piece)
        name = name.strip()
        if name not in AXES:
            raise ValueError(f"unknown axis {name!r}; expected one of {AXES}")
        axes.append((name, options))
    # A mutating axis cannot be swept in one process: the first value rewrites the store
    # and every later value then measures the store the first one left behind. Two
    # eviction caps reported identical numbers that way, and the second was a no-op
    # wearing the first one's result.
    mutating = {"decay_halflife", "evict_to"}
    for name, options in axes:
        if name in mutating and len(options) > 1:
            raise ValueError(
                f"{name!r} writes to the store, so it takes one value per run; "
                "sweep it by running once per value against a fresh copy"
            )
    if not axes:
        return [{}]
    return [
        dict(zip([name for name, _ in axes], combination, strict=True))
        for combination in product(*[options for _, options in axes])
    ]


def apply_configuration(runner, configuration: dict, now) -> None:
    """Set the axes that are not constructor arguments.

    `top_k` and `rerank` go through `_build` because they change how the runner is
    assembled. The rest are plain attributes on an already-built retriever, and setting
    them here keeps one code path for every sweep rather than six near-identical configs.
    """
    if "recency" in configuration:
        runner.retriever.weights = {
            **runner.retriever.weights,
            "recency": float(configuration["recency"]),
        }
    if "halflife" in configuration:
        runner.retriever.recency_halflife_days = float(configuration["halflife"])
    if "strength" in configuration:
        runner.retriever.use_strength = bool(configuration["strength"])
    if "evict_to" in configuration:
        from llm_long_term_memory.lifecycle import evict_to_limit

        for namespace in runner.store.user_ids():
            evict_to_limit(runner.store, namespace, limit=int(configuration["evict_to"]))
    if "decay_halflife" in configuration:
        from llm_long_term_memory.lifecycle import apply_decay

        runner.decay_enabled = True
        runner.decay_halflife_days = float(configuration["decay_halflife"])
        for namespace in runner.store.user_ids():
            apply_decay(
                runner.store,
                namespace,
                now=now,
                halflife_days=float(configuration["decay_halflife"]),
            )


def ingested_namespaces(runner) -> set[str]:
    """Which conversations the store actually holds.

    A half-built store is the normal state during a multi-day ingest, and questions whose
    conversation has not been ingested retrieve nothing. Counting those as recall failures
    would read a partial ingest as a broken retriever.
    """
    return set(runner.store.user_ids())


def replay(runner, instances, coverage_by_question=None) -> dict:
    """Answer every instance with a stubbed answerer and read the retrieval facts off the
    rows it would have written."""
    rows = []
    for instance in instances:
        answer = runner.answer(instance)
        notes = answer.notes
        rows.append(
            {
                "question_id": instance.question_id,
                "any_source_session": bool((notes.get("recall_stages") or {}).get("selected")),
                "all_source_sessions": (notes.get("recall_coverage") or {}).get("selected") == 1.0,
                "ranked": list(notes.get("ranked_memory_ids") or []),
                "selected": [hit["memory_id"] for hit in notes.get("retrieval") or []],
                "context_tokens": answer.context_tokens,
                "has_evidence": bool(instance.answer_session_ids),
            }
        )
    scored = [row for row in rows if row["has_evidence"]]
    return {
        "questions": len(rows),
        "questions_with_labelled_evidence": len(scored),
        "any_source_session": fmean(r["any_source_session"] for r in scored) if scored else None,
        "all_source_sessions": fmean(r["all_source_sessions"] for r in scored) if scored else None,
        "median_context_tokens": sorted(r["context_tokens"] for r in rows)[len(rows) // 2]
        if rows
        else None,
        "mean_memories_in_context": fmean(len(r["selected"]) for r in rows) if rows else None,
        "mean_memories_ranked": fmean(len(r["ranked"]) for r in rows) if rows else None,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True, help="store filename stem")
    parser.add_argument("--questions", required=True, help="manifest of question ids")
    parser.add_argument("--variant", default="two_stage_fallback")
    parser.add_argument("--config", default="configs/v2.yaml")
    parser.add_argument("--limit", type=int, default=None, help="first N questions, for a smoke")
    parser.add_argument(
        "--sweep",
        nargs="*",
        default=[],
        help="axes to cross, e.g. top_k=10,20,40 rerank=false,true",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    os.environ.setdefault("GEMINI_API_KEY", "replay-no-request-is-sent")

    from llm_long_term_memory.cli import _build, _manifest_instances
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.evaluation.manifest import load_manifest

    settings = Settings()
    manifest = load_manifest(args.questions)
    by_id = {i.question_id: i for i in _manifest_instances(manifest, settings)}
    instances = [by_id[q] for q in manifest.question_ids if q in by_id]
    if args.limit:
        instances = instances[: args.limit]

    configurations = parse_sweep(args.sweep)
    results = []
    for configuration in configurations:
        # A fresh fake per configuration; `need_source_percent=0` keeps the raw-source
        # fallback out of the picture, because a fallback would put turns in the context
        # that retrieval did not choose and the comparison would stop being about retrieval.
        fake = StubAnswerer()
        _cfg, _settings, runner, _judge, _usage = _build(
            args.variant,
            args.config,
            store_name=args.store,
            top_k=configuration.get("top_k"),
            rerank=configuration.get("rerank"),
            client_override=fake,
            # Decay writes strengths, so a sweep that decays cannot hold the store open
            # read-only. Everything else can, and does.
            read_only_store=not ({"decay_halflife", "evict_to"} & set(configuration)),
        )
        try:
            from datetime import datetime

            apply_configuration(runner, configuration, datetime.now())
            present = ingested_namespaces(runner)
            chosen = [i for i in instances if i.store_namespace in present]
            skipped = len(instances) - len(chosen)
            outcome = replay(runner, chosen)
        finally:
            runner.store.close()
        calls = fake.calls
        if calls != len(chosen):
            # One stubbed answerer call per question and nothing else. Anything more means
            # a second pass ran and the row is not a pure retrieval measurement.
            print(f"STOP: {calls} model calls for {len(chosen)} questions")
            return 1
        results.append(
            {
                "configuration": configuration or {"default": True},
                "questions_skipped_not_yet_ingested": skipped,
                **{k: v for k, v in outcome.items() if k != "rows"},
            }
        )

    payload = {
        "name": "retrieval-replay",
        "provider_calls": 0,
        "store": args.store,
        "manifest": manifest.name,
        "variant": args.variant,
        "questions": len(instances),
        "measures": "retrieval only; whether the model uses what it is given is out of reach here",
        "configurations": results,
    }
    out = Path(args.out) if args.out else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"{payload['questions']} questions · {payload['store']} · zero provider calls\n")
    header = f"{'configuration':28} {'any src':>8} {'all src':>8} {'ctx tok':>8}"
    print(header)
    for row in results:
        label = ", ".join(f"{k}={v}" for k, v in row["configuration"].items())
        any_src = f"{row['any_source_session']:.1%}" if row["any_source_session"] else "n/a"
        all_src = f"{row['all_source_sessions']:.1%}" if row["all_source_sessions"] else "n/a"
        print(f"{label:28} {any_src:>8} {all_src:>8} {row['median_context_tokens']:>8,}")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
