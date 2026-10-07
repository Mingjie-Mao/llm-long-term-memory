"""LLTM command line.

P0 commands are the ones that need no API key: download the benchmark, measure it,
and plan the ingestion budget against the free-tier quota.
"""

from __future__ import annotations

from pathlib import Path

import typer

from llm_long_term_memory.commands import eval_execution, eval_reporting
from llm_long_term_memory.commands.common import console, manifest_instances
from llm_long_term_memory.commands.data import data_app
from llm_long_term_memory.commands.doctor import mask_secret, render_doctor
from llm_long_term_memory.commands.influence import create_influence_app
from llm_long_term_memory.commands.ingest_coverage import ingest_coverage
from llm_long_term_memory.commands.ingest_fidelity import ingest_fidelity
from llm_long_term_memory.commands.ingest_run import ingest_run
from llm_long_term_memory.commands.ingest_temporal import ingest_temporal_gate
from llm_long_term_memory.commands.lifecycle import lifecycle_app
from llm_long_term_memory.commands.service import mcp_serve, resolve
from llm_long_term_memory.config import Settings

app = typer.Typer(add_completion=False, help="LLTM — long-term memory for LLM agents")
app.add_typer(data_app, name="data")


def _mask(secret: str) -> str:
    """Enough to confirm which key is loaded, not enough to leak it."""
    return mask_secret(secret)


@app.command()
def doctor() -> None:
    """Check that credentials and data are where the pipeline expects them."""
    render_doctor(Settings())


eval_app = typer.Typer(help="Run and report evaluations")
app.add_typer(eval_app, name="eval")


# The budget the offline gate measured for question-found turns: 78.1% all-gold coverage
# on train150 at 4,000 tokens (results/memory-as-index-offline-decision.md). Fixed by
# registration, not tuned.
RAW_PRIMARY_TOKENS = 4000
RAW_PRIMARY_V2_OUTPUT_TOKENS = 1024
GROUNDED_VARIANTS = {
    "two_stage_raw_primary_grounded",
    "two_stage_raw_primary_grounded_v2",
    "two_stage_raw_primary_grounded_v3",
    "two_stage_raw_primary_grounded_v4",
    "two_stage_raw_primary_grounded_v5",
    "two_stage_raw_primary_grounded_v6",
    "two_stage_raw_primary_grounded_v7",
    "two_stage_raw_primary_grounded_v8",
    "two_stage_raw_primary_grounded_v9",
    "two_stage_raw_primary_grounded_v10",
    "two_stage_raw_primary_grounded_v11",
    "two_stage_raw_primary_grounded_v12",
    "two_stage_raw_primary_grounded_v13",
    "two_stage_raw_primary_grounded_v14",
    "two_stage_raw_primary_grounded_v15",
    "two_stage_raw_primary_grounded_v16",
    "two_stage_raw_primary_grounded_v17",
    "two_stage_raw_primary_grounded_v18",
}


def _turn_index(settings, stem: str, cfg, kind: str = "turn-index"):
    """The dense index over this store's turns, which the v2 and v4 arms cannot run without."""
    from llm_long_term_memory.store import NumpyFlatIndex

    index = NumpyFlatIndex(settings.store_dir / f"{stem}-{kind}", dim=cfg.models.embedding_dim)
    if not len(index):
        flag = " --fact-keys" if kind == "turn-key-index" else ""
        raise typer.BadParameter(
            f"{stem!r} has no {kind} — run `lltm lifecycle build-turn-index "
            f"--store-name {stem}{flag}` first"
        )
    return index


def _default_store_name(variant: str) -> str:
    """Keep new extraction variants physically separate from frozen v1 data."""
    return "two-stage" if variant.startswith("two_stage") else "memories"


def _relation_scanner(store, encoder):
    """Build the v4.1 routed scanner, or fail loudly if its mapping is absent.

    Returning None on a missing map would silently produce an arm identical to the one
    it is meant to be compared against, and the comparison would report no difference for
    the wrong reason.
    """
    import csv

    from llm_long_term_memory.retrieve.relation_router import RelationRouter
    from llm_long_term_memory.retrieve.scan import RelationScanner

    mapping = Path(__file__).resolve().parents[2] / "results/analysis/predicate-map.csv"
    if not mapping.is_file():
        raise typer.BadParameter(f"two_stage_synthesis_scan needs {mapping}, which does not exist")
    with open(mapping, encoding="utf-8") as handle:
        relation_of = {r["predicate_raw"]: r["relation_type"] for r in csv.DictReader(handle)}
    return RelationScanner(store, RelationRouter(encoder), relation_of)


# `tools/retrieval_replay.py` resolves a manifest through the CLI's own loader, so
# the name stays exported here even though the implementation moved.
_manifest_instances = manifest_instances


def _build(
    variant: str,
    cfg_path: str,
    store_name: str | None = None,
    top_k: int | None = None,
    rerank: bool | None = None,
    *,
    client_override=None,
    settings_override=None,
    read_only_store: bool = False,
):
    """Wire up client, judge, and runner for one variant.

    `top_k` and `rerank` override the config so that an accuracy-vs-context sweep is
    a loop over flags rather than six near-identical YAML files. Anything reported
    as a table row should still come from a config, not from a flag.
    """
    from llm_long_term_memory.config import ExperimentConfig
    from llm_long_term_memory.embed import Encoder
    from llm_long_term_memory.evaluation.judge import Judge
    from llm_long_term_memory.evaluation.runners.full_context import FullContextRunner
    from llm_long_term_memory.evaluation.runners.memory import MemoryRunner
    from llm_long_term_memory.evaluation.runners.naive_rag import NaiveRAGRunner
    from llm_long_term_memory.llm import Limits, QuotaManager, UsageTracker
    from llm_long_term_memory.llm.client import GeminiClient

    settings = settings_override or Settings()
    cfg = ExperimentConfig.from_yaml(cfg_path)
    if top_k is not None:
        cfg.retrieval.top_k = top_k
    if rerank is not None:
        cfg.retrieval.rerank.enabled = rerank

    quota = QuotaManager(
        state_dir=settings.store_dir / "quota",
        default=Limits(rpm=cfg.quota.rpm, tpm=cfg.quota.tpm, rpd=cfg.quota.rpd),
    )
    learned = quota.load_learned()
    if learned:
        console.print(
            "[dim]observed limits: "
            + ", ".join(f"{m} rpd={lim.rpd} rpm={lim.rpm}" for m, lim in sorted(learned.items()))
            + "[/dim]"
        )
    usage = UsageTracker()
    client = client_override or GeminiClient(settings.require_api_key(), quota=quota, usage=usage)
    judge = Judge(client, model=cfg.models.judge)

    if variant == "full_context":
        runner = FullContextRunner(client, model=cfg.models.answerer)
    elif variant == "naive_rag":
        runner = NaiveRAGRunner(client, model=cfg.models.answerer, encoder=Encoder())
    elif variant in (
        "chronomem",
        "chronomem_no_temporal",
        "two_stage",
        "two_stage_no_temporal",
        "two_stage_hydrated",
        "two_stage_hydrated_no_temporal",
        # A3. Same store and same pipeline; the only difference is
        # `retrieval.rerank.enabled` in the config, so the row is attributable.
        "two_stage_hydrated_rerank",
        # Memory first, raw source only when the answerer says it needs it.
        "two_stage_fallback",
        # P6. Same store, same retrieval, same fallback; the memories are grouped
        # into whole sessions in event order rather than handed over as a ranking.
        # The only difference from `two_stage_fallback` is the context's shape.
        "two_stage_coherent",
        # Validation-only ceiling: session ids come from benchmark gold labels.
        # It can explain a null but is never a product variant.
        "two_stage_coherent_oracle",
        # Structured memory with the raw archive switched off, whatever the config
        # says. The existing names cannot express this: `two_stage` and
        # `two_stage_fallback` both read `fallback.enabled`, so "memory only" was
        # only ever reachable by editing the config — which a single-config freeze
        # forbids. Reporting it as a split of the fallback arm is not the same
        # measurement either: in that arm the answerer knows it may ask for source.
        "two_stage_memory_only",
        # v2c: deterministic query-time repairs over the completed v2b store. The
        # memory-only arm isolates update/temporal assembly; the hybrid arm also
        # hydrates source text when structured precision is insufficient.
        "two_stage_v2c_memory_only",
        "two_stage_v2c",
        # v2d: v2c evidence planning plus source-labelled operands and deterministic
        # multi-session counting/arithmetic.
        "two_stage_v2d_memory_only",
        "two_stage_v2d",
        # v5.0. Everything v2c does, plus verbatim turns from the sessions retrieval
        # already selected, attached before the first answer call instead of after a
        # refusal. `_fixed` spends the same budget on every question; `_planned` spends
        # by what the question asks for. The registered comparison for both is the v2c
        # arm, so the raw windows are the only difference.
        "two_stage_v5_fixed",
        "two_stage_v5_planned",
        # v2e. Everything v2c does, with one change: a fact that states no date is
        # shown as "mentioned <date>" rather than "since <date>", so a mention date
        # stops reading as the day the fact became true. Requires a store whose
        # `event_time` means what it says — see tools/backfill_event_time.py.
        "two_stage_v2e",
        # Raw-primary (results/prereg-raw-primary-heldout100-v1.md). Everything
        # `two_stage_hydrated` does, plus verbatim turns found in the whole archive by
        # the question, before the first answer call. `_raw_only` keeps only those
        # turns: no memory context and no memory-located fallback.
        "two_stage_raw_primary",
        "two_stage_raw_only",
        # Raw-primary t1 (results/prereg-raw-primary-time-notes-v1.md): the same arm with
        # each conversation header dated relative to the question, and the relative
        # dates inside its turns resolved from that conversation's date.
        "two_stage_raw_primary_t1",
        # Raw-primary t2: t1 with the answerer told that the latest value wins, and
        # counts or totals answered item by item (answer policy `v2_cl`).
        "two_stage_raw_primary_t2",
        # Raw-primary t3: t2 with questions about dates, durations or order also answered
        # from notes (answer policy `v2_clt`; results/prereg-t-probe-v2.md).
        "two_stage_raw_primary_t3",
        # Raw-primary t4: t3 with the user's turns packed first and the period a question
        # names searched by time (results/prereg-raw-primary-t3-t4-heldout100-v1.md).
        "two_stage_raw_primary_t4",
        # Raw-primary v2 (results/prereg-raw-primary-dev100-v3.md): the same arm with
        # BM25 fused with a dense turn ranking, conversation dates written relative to
        # the question, and an answerer that lists its evidence before deciding.
        "two_stage_raw_primary_v2",
        # Raw-primary v4 (results/prereg-raw-primary-dev100-v4.md): v2 plus the turns
        # the retrieved memories point at as a third ranking, and turns keyed by their
        # text plus the facts anchored to them (lexical and dense).
        "two_stage_raw_primary_v4",
        "two_stage_raw_primary_grounded",
        "two_stage_raw_primary_grounded_v2",
        "two_stage_raw_primary_grounded_v3",
        "two_stage_raw_primary_grounded_v4",
        "two_stage_raw_primary_grounded_v5",
        "two_stage_raw_primary_grounded_v6",
        "two_stage_raw_primary_grounded_v7",
        "two_stage_raw_primary_grounded_v8",
        "two_stage_raw_primary_grounded_v9",
        "two_stage_raw_primary_grounded_v10",
        "two_stage_raw_primary_grounded_v11",
        "two_stage_raw_primary_grounded_v12",
        "two_stage_raw_primary_grounded_v13",
        "two_stage_raw_primary_grounded_v14",
        "two_stage_raw_primary_grounded_v15",
        "two_stage_raw_primary_grounded_v16",
        "two_stage_raw_primary_grounded_v17",
        "two_stage_raw_primary_grounded_v18",
        # v3 research arm: same v2 store/retrieval/fallback, with an explicit
        # evidence/check/calculation answer policy. It never uses benchmark labels.
        "two_stage_reasoned",
        # v3.2: hydrate compact source spans only for time, aggregation and current-
        # state reasoning. Direct lookups keep the short v2 context.
        "two_stage_reasoned_evidence",
        # v4.0: everything v3.3 does on the retrieval side, unchanged, plus an
        # answerer that names its operation before deciding whether it can answer
        # and hands count/duration arithmetic to Python. The registered comparison
        # is against `two_stage_reasoned_evidence`, so nothing else may differ.
        "two_stage_synthesis",
        # v4.0 attempt 2: the same answerer minus the timeline rendering, so that
        # `current_state` — which deterministic arithmetic never touches — attributes
        # its movement to the prompt alone.
        "two_stage_synthesis_flat",
        # v4.1: the flat v4 answerer plus a routed exhaustive relation scan. Registered
        # separately because it changes *retrieval*, which inverts Gate 0 — retrieval
        # must now be identical only on the questions the router declined.
        "two_stage_synthesis_scan",
        # v4.2: v4.0-flat's answerer, with count members cited by the label of the
        # memory they come from instead of written as free text. The registered
        # comparison is against `two_stage_synthesis_flat`, so retrieval, hydration and
        # every other operation are unchanged.
        "two_stage_synthesis_enumerate",
    ):
        from llm_long_term_memory.retrieve import SessionBudget
        from llm_long_term_memory.store import NumpyFlatIndex, SQLiteMemoryStore

        utility_model = None
        if cfg.pack.enabled and cfg.pack.utility_model_path:
            from llm_long_term_memory.influence import UtilityPredictor

            utility_model = UtilityPredictor.load(cfg.pack.utility_model_path)

        stem = store_name or _default_store_name(variant)
        store = SQLiteMemoryStore(settings.store_dir / f"{stem}.db", read_only=read_only_store)
        store.initialize()
        index = NumpyFlatIndex(settings.store_dir / f"{stem}-index", dim=cfg.models.embedding_dim)
        if not len(index):
            raise typer.BadParameter(
                f"the {stem!r} memory store is empty — run `lltm ingest run "
                f"--store-name {stem}` first"
            )
        try:
            index.validate_ids(store.memory_ids())
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        reranker = None
        if cfg.retrieval.rerank.enabled:
            from llm_long_term_memory.retrieve import CrossEncoderReranker

            reranker = CrossEncoderReranker(
                model_name=cfg.retrieval.rerank.model,
                candidates=cfg.retrieval.rerank.candidates,
                batch_size=cfg.retrieval.rerank.batch_size,
            )
        # One encoder for the whole arm. The scanner's router embeds its class glosses
        # with the same model the retriever embeds questions with, so handing it a second
        # instance loaded a second copy of MiniLM onto the device for no behavioural
        # difference — and on a routed arm that is the difference between one model in
        # memory and two.
        encoder = Encoder()
        runner = MemoryRunner(
            client,
            model=cfg.models.answerer,
            encoder=encoder,
            store=store,
            index=index,
            top_k=cfg.retrieval.top_k,
            temporal=not variant.endswith("_no_temporal"),
            retrieval_weights=cfg.retrieval.weights.model_dump(),
            candidate_limit=cfg.retrieval.candidate_limit,
            recency_halflife_days=cfg.retrieval.recency_halflife_days,
            decay_enabled=cfg.decay.enabled,
            decay_halflife_days=cfg.decay.halflife_days,
            reinforcement=cfg.decay.reinforcement,
            evidence_hydration="_hydrated" in variant
            or variant
            in {
                "two_stage_raw_primary",
                "two_stage_raw_primary_t1",
                "two_stage_raw_primary_t2",
                "two_stage_raw_primary_t3",
                "two_stage_raw_primary_t4",
                "two_stage_raw_primary_v2",
                "two_stage_raw_primary_v4",
            },
            adaptive_reasoning_hydration=variant
            in {
                "two_stage_reasoned_evidence",
                "two_stage_synthesis",
                "two_stage_synthesis_flat",
                "two_stage_synthesis_scan",
                "two_stage_synthesis_enumerate",
            },
            hydration_neighbouring_sentences=cfg.hydration.neighbouring_sentences,
            hydration_max_tokens=cfg.hydration.max_tokens,
            hydration_allocation=cfg.hydration.allocation,
            token_budget=cfg.pack.token_budget if cfg.pack.enabled else 0,
            utility_model=utility_model,
            type_floors=cfg.pack.type_floors if cfg.pack.enabled else None,
            reranker=reranker,
            # The variant name wins over the config here, so a memory-only arm can
            # sit in the same freeze as the arms it is compared against.
            raw_fallback=cfg.fallback.enabled
            and variant
            not in {
                "two_stage_memory_only",
                "two_stage_v2c_memory_only",
                "two_stage_v2d_memory_only",
                "two_stage_raw_only",
            },
            raw_fallback_max_turns=cfg.fallback.max_turns,
            raw_fallback_max_chars=cfg.fallback.max_chars,
            session_budget=SessionBudget(
                max_sessions=cfg.context.max_sessions,
                window_radius=cfg.context.window_radius,
                max_total_memories=cfg.context.max_total_memories,
                aggregate=cfg.context.aggregate,
                session_order=cfg.context.session_order,
                include_superseded=cfg.context.include_superseded,
            )
            if variant in {"two_stage_coherent", "two_stage_coherent_oracle"}
            else None,
            oracle_session_context=variant == "two_stage_coherent_oracle",
            # The fixed arm spends exactly what the conditional path spends
            # (`fallback.max_turns`), so the only difference from v2c is *when* the
            # raw turns arrive, not how many.
            parallel_raw_windows=(
                cfg.fallback.max_turns if variant.startswith("two_stage_v5") else 0
            ),
            parallel_raw_planned=variant == "two_stage_v5_planned",
            raw_primary_tokens=RAW_PRIMARY_TOKENS
            if variant
            in {
                "two_stage_raw_primary",
                "two_stage_raw_primary_t1",
                "two_stage_raw_primary_t2",
                "two_stage_raw_primary_t3",
                "two_stage_raw_primary_t4",
                "two_stage_raw_only",
                "two_stage_raw_primary_v2",
                "two_stage_raw_primary_v4",
                "two_stage_raw_primary_grounded",
                "two_stage_raw_primary_grounded_v2",
                "two_stage_raw_primary_grounded_v3",
                "two_stage_raw_primary_grounded_v4",
                "two_stage_raw_primary_grounded_v5",
                "two_stage_raw_primary_grounded_v6",
                "two_stage_raw_primary_grounded_v7",
                "two_stage_raw_primary_grounded_v8",
                "two_stage_raw_primary_grounded_v9",
                "two_stage_raw_primary_grounded_v10",
                "two_stage_raw_primary_grounded_v11",
                "two_stage_raw_primary_grounded_v12",
                "two_stage_raw_primary_grounded_v13",
                "two_stage_raw_primary_grounded_v14",
                "two_stage_raw_primary_grounded_v15",
                "two_stage_raw_primary_grounded_v16",
                "two_stage_raw_primary_grounded_v17",
                "two_stage_raw_primary_grounded_v18",
            }
            else 0,
            raw_primary_only=variant == "two_stage_raw_only",
            raw_primary_turn_index=_turn_index(settings, stem, cfg)
            if variant == "two_stage_raw_primary_v2"
            else _turn_index(settings, stem, cfg, kind="turn-key-index")
            if variant in {"two_stage_raw_primary_v4"} | GROUNDED_VARIANTS
            else None,
            raw_primary_dated=variant
            in {
                "two_stage_raw_primary_t1",
                "two_stage_raw_primary_t2",
                "two_stage_raw_primary_t3",
                "two_stage_raw_primary_t4",
                "two_stage_raw_primary_v2",
                "two_stage_raw_primary_v4",
            },
            raw_primary_prefer_user=variant == "two_stage_raw_primary_t4",
            raw_primary_time_window=variant == "two_stage_raw_primary_t4",
            raw_primary_time_notes=variant
            in {
                "two_stage_raw_primary_t1",
                "two_stage_raw_primary_t2",
                "two_stage_raw_primary_t3",
                "two_stage_raw_primary_t4",
            },
            raw_primary_memory_fusion=variant in {"two_stage_raw_primary_v4"} | GROUNDED_VARIANTS,
            raw_primary_fact_keys=variant in {"two_stage_raw_primary_v4"} | GROUNDED_VARIANTS,
            # Listing the evidence first spends output; 512 would truncate the JSON.
            max_output_tokens=2048
            if variant in GROUNDED_VARIANTS
            else RAW_PRIMARY_V2_OUTPUT_TOKENS
            if variant in {"two_stage_raw_primary_v2", "two_stage_raw_primary_v4"}
            else 512,
            date_provenance=variant == "two_stage_v2e",
            answer_policy={
                "two_stage_reasoned": "reasoned_v3",
                "two_stage_reasoned_evidence": "reasoned_v3",
                "two_stage_synthesis": "synthesis_v4",
                "two_stage_synthesis_flat": "synthesis_v4",
                "two_stage_synthesis_scan": "synthesis_v4",
                "two_stage_synthesis_enumerate": "synthesis_v4_enumerate",
                "two_stage_v2c_memory_only": "v2c",
                "two_stage_v2c": "v2c",
                "two_stage_v2d_memory_only": "v2d",
                "two_stage_v2d": "v2d",
                # v5.0 changes the evidence, not the answering policy: the comparison
                # against v2c is only meaningful while everything else is v2c.
                "two_stage_v5_fixed": "v2c",
                "two_stage_v5_planned": "v2c",
                # v2e changes the rendering, not the policy: the comparison against
                # v2c is only meaningful while everything else is v2c.
                "two_stage_v2e": "v2c",
                "two_stage_raw_primary_v2": "v2_notes",
                "two_stage_raw_primary_v4": "v2_notes",
                "two_stage_raw_primary_t2": "v2_cl",
                "two_stage_raw_primary_t3": "v2_clt",
                "two_stage_raw_primary_t4": "v2_clt",
                "two_stage_raw_primary_grounded": "grounded_v1",
                "two_stage_raw_primary_grounded_v2": "grounded_v2",
                "two_stage_raw_primary_grounded_v3": "grounded_v3",
                "two_stage_raw_primary_grounded_v4": "grounded_v4",
                "two_stage_raw_primary_grounded_v5": "grounded_v5",
                "two_stage_raw_primary_grounded_v6": "grounded_v6",
                "two_stage_raw_primary_grounded_v7": "grounded_v7",
                "two_stage_raw_primary_grounded_v8": "grounded_v8",
                "two_stage_raw_primary_grounded_v9": "grounded_v9",
                "two_stage_raw_primary_grounded_v10": "grounded_v10",
                "two_stage_raw_primary_grounded_v11": "grounded_v11",
                "two_stage_raw_primary_grounded_v12": "grounded_v12",
                "two_stage_raw_primary_grounded_v13": "grounded_v13",
                "two_stage_raw_primary_grounded_v14": "grounded_v14",
                "two_stage_raw_primary_grounded_v15": "grounded_v15",
                "two_stage_raw_primary_grounded_v16": "grounded_v16",
                "two_stage_raw_primary_grounded_v17": "grounded_v17",
                "two_stage_raw_primary_grounded_v18": "grounded_v18",
            }.get(variant, "v2"),
            timeline_rendering=variant
            not in {
                "two_stage_synthesis_flat",
                "two_stage_synthesis_scan",
                "two_stage_synthesis_enumerate",
            },
            scanner=_relation_scanner(store, encoder)
            if variant == "two_stage_synthesis_scan"
            else None,
        )
        runner.name = variant
    else:
        raise typer.BadParameter(f"unknown variant {variant!r}")
    return cfg, settings, runner, judge, usage


eval_execution.register_run_commands(
    eval_app,
    build_runner=_build,
    manifest_instances=_manifest_instances,
)


eval_app.command("report")(eval_reporting.eval_report)
eval_app.command("label")(eval_reporting.eval_label)
eval_app.command("agreement")(eval_reporting.eval_agreement)
eval_app.command("failure-audit")(eval_reporting.eval_failure_audit)
eval_app.command("failure-report")(eval_reporting.eval_failure_report)


ingest_app = typer.Typer(help="Build the memory store from a corpus")
app.add_typer(ingest_app, name="ingest")


ingest_app.command("run")(ingest_run)
ingest_app.command("coverage")(ingest_coverage)
ingest_app.command("fidelity")(ingest_fidelity)


app.command("resolve")(resolve)
app.command("mcp")(mcp_serve)


eval_app.command("freeze")(eval_reporting.eval_freeze)
eval_app.command("compare")(eval_reporting.eval_compare)
eval_app.command("variability")(eval_reporting.eval_variability)
eval_execution.register_repeat_command(eval_app, build_runner=_build)


app.add_typer(lifecycle_app, name="lifecycle")


influence_app = create_influence_app(_build)
app.add_typer(influence_app, name="influence")


ingest_app.command("temporal-gate")(ingest_temporal_gate)


if __name__ == "__main__":
    app()
