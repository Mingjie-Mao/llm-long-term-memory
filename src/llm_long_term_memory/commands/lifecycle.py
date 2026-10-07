"""Memory lifecycle command group."""

from __future__ import annotations

import typer

from llm_long_term_memory.commands.common import cli_lock, console
from llm_long_term_memory.config import Settings

lifecycle_app = typer.Typer(help="P5: decay, eviction, and traceable consolidation")


@lifecycle_app.command("rebuild-index")
def lifecycle_rebuild_index(
    config: str = typer.Option("configs/fallback.yaml", "--config", "-c"),
    store_name: str = typer.Option("live", help="Memory store filename stem"),
) -> None:
    """Rebuild the derived vector index from every durable SQLite memory."""
    from llm_long_term_memory.config import ExperimentConfig

    cfg = ExperimentConfig.from_yaml(config)
    settings = Settings()
    # The writer lock, not only a read: a service still running on this store would save
    # its in-memory index over the rebuilt one on its next write.
    with cli_lock(settings.store_dir / f"{store_name}.db", what="index rebuild"):
        _rebuild_index(cfg, settings, store_name)


def _rebuild_index(cfg, settings: Settings, store_name: str) -> None:
    from llm_long_term_memory.embed import Encoder
    from llm_long_term_memory.store import NumpyFlatIndex, SQLiteMemoryStore

    store = SQLiteMemoryStore(settings.store_dir / f"{store_name}.db", read_only=True)
    store.initialize()
    memories = [memory for user_id in store.user_ids() for memory in store.iter_all(user_id)]
    temp_stem = settings.store_dir / f".{store_name}-index-rebuild"
    rebuilt = NumpyFlatIndex(temp_stem, dim=cfg.models.embedding_dim)
    try:
        if memories:
            vectors = Encoder(cfg.models.embedder).encode([memory.content for memory in memories])
            rebuilt.add([memory.id for memory in memories], vectors)
        rebuilt.save()
        rebuilt.validate_ids(store.memory_ids())
        promoted = rebuilt.promote(settings.store_dir / f"{store_name}-index")
        promoted.validate_ids(store.memory_ids())
        console.print(f"[green]✓[/green] rebuilt {len(memories):,} vectors for {store_name}")
    finally:
        store.close()
        for suffix in (".npy", ".ids.json", ".manifest.json"):
            temp_stem.with_suffix(suffix).unlink(missing_ok=True)


@lifecycle_app.command("build-turn-index")
def lifecycle_build_turn_index(
    config: str = typer.Option("configs/fallback.yaml", "--config", "-c"),
    store_name: str = typer.Option("live", help="Memory store filename stem"),
    fact_keys: bool = typer.Option(
        False,
        "--fact-keys",
        help="Embed each turn with the facts anchored to it (writes <store>-turn-key-index)",
    ),
) -> None:
    """Embed every raw turn with the local encoder, for dense turn retrieval.

    A derivative of the durable turns, like the memory index: no provider call, written
    to a temporary stem and swapped in whole, and taken under the store's writer lock.
    """
    from llm_long_term_memory.config import ExperimentConfig

    cfg = ExperimentConfig.from_yaml(config)
    settings = Settings()
    with cli_lock(settings.store_dir / f"{store_name}.db", what="turn index build"):
        count = _build_turn_index(cfg, settings, store_name, fact_keys=fact_keys)
    console.print(f"[green]✓[/green] embedded {count:,} turns for {store_name}")


def _build_turn_index(cfg, settings: Settings, store_name: str, fact_keys: bool = False) -> int:
    from llm_long_term_memory.embed import Encoder
    from llm_long_term_memory.retrieve.excerpts import fact_keyed_texts
    from llm_long_term_memory.store import NumpyFlatIndex, SQLiteMemoryStore

    store = SQLiteMemoryStore(settings.store_dir / f"{store_name}.db", read_only=True)
    store.initialize()
    try:
        turns = [
            turn
            for session_id in sorted(store.session_ids())
            for turn in store.turns_for_session(session_id)
        ]
        texts = {turn.id: turn.content for turn in turns}
        if fact_keys:
            # Keys are per user: a fact is only ever attached to its own user's turn.
            for user_id in store.user_ids():
                own = [
                    turn
                    for session_id in store.session_ids_for_user(user_id)
                    for turn in store.turns_for_session(session_id)
                ]
                texts.update(fact_keyed_texts(store, user_id, own))
    finally:
        store.close()
    suffix_stem = "turn-key-index" if fact_keys else "turn-index"
    temp_stem = settings.store_dir / f".{store_name}-{suffix_stem}-build"
    built = NumpyFlatIndex(temp_stem, dim=cfg.models.embedding_dim)
    try:
        if turns:
            vectors = Encoder(cfg.models.embedder).encode([texts[t.id] for t in turns])
            built.add([t.id for t in turns], vectors)
        built.save()
        built.promote(settings.store_dir / f"{store_name}-{suffix_stem}")
    finally:
        for suffix in (".npy", ".ids.json", ".manifest.json"):
            temp_stem.with_suffix(suffix).unlink(missing_ok=True)
    return len(turns)


@lifecycle_app.command("run")
def lifecycle_run(
    config: str = typer.Option("configs/baselines.yaml", "--config", "-c"),
    store_name: str = typer.Option("two-stage", help="Memory store filename stem"),
    user_id: str | None = typer.Option(None, help="Only process one namespace"),
) -> None:
    """Apply configured strength decay and optional capacity eviction."""
    from datetime import datetime

    from llm_long_term_memory.config import ExperimentConfig
    from llm_long_term_memory.lifecycle import apply_decay, evict_to_limit
    from llm_long_term_memory.store import SQLiteMemoryStore

    cfg = ExperimentConfig.from_yaml(config)
    settings = Settings()
    store = SQLiteMemoryStore(settings.store_dir / f"{store_name}.db")
    store.initialize()
    users = [user_id] if user_id else store.user_ids()
    now = datetime.now()
    try:
        for namespace in users:
            if cfg.decay.enabled:
                decay = apply_decay(
                    store,
                    namespace,
                    now=now,
                    halflife_days=cfg.decay.halflife_days,
                )
            else:
                decay = None
            eviction = (
                evict_to_limit(store, namespace, limit=cfg.decay.max_memories_per_user)
                if cfg.decay.max_memories_per_user
                else None
            )
            console.print(
                f"{namespace}: decayed={decay.updated if decay else 0} "
                f"evicted={len(eviction.evicted_ids) if eviction else 0}"
            )
    finally:
        store.close()


@lifecycle_app.command("consolidate")
def lifecycle_consolidate(
    config: str = typer.Option("configs/baselines.yaml", "--config", "-c"),
    store_name: str = typer.Option("two-stage", help="Memory store filename stem"),
    user_id: str | None = typer.Option(None, help="Only process one namespace"),
) -> None:
    """Synthesize related active memories while retaining their source evidence."""
    from llm_long_term_memory.config import ExperimentConfig
    from llm_long_term_memory.consolidate import Consolidator
    from llm_long_term_memory.embed import Encoder
    from llm_long_term_memory.llm import Limits, QuotaManager, UsageTracker
    from llm_long_term_memory.llm.client import GeminiClient
    from llm_long_term_memory.store import NumpyFlatIndex, SQLiteMemoryStore

    cfg = ExperimentConfig.from_yaml(config)
    settings = Settings()
    store = SQLiteMemoryStore(settings.store_dir / f"{store_name}.db")
    store.initialize()
    index = NumpyFlatIndex(settings.store_dir / f"{store_name}-index", dim=cfg.models.embedding_dim)
    quota = QuotaManager(
        state_dir=settings.store_dir / "quota",
        default=Limits(rpm=cfg.quota.rpm, tpm=cfg.quota.tpm, rpd=cfg.quota.rpd),
    )
    quota.load_learned()
    usage = UsageTracker()
    consolidator = Consolidator(
        GeminiClient(settings.require_api_key(), quota=quota, usage=usage),
        cfg.models.extractor,
        Encoder(),
        store,
        index,
        similarity_threshold=cfg.consolidation_config.similarity_threshold,
        min_cluster_size=cfg.consolidation_config.min_cluster_size,
        source_strength_multiplier=cfg.consolidation_config.source_strength_multiplier,
    )
    users = [user_id] if user_id else store.user_ids()
    try:
        for namespace in users:
            report = consolidator.consolidate(namespace)
            console.print(
                f"{namespace}: clusters={report.clusters_found} "
                f"created={len(report.memories_created)}"
            )
    finally:
        index.save()
        usage.save(settings.results_dir / "raw" / f"consolidate-{store_name}.usage.json")
        store.close()
