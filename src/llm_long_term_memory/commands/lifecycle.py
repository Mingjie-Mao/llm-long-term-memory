"""Memory lifecycle command group."""

from __future__ import annotations

import typer

from llm_long_term_memory.commands.common import console
from llm_long_term_memory.config import Settings

lifecycle_app = typer.Typer(help="P5: decay, eviction, and traceable consolidation")


@lifecycle_app.command("rebuild-index")
def lifecycle_rebuild_index(
    config: str = typer.Option("configs/fallback.yaml", "--config", "-c"),
    store_name: str = typer.Option("live", help="Memory store filename stem"),
) -> None:
    """Rebuild the derived vector index from every durable SQLite memory."""
    from llm_long_term_memory.config import ExperimentConfig
    from llm_long_term_memory.embed import Encoder
    from llm_long_term_memory.store import NumpyFlatIndex, SQLiteMemoryStore

    cfg = ExperimentConfig.from_yaml(config)
    settings = Settings()
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
        target = settings.store_dir / f"{store_name}-index"
        for suffix in (".npy", ".ids.json", ".manifest.json"):
            temp_stem.with_suffix(suffix).replace(target.with_suffix(suffix))
        console.print(f"[green]✓[/green] rebuilt {len(memories):,} vectors for {store_name}")
    finally:
        store.close()
        for suffix in (".npy", ".ids.json", ".manifest.json"):
            temp_stem.with_suffix(suffix).unlink(missing_ok=True)


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
