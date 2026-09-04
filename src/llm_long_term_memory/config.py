"""Configuration.

Every knob that an ablation varies lives here and is loaded from a YAML file in
`configs/`, so a variant is a config file rather than a code branch. The retrieval
weights in particular must be data, not constants: the ablation table is produced by
running the same code against v1.yaml … v6.yaml.

Model IDs are deliberately not defaulted to anything real. They are discovered from
the provider's own model-list endpoint in P0 and pinned into config, because
hard-coding a version string from memory is how you end up calling a deprecated
model and silently changing your results mid-project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Secrets and paths, from the environment or a gitignored `.env`.

    Kept separate from `ExperimentConfig` on purpose: experiment configs are
    committed so that every row of the results table is reproducible, and a
    credential must never end up in one of them.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Unprefixed: GEMINI_API_KEY is the name the provider's own docs use, and the
    # one most likely already exported in the shell.
    gemini_api_key: str = ""

    # Prefixed, because DATA_DIR is too generic a name to claim from the environment.
    data_dir: Path = Field(default=Path("data"), validation_alias="LLTM_DATA_DIR")
    store_dir: Path = Field(default=Path("stores"), validation_alias="LLTM_STORE_DIR")
    results_dir: Path = Field(default=Path("results"), validation_alias="LLTM_RESULTS_DIR")

    @property
    def has_api_key(self) -> bool:
        return bool(self.gemini_api_key.strip())

    def require_api_key(self) -> str:
        if not self.has_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key "
                "(free, no card: https://aistudio.google.com/apikey), or export it in "
                "your shell. Run `lltm doctor` to check."
            )
        return self.gemini_api_key.strip()


class ModelConfig(BaseModel):
    """Three roles, three independently swappable models.

    They are separate fields because they have genuinely different requirements:
    the extractor is high-volume and cheap, the answerer is the thing under test and
    must be held constant across every variant, and the judge must be pinned for the
    life of the project or results from different weeks stop being comparable.
    """

    extractor: str = ""
    answerer: str = ""
    judge: str = ""
    embedder: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384


class QuotaConfig(BaseModel):
    rpm: int = 10
    tpm: int = 250_000
    rpd: int = 1_500
    max_wait_seconds: float = 300.0


class IngestConfig(BaseModel):
    sessions_per_request: int = 10
    dedupe_sessions: bool = True
    dedupe_similarity_threshold: float = 0.92
    two_stage: bool = False
    """Stage A emits bare fact strings and Stage B structures them with rules,
    instead of asking for nine fields per memory in one call. A config flag rather
    than a replacement so the two can be compared on the same sessions."""
    checkpoint_every: int = 25


class RetrievalWeights(BaseModel):
    """The five hybrid-retrieval signals. Setting a weight to 0.0 disables that
    signal, which is exactly how the ablation rows are produced."""

    semantic: float = 1.0
    bm25: float = 0.0
    recency: float = 0.0
    importance: float = 0.0
    entity: float = 0.0

    def enabled(self) -> list[str]:
        return [k for k, v in self.model_dump().items() if v > 0]


class RerankConfig(BaseModel):
    """Cross-encoder reranking between hybrid recall and top-k truncation.

    Off by default so that enabling it is one ablation row rather than a silent
    change to every existing result. Runs locally, so unlike almost everything else
    in this project it costs no API quota — only latency, which the results table
    already reports a p95 column for.
    """

    enabled: bool = False
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidates: int = 50
    """How many hybrid-ranked candidates get rescored. Bounds the added latency."""
    batch_size: int = 32


class RetrievalConfig(BaseModel):
    weights: RetrievalWeights = Field(default_factory=RetrievalWeights)
    candidate_limit: int = 50
    top_k: int = 20
    recency_halflife_days: float = 30.0
    rerank: RerankConfig = Field(default_factory=RerankConfig)


class PackConfig(BaseModel):
    enabled: bool = False
    token_budget: int = 2_000
    utility_model_path: str | None = None
    # Floors guarantee a slice of the budget per memory type so that a flood of
    # high-scoring episodic memories cannot evict the user profile entirely.
    type_floors: dict[str, float] = Field(
        default_factory=lambda: {"profile": 0.10, "semantic": 0.30}
    )


class DecayConfig(BaseModel):
    enabled: bool = False
    halflife_days: float = 60.0
    reinforcement: float = 0.30
    max_memories_per_user: int = 0  # 0 = no eviction


class ConsolidationConfig(BaseModel):
    similarity_threshold: float = 0.84
    min_cluster_size: int = 3
    source_strength_multiplier: float = 0.5


class HydrationConfig(BaseModel):
    neighbouring_sentences: int = 1
    max_tokens: int = 800
    allocation: Literal["ranked", "session_fair"] = "ranked"


class FallbackConfig(BaseModel):
    """Raw-conversation recovery, used only when structured memory is insufficient.

    Distinct from `hydration`, which attaches source evidence to *every* answer:
    that was measured at 3x the context for no detectable gain. This pays for raw
    text only when the answerer reports it cannot answer without it.
    """

    enabled: bool = False
    max_turns: int = 3
    max_chars: int = 2400


class ContextConfig(BaseModel):
    """Session-coherent context assembly, off unless a variant asks for it.

    These live in the config rather than behind a CLI flag on purpose. The
    pre-registration freezes the session budget on `train150` before `dev100` is
    touched, and `scripts/freeze.py` hashes the config — a budget passed on the
    command line would be a variable the freeze record cannot see.
    """

    max_sessions: int = 3
    window_radius: int | None = None
    max_total_memories: int | None = 20
    aggregate: str = "sum_top3"
    session_order: str = "chronological"
    include_superseded: bool = False


class ServiceConfig(BaseModel):
    """Defaults for the running service, deliberately separate from the benchmark's.

    `retrieval.top_k` is an experiment variable — changing it changes what a results
    row means. The service needs its own knob so the product can ship the value the
    pilot favoured without silently re-defining every benchmark variant.

    Whatever is set here is the single source of truth for *both* the inspector and
    the live answerer. They previously disagreed (10 versus 20), so the panel
    explaining "why these memories" was describing a different set from the one the
    answerer actually received — an explainability tool that does not explain the
    thing it sits next to.
    """

    top_k: int = 10
    """Pilot default: k=10 matched naive RAG on 242 median context tokens where k=20
    spent 439 for no measurable gain (results/rerank-pareto.md). Not frozen — 31
    questions cannot settle it."""


class ExperimentConfig(BaseModel):
    """One ablation variant."""

    name: str = "baseline"
    description: str = ""
    models: ModelConfig = Field(default_factory=ModelConfig)
    quota: QuotaConfig = Field(default_factory=QuotaConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    pack: PackConfig = Field(default_factory=PackConfig)
    decay: DecayConfig = Field(default_factory=DecayConfig)
    consolidation_config: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    hydration: HydrationConfig = Field(default_factory=HydrationConfig)
    fallback: FallbackConfig = Field(default_factory=FallbackConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)

    temporal_resolution: bool = False
    consolidation: bool = False

    dataset_variant: str = "s"
    dataset_limit: int | None = 50  # dev subset; None = all 500

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        data: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.model_dump(), sort_keys=False), encoding="utf-8")
