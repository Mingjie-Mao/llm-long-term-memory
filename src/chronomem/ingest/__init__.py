from .dedup import Deduplicator, DedupOutcome
from .extract import ExtractionOutcome, Extractor, memory_id
from .extract_facts import FactExtractor
from .pipeline import (
    IngestionPipeline,
    IngestOutcome,
    IngestProgress,
    group_by_namespace,
    namespaced_sessions,
)
from .provenance import attach_source_span, source_span_for
from .schemas import (
    SINGLE_VALUED_PREDICATES,
    DedupDecision,
    ExtractedMemory,
    ExtractionResult,
    is_single_valued,
    normalize_predicate,
)
from .two_stage import TwoStageExtractor

__all__ = [
    "SINGLE_VALUED_PREDICATES",
    "DedupDecision",
    "DedupOutcome",
    "Deduplicator",
    "ExtractedMemory",
    "ExtractionOutcome",
    "ExtractionResult",
    "Extractor",
    "FactExtractor",
    "IngestOutcome",
    "IngestProgress",
    "IngestionPipeline",
    "TwoStageExtractor",
    "attach_source_span",
    "group_by_namespace",
    "is_single_valued",
    "memory_id",
    "namespaced_sessions",
    "normalize_predicate",
    "source_span_for",
]
