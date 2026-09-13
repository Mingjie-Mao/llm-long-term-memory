"""Ingestion public API, loaded on demand.

Importing ``ingest.schemas`` is needed by the temporal resolver and the lightweight
ONNX playground. The old eager re-exports pulled in the provider SDK and every
pipeline module just to read four predicate names. Keep the convenient package API
without making unrelated consumers install the LLM stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "Deduplicator": (".dedup", "Deduplicator"),
    "DedupOutcome": (".dedup", "DedupOutcome"),
    "ExtractionOutcome": (".extract", "ExtractionOutcome"),
    "Extractor": (".extract", "Extractor"),
    "memory_id": (".extract", "memory_id"),
    "FactExtractor": (".extract_facts", "FactExtractor"),
    "IngestionPipeline": (".pipeline", "IngestionPipeline"),
    "IngestOutcome": (".pipeline", "IngestOutcome"),
    "IngestProgress": (".pipeline", "IngestProgress"),
    "group_by_namespace": (".pipeline", "group_by_namespace"),
    "namespace_batch_count": (".pipeline", "namespace_batch_count"),
    "namespaced_sessions": (".pipeline", "namespaced_sessions"),
    "attach_source_span": (".provenance", "attach_source_span"),
    "source_span_for": (".provenance", "source_span_for"),
    "SINGLE_VALUED_PREDICATES": (".schemas", "SINGLE_VALUED_PREDICATES"),
    "DedupDecision": (".schemas", "DedupDecision"),
    "ExtractedMemory": (".schemas", "ExtractedMemory"),
    "ExtractionResult": (".schemas", "ExtractionResult"),
    "is_single_valued": (".schemas", "is_single_valued"),
    "normalize_predicate": (".schemas", "normalize_predicate"),
    "TwoStageExtractor": (".two_stage", "TwoStageExtractor"),
}

__all__ = [*_EXPORTS, "fingerprint"]


def __getattr__(name: str) -> Any:
    if name == "fingerprint":
        value = import_module(".fingerprint", __name__)
    else:
        try:
            module_name, attribute = _EXPORTS[name]
        except KeyError as exc:  # pragma: no cover - normal Python attribute semantics
            raise AttributeError(name) from exc
        value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
