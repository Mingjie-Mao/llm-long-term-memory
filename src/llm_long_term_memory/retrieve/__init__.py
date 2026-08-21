"""Retrieval policies kept separate from persistence and prompt assembly."""

from .coherent import (
    CoherentContext,
    SessionBudget,
    build_coherent_context,
    rank_sessions,
    session_recall,
)
from .fallback import RawEvidence, RawFallback
from .hybrid import HybridRetriever, RetrievalSignals, RetrievalTrace, RetrievedMemory
from .hydrate import EvidenceHydrator, HydratedEvidence, HydrationResult, render_evidence
from .rerank import CrossEncoderReranker

__all__ = [
    "CoherentContext",
    "CrossEncoderReranker",
    "EvidenceHydrator",
    "HybridRetriever",
    "HydratedEvidence",
    "HydrationResult",
    "RawEvidence",
    "RawFallback",
    "RetrievalSignals",
    "RetrievalTrace",
    "RetrievedMemory",
    "SessionBudget",
    "build_coherent_context",
    "rank_sessions",
    "render_evidence",
    "session_recall",
]
