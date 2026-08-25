"""Diagnostic: can the raw-conversation fallback find the right turn at all?

The fallback currently searches with BM25. Whether it needs dense retrieval as well
is an empirical question, and the wrong way to answer it is end-to-end accuracy —
that folds retrieval, answering and judging into one number and cannot say which
moved. The direct question is narrower:

    **When a question needs a raw turn, does that turn come back in the top k?**

So this measures evidence recall only. No answerer, no judge, no API calls.

**Query types are the point of the experiment.** BM25 scores lexical overlap, so a
question that reuses the source's wording flatters it. Users months later will not
reuse that wording. Three types, constructed from the data rather than invented:

* `keyword` — salient terms lifted straight out of the gold turn. BM25's best case,
  and therefore an upper bound rather than a realistic query.
* `natural` — LongMemEval's own question, written by the benchmark's authors as a
  user would ask it. The realistic case.
* `disjoint` — the natural question with every content word it shares with the gold
  turn removed. A mechanical stand-in for "asked in completely different words",
  which is precisely where lexical matching should fail if it is going to.

`disjoint` is generated, not hand-written, so it cannot be unconsciously tuned to
make a point — and it is the type whose result decides whether embeddings are worth
their cost.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Literal

from llm_long_term_memory.evaluation.datasets.longmemeval import Instance
from llm_long_term_memory.store import MemoryStore, external_session_id

QueryType = Literal["keyword", "natural", "disjoint"]

_WORD = re.compile(r"[A-Za-z0-9']+")

# Removed before measuring overlap: they carry no retrieval signal, and leaving them
# in would make a `disjoint` query look lexically distinct while still sharing every
# word that matters.
_STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "with",
        "without",
        "about",
        "into",
        "over",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "too",
        "very",
        "can",
        "will",
        "just",
        "should",
        "now",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "it",
        "its",
        "they",
        "them",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "be",
        "been",
        "being",
        "am",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "doing",
        "have",
        "has",
        "had",
        "having",
        "would",
        "could",
        "may",
        "might",
        "must",
        "shall",
        "us",
        "also",
        "get",
        "got",
        "give",
        "given",
        "tell",
        "told",
        "say",
        "said",
        "ask",
        "asked",
        "remind",
        "remember",
        "recall",
        "previous",
        "previously",
        "earlier",
        "before",
        "mentioned",
        "discussed",
        "talked",
        "conversation",
        "chat",
        "me",
    ]
)


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def _content_words(text: str) -> list[str]:
    return [w for w in _words(text) if w not in _STOP and len(w) > 2]


def keyword_query(gold_turn_text: str, limit: int = 8) -> str:
    """Salient terms from the gold turn: BM25's best case.

    Longest-first rather than frequency-ranked, because in a short turn the rare
    specific token ("ergonomics", "UfOvNlX9Hh0") is what a searcher would reach for,
    and term frequency in a single passage is mostly noise.
    """
    seen: dict[str, None] = {}
    for word in sorted(set(_content_words(gold_turn_text)), key=len, reverse=True):
        seen[word] = None
        if len(seen) >= limit:
            break
    return " ".join(seen)


def disjoint_query(question: str, gold_turn_text: str) -> str:
    """The question stripped of every content word it shares with the gold turn.

    Returns "" when nothing survives — a question made entirely of words from its
    source cannot be asked disjointly, and counting it as a retrieval failure would
    blame the retriever for an impossible query.
    """
    gold = set(_content_words(gold_turn_text))
    kept = [w for w in _content_words(question) if w not in gold]
    return " ".join(kept)


@dataclass(slots=True)
class Probe:
    question_id: str
    question_type: str
    query_type: QueryType
    query: str
    gold_turn_ids: set[str]
    rank: int | None = None
    """1-based position of the first gold turn, or None if it never appeared."""
    latency_ms: float = 0.0

    def hit_at(self, k: int) -> bool:
        return self.rank is not None and self.rank <= k

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.rank if self.rank else 0.0


@dataclass(slots=True)
class RecallReport:
    probes: list[Probe] = field(default_factory=list)

    def subset(self, query_type: QueryType | None = None) -> list[Probe]:
        return [p for p in self.probes if query_type is None or p.query_type == query_type]

    def recall_at(self, k: int, query_type: QueryType | None = None) -> float:
        rows = self.subset(query_type)
        return sum(p.hit_at(k) for p in rows) / len(rows) if rows else 0.0

    def mrr(self, query_type: QueryType | None = None) -> float:
        rows = self.subset(query_type)
        return sum(p.reciprocal_rank for p in rows) / len(rows) if rows else 0.0

    def median_latency_ms(self, query_type: QueryType | None = None) -> float:
        rows = sorted(p.latency_ms for p in self.subset(query_type))
        return rows[len(rows) // 2] if rows else 0.0

    def render(self, retriever: str = "bm25") -> str:
        header = (
            "| retriever | query type | n | R@1 | R@3 | R@5 | MRR | median latency |\n"
            "|---|---|---:|---:|---:|---:|---:|---:|\n"
        )
        rows = []
        for qt in ("keyword", "natural", "disjoint"):
            n = len(self.subset(qt))
            if not n:
                continue
            rows.append(
                f"| {retriever} | {qt} | {n} | "
                f"{self.recall_at(1, qt):.1%} | {self.recall_at(3, qt):.1%} | "
                f"{self.recall_at(5, qt):.1%} | {self.mrr(qt):.3f} | "
                f"{self.median_latency_ms(qt):.1f}ms |"
            )
        rows.append(
            f"| {retriever} | **all** | {len(self.probes)} | "
            f"{self.recall_at(1):.1%} | {self.recall_at(3):.1%} | {self.recall_at(5):.1%} | "
            f"{self.mrr():.3f} | {self.median_latency_ms():.1f}ms |"
        )
        return header + "\n".join(rows)


def build_probes(instances: list[Instance], namespaces: set[str]) -> list[Probe]:
    """One probe per (question, query type), using LongMemEval's own gold turn flags.

    Questions whose namespace is not in the store are skipped rather than counted as
    misses: an un-ingested namespace measures the ingest, not the retriever.
    """
    probes: list[Probe] = []
    for inst in instances:
        if inst.question_id not in namespaces:
            continue
        gold_ids: set[str] = set()
        gold_texts: list[str] = []
        for session in inst.sessions:
            for index, turn in enumerate(session.turns):
                if turn.has_answer:
                    gold_ids.add(f"{session.session_id}:{index}")
                    gold_texts.append(turn.content)
        if not gold_ids:
            continue

        gold_text = "\n".join(gold_texts)
        candidates: list[tuple[QueryType, str]] = [
            ("keyword", keyword_query(gold_text)),
            ("natural", inst.question),
            ("disjoint", disjoint_query(inst.question, gold_text)),
        ]
        for query_type, query in candidates:
            if not query.strip():
                continue
            probes.append(
                Probe(
                    question_id=inst.question_id,
                    question_type=inst.question_type,
                    query_type=query_type,
                    query=query,
                    gold_turn_ids=gold_ids,
                )
            )
    return probes


def run_probes(store: MemoryStore, probes: list[Probe], depth: int = 5) -> RecallReport:
    """Search once per probe and record where the gold turn landed."""
    for probe in probes:
        started = time.perf_counter()
        hits = store.search_turns(probe.question_id, probe.query, limit=depth)
        probe.latency_ms = (time.perf_counter() - started) * 1000
        for position, turn in enumerate(hits, start=1):
            public_turn_id = f"{external_session_id(turn.session_id)}:{turn.turn_index}"
            if public_turn_id in probe.gold_turn_ids:
                probe.rank = position
                break
    return RecallReport(probes=probes)
