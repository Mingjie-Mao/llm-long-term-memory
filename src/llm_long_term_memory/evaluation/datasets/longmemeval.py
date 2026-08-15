"""LongMemEval loader, plus the ingestion-budget planner.

LongMemEval (ICLR'25) is the benchmark ChronoMem reports against. LoCoMo is
deliberately excluded — see docs/DECISIONS.md.

Each of the three files holds 500 instances. The `_s` variant carries ~40 haystack
sessions per question (~115k tokens); `_m` carries ~500; `oracle` carries only the
evidence sessions and is used as a retrieval upper bound.

The planner at the bottom exists because on the Gemini free tier the number of
*requests* an ingestion costs decides how many days it runs. Measuring that before
writing the extraction pipeline is the whole point of P0.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx

BASE_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"

VARIANTS = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}

# Measured against the provider tokenizer on 2026-08-10 over a sample of sessions
# spanning 786-15,277 chars: the observed ratio was 4.55-4.92 chars/token. The
# textbook 4.0 overestimates this corpus by ~15%. 4.6 is used rather than the 4.68
# median so that planning errs toward overestimating the budget.
CHARS_PER_TOKEN = 4.6


@dataclass(slots=True)
class HaystackTurn:
    role: str
    content: str
    has_answer: bool = False


@dataclass(slots=True)
class HaystackSession:
    session_id: str
    date: str
    turns: list[HaystackTurn]

    @property
    def char_count(self) -> int:
        return sum(len(t.content) for t in self.turns)

    @property
    def is_evidence(self) -> bool:
        return any(t.has_answer for t in self.turns)


@dataclass(slots=True)
class Instance:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    sessions: list[HaystackSession]
    answer_session_ids: list[str]

    @property
    def is_abstention(self) -> bool:
        """`_abs` questions have no answer in the haystack; the correct behavior is
        to decline. These are the reason a memory system cannot be scored on recall
        alone."""
        return self.question_id.endswith("_abs")

    @property
    def char_count(self) -> int:
        return sum(s.char_count for s in self.sessions)

    @property
    def est_tokens(self) -> int:
        return int(self.char_count / CHARS_PER_TOKEN)


def download(variant: str, data_dir: str | Path = "data", force: bool = False) -> Path:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
    dest = Path(data_dir) / VARIANTS[variant]
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{VARIANTS[variant]}"
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(".part")
        with tmp.open("wb") as fh:
            for chunk in r.iter_bytes(chunk_size=1 << 20):
                fh.write(chunk)
        tmp.replace(dest)
    return dest


def _parse_instance(raw: dict) -> Instance:
    session_ids = raw.get("haystack_session_ids") or []
    dates = raw.get("haystack_dates") or []
    sessions = []
    for i, turns in enumerate(raw.get("haystack_sessions") or []):
        sessions.append(
            HaystackSession(
                session_id=session_ids[i] if i < len(session_ids) else f"sess_{i}",
                date=dates[i] if i < len(dates) else "",
                turns=[
                    HaystackTurn(
                        role=t.get("role", "user"),
                        content=t.get("content", ""),
                        has_answer=bool(t.get("has_answer", False)),
                    )
                    for t in turns
                ],
            )
        )
    return Instance(
        question_id=raw["question_id"],
        question_type=raw.get("question_type", ""),
        question=raw["question"],
        answer=raw.get("answer", ""),
        question_date=raw.get("question_date", ""),
        sessions=sessions,
        answer_session_ids=raw.get("answer_session_ids") or [],
    )


def stratify(instances: list[Instance], limit: int, seed: int = 0) -> list[Instance]:
    """Take a subset that preserves the question-type mix.

    The file is ordered by question type, so `instances[:50]` is 50 questions of a
    single type — and that type is `single-session-user`, the one a plain vector
    search already handles. A dev subset built that way cannot see temporal
    reasoning, knowledge updates, or abstention at all, which is most of what a
    memory system exists to do.

    Sampling is seeded so the subset is identical across variants; comparing two
    systems on different questions would not be a comparison.
    """
    if limit >= len(instances):
        return instances

    buckets: dict[str, list[Instance]] = {}
    for inst in instances:
        buckets.setdefault(inst.question_type, []).append(inst)

    rng = random.Random(seed)
    for group in buckets.values():
        rng.shuffle(group)

    # Largest-remainder allocation, so small categories are not rounded away.
    total = len(instances)
    quotas = {k: limit * len(v) / total for k, v in buckets.items()}
    take = {k: int(q) for k, q in quotas.items()}
    for key in sorted(buckets, key=lambda k: quotas[k] - take[k], reverse=True):
        if sum(take.values()) >= limit:
            break
        take[key] += 1

    picked = [inst for k, group in buckets.items() for inst in group[: take[k]]]
    picked.sort(key=lambda i: i.question_id)  # stable order across runs
    return picked


def split_dev_test(
    instances: list[Instance], dev_size: int = 50, test_size: int = 100, seed: int = 0
) -> tuple[list[Instance], list[Instance]]:
    """Disjoint stratified dev and test sets.

    The dev 50 are burned. Batch size, the predicate arity list, the v2 extraction
    prompt, and the decision to build temporal resolution before hybrid retrieval
    were all chosen by looking at them, so a number produced on those questions
    measures the fit of those choices as much as the system. Every headline claim
    has to land on questions no decision has ever been made against.

    Test is drawn from what dev did not take, with the same stratification, and is
    meant to be run once. Nothing about it should be inspected before that.
    """
    dev = stratify(instances, dev_size, seed=seed)
    dev_ids = {i.question_id for i in dev}
    remaining = [i for i in instances if i.question_id not in dev_ids]
    return dev, stratify(remaining, test_size, seed=seed)


def load(
    variant: str = "s",
    data_dir: str | Path = "data",
    limit: int | None = None,
    stratified: bool = True,
    seed: int = 0,
) -> list[Instance]:
    path = download(variant, data_dir)
    raw = json.loads(path.read_text(encoding="utf-8"))
    instances = [_parse_instance(r) for r in raw]
    if limit is None or limit >= len(instances):
        return instances
    return stratify(instances, limit, seed=seed) if stratified else instances[:limit]


def iter_sessions(instances: list[Instance]) -> Iterator[tuple[str, HaystackSession]]:
    for inst in instances:
        for sess in inst.sessions:
            yield inst.question_id, sess


# ------------------------------------------------------------------ statistics


@dataclass(slots=True)
class Stats:
    variant: str
    n_questions: int
    n_abstention: int
    question_types: dict[str, int]
    n_sessions: int
    n_unique_sessions: int
    n_turns: int
    total_chars: int
    est_total_tokens: int
    median_sessions_per_q: float
    median_tokens_per_q: float

    @property
    def session_sharing_factor(self) -> float:
        """>1 means sessions are reused across questions, so ingestion only has to
        process each once. This single number can change the ingestion cost by an
        order of magnitude, which is why P0 measures it before anything is built."""
        return self.n_sessions / self.n_unique_sessions if self.n_unique_sessions else 1.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def compute_stats(instances: list[Instance], variant: str = "s") -> Stats:
    types: dict[str, int] = {}
    unique: set[str] = set()
    n_sessions = n_turns = total_chars = 0

    for inst in instances:
        types[inst.question_type] = types.get(inst.question_type, 0) + 1
        for sess in inst.sessions:
            n_sessions += 1
            unique.add(sess.session_id)
            n_turns += len(sess.turns)
            total_chars += sess.char_count

    return Stats(
        variant=variant,
        n_questions=len(instances),
        n_abstention=sum(1 for i in instances if i.is_abstention),
        question_types=dict(sorted(types.items(), key=lambda kv: -kv[1])),
        n_sessions=n_sessions,
        n_unique_sessions=len(unique),
        n_turns=n_turns,
        total_chars=total_chars,
        est_total_tokens=int(total_chars / CHARS_PER_TOKEN),
        median_sessions_per_q=_median([len(i.sessions) for i in instances]),
        median_tokens_per_q=_median([float(i.est_tokens) for i in instances]),
    )


# -------------------------------------------------------------- budget planner


@dataclass(slots=True)
class IngestionPlan:
    sessions_per_request: int
    n_requests: int
    est_tokens_per_request: int
    days_at_quota: float
    fits_in_one_day: bool


def plan_ingestion(
    stats: Stats,
    sessions_per_request: int,
    rpd: int = 1_500,
    dedupe_sessions: bool = True,
) -> IngestionPlan:
    """How many requests, and therefore how many days, one full ingestion costs.

    `dedupe_sessions` assumes the pipeline extracts each unique session once and
    reuses the resulting memories across every question that references it. That
    is the single biggest lever available on a request-capped quota.
    """
    total = stats.n_unique_sessions if dedupe_sessions else stats.n_sessions
    n_requests = -(-total // sessions_per_request)  # ceil
    tokens_per_session = stats.est_total_tokens / stats.n_sessions if stats.n_sessions else 0
    return IngestionPlan(
        sessions_per_request=sessions_per_request,
        n_requests=n_requests,
        est_tokens_per_request=int(tokens_per_session * sessions_per_request),
        days_at_quota=n_requests / rpd,
        fits_in_one_day=n_requests <= rpd,
    )
