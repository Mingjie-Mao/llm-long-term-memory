"""Per-call accounting for requests, tokens, and latency.

Every LLM call goes through here so that "how many tokens and how long?" is a
property of a run rather than something reconstructed afterwards. The results table
reports token cost and p95 latency next to accuracy; without this the interesting
half of that table cannot be filled in.

Latency percentiles are computed from the full sample rather than a streaming
estimate — runs are thousands of calls, not millions, so exactness is free.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class CallRecord:
    role: str  # 'extractor' | 'answerer' | 'judge' | 'embedder'
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    ok: bool = True
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    # Nearest-rank; with n in the thousands the interpolation choice is noise.
    idx = min(len(sorted_values) - 1, round(q * (len(sorted_values) - 1)))
    return sorted_values[idx]


@dataclass(slots=True)
class RoleStats:
    role: str
    calls: int
    failures: int
    input_tokens: int
    output_tokens: int
    p50_ms: float
    p95_ms: float
    max_ms: float

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class UsageTracker:
    records: list[CallRecord] = field(default_factory=list)

    def record(self, rec: CallRecord) -> None:
        self.records.append(rec)

    @contextmanager
    def measure(self, role: str, model: str):
        """Times a call and records it even when it raises.

        A failed call still consumed quota, so it must appear in the accounting —
        otherwise a run that burned its daily budget on retries looks free.
        """
        started = time.perf_counter()
        box: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
        try:
            yield box
        except Exception as exc:
            self.record(
                CallRecord(
                    role=role,
                    model=model,
                    input_tokens=box["input_tokens"],
                    output_tokens=box["output_tokens"],
                    latency_ms=(time.perf_counter() - started) * 1000,
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}"[:300],
                )
            )
            raise
        else:
            self.record(
                CallRecord(
                    role=role,
                    model=model,
                    input_tokens=box["input_tokens"],
                    output_tokens=box["output_tokens"],
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            )

    def by_role(self) -> dict[str, RoleStats]:
        groups: dict[str, list[CallRecord]] = defaultdict(list)
        for r in self.records:
            groups[r.role].append(r)

        out: dict[str, RoleStats] = {}
        for role, recs in groups.items():
            lat = sorted(r.latency_ms for r in recs)
            out[role] = RoleStats(
                role=role,
                calls=len(recs),
                failures=sum(1 for r in recs if not r.ok),
                input_tokens=sum(r.input_tokens for r in recs),
                output_tokens=sum(r.output_tokens for r in recs),
                p50_ms=_percentile(lat, 0.50),
                p95_ms=_percentile(lat, 0.95),
                max_ms=lat[-1] if lat else 0.0,
            )
        return out

    @property
    def total_requests(self) -> int:
        return len(self.records)

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.records)

    def summary(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "failures": sum(1 for r in self.records if not r.ok),
            "by_role": {k: asdict(v) for k, v in self.by_role().items()},
        }

    @classmethod
    def _load_records(cls, path: Path) -> list[CallRecord]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("usage artifact must be a JSON object")
        calls = payload.get("calls", [])
        if not isinstance(calls, list):
            raise ValueError("usage artifact has a non-list calls field")
        return [CallRecord(**call) for call in calls]

    def save(self, path: str | Path, *, merge: bool = False) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        records = self.records
        if merge and p.exists():
            records = [*self._load_records(p), *records]
        combined = UsageTracker(records=records)
        temporary = p.with_suffix(p.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {"summary": combined.summary(), "calls": [asdict(r) for r in combined.records]},
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(p)
