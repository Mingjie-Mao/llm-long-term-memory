"""Recorded demonstration runs, and the fingerprint that keeps them honest.

A demo page has two bad options and one good one. Running the answerer on every page
load spends quota on every visitor and refresh. Hard-coding an answer into the page
is a mock, and a mock that looks like a live result is a lie. The third option is to
**record a real run, show it labelled as recorded, and offer to re-run it on demand**.

The risk that creates is staleness: a recorded answer from an older store or an older
prompt, displayed next to a system that no longer behaves that way, is exactly the
"old result impersonating a current one" failure this project has been careful about
elsewhere (see result versioning in the harness). So every recording carries a
fingerprint of what produced it, and the API reports whether the current process
still matches.

The fingerprint deliberately excludes anything that cannot change the answer — the
port, the log level — and includes everything that can: the question, the namespace,
the three prompt/extractor versions, and the state of the store the answer was drawn
from.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

GOLDEN_PATH = Path("results/golden-runs.json")


@dataclass(slots=True)
class RecordedEvidence:
    session_id: str
    turn_index: int
    role: str
    text: str


@dataclass(slots=True)
class GoldenRun:
    """One recorded answer, with everything needed to judge whether it still holds."""

    name: str
    namespace: str
    query: str
    answer: str
    answer_status: str
    """The answerer's own verdict: answer | need_source | no_evidence."""
    fallback_level: str
    fallback_reason: str | None
    evidence: list[RecordedEvidence] = field(default_factory=list)
    memories_selected: int = 0
    candidates_considered: int = 0
    judge_verdict: str | None = None
    judge_reason: str | None = None
    recorded_at: str = ""
    runs: int = 1
    """How many independent runs agreed. One run is an anecdote; three is a claim."""
    fingerprint: str = ""
    versions: dict[str, str | None] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fingerprint(
    namespace: str,
    query: str,
    versions: dict[str, str | None],
    store_state: str,
) -> str:
    """Stable hash of everything that could change the answer.

    Order-independent over `versions` so that adding a key later does not silently
    invalidate every existing recording for a reason unrelated to behaviour.
    """
    payload = json.dumps(
        {
            "namespace": namespace,
            "query": query.strip().lower(),
            "versions": {k: v for k, v in sorted(versions.items())},
            "store": store_state,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def store_state(store, namespace: str) -> str:
    """A cheap summary of the data an answer was drawn from.

    Count plus latest ingest time, not a hash of every row: the point is to notice a
    re-ingest or a schema migration, and walking thousands of rows on every page load
    to catch an edit that changes neither would cost more than it is worth.
    """
    memories = store.iter_all(namespace)
    latest = max((m.ingested_at for m in memories if m.ingested_at), default=None)
    extractor = store.get_meta("extractor_version") if hasattr(store, "get_meta") else None
    return f"n={len(memories)};latest={latest.isoformat() if latest else 'none'};ext={extractor}"


def load_runs(path: str | Path | None = None) -> dict[str, GoldenRun]:
    p = Path(path or GOLDEN_PATH)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, GoldenRun] = {}
    for entry in data.get("runs", []):
        evidence = [RecordedEvidence(**e) for e in entry.pop("evidence", [])]
        out[entry["name"]] = GoldenRun(**entry, evidence=evidence)
    return out


def save_runs(runs: dict[str, GoldenRun], path: str | Path | None = None) -> Path:
    p = Path(path or GOLDEN_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "note": (
                    "Recorded answer runs used by the inspector's default view. Each is a "
                    "real run against a real model, replayed rather than re-executed, and "
                    "labelled as such in the UI. `fingerprint` is checked against the live "
                    "process so a recording made under a different store or prompt is "
                    "reported stale instead of shown as current."
                ),
                "runs": [r.to_dict() for r in runs.values()],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return p


def recorded_at_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
