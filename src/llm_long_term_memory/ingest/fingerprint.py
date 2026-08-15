"""What produced a store's memories, as a value that can be compared.

`extractor_version` alone is not enough. It is a string somebody edits, so it
catches a deliberate generation change (`two-stage-v4` -> `two-stage-p10-v2`) and
misses the more likely failure: a prompt reworded, a schema column added, or a
batch size changed, with the version string left alone. The store then holds
memories from two systems that both call themselves the same thing, which is the
mixed-store incident with no label to notice it by.

So the fingerprint is computed from the inputs themselves — prompt text, schema
text, model id, the extraction config — rather than from a name for them. Editing
a prompt changes it without anyone remembering to bump a constant.

It is a dict, not one opaque hash, so a mismatch can say *which* input moved. An
error that reports "fingerprint differs" sends the reader looking everywhere.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_SCHEMA = Path(__file__).resolve().parent.parent / "store" / "schema.sql"


def digest(*parts: str) -> str:
    """A short content hash. Twelve hex chars — this identifies a build, it does not
    defend against anyone constructing a collision."""
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x00")  # so ("ab","c") and ("a","bc") differ
    return h.hexdigest()[:12]


def schema_version() -> str:
    """Derived from schema.sql, because a hand-maintained constant is exactly the
    thing that does not get bumped when a column is added."""
    return digest(_SCHEMA.read_text(encoding="utf-8"))


def build(extractor: Any, *, sessions_per_request: int, dedup: Any = None) -> dict[str, str]:
    """The full fingerprint of an ingestion run.

    `sessions_per_request` is in it because batch size is not a performance knob
    here: it changes how many sessions share one prompt, and so what the model sees
    and how it segments its answer.
    """
    fp: dict[str, str] = {
        "extractor_version": str(getattr(extractor, "version", "unversioned")),
        "schema_version": schema_version(),
        "model": str(getattr(extractor, "model", "unknown")),
        "sessions_per_request": str(sessions_per_request),
    }
    prompts = getattr(extractor, "prompt_texts", None)
    fp["prompts"] = digest(*prompts()) if callable(prompts) else "unfingerprinted"
    threshold = getattr(dedup, "threshold", None)
    if threshold is not None:
        fp["dedup_threshold"] = f"{float(threshold):.4f}"
    return fp


def differences(previous: dict[str, str], current: dict[str, str]) -> list[str]:
    """Which components moved, named individually. Keys present on one side only
    count as differences — a component that did not exist before is a change."""
    return sorted(
        f"{k}: {previous.get(k, '(absent)')} -> {current.get(k, '(absent)')}"
        for k in set(previous) | set(current)
        if previous.get(k) != current.get(k)
    )


def dumps(fp: dict[str, str]) -> str:
    return json.dumps(fp, sort_keys=True)


def loads(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}
