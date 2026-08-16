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

## Two builders, one spec

The identity is `IngestSpec`, and there are two ways to arrive at it:

    from_runtime(extractor, dedup, ...)   the objects that are about to run
    from_config(cfg, ...)                 resolved configuration, nothing else

They must agree, and `tests/test_fingerprint.py` asserts that they do. The split
exists because the two callers can afford different things. Ingestion already
holds a live extractor and deduplicator, so it should fingerprint *those* — the
objects that will actually write rows. Preflight validates without starting the
system, and must not construct a Gemini client or load an encoder to decide that
`model` is `gemini-3.1-flash-lite`; a check that needs the model dependencies
installed in order to run is a check that fails for reasons unrelated to what it
is checking, and one that could reach the network is worse.

Every field is therefore reachable without a live object: `version` and
`prompt_texts()` are class-level on both extractors, and the model and threshold
are configuration. What is *not* derivable from config alone is
`sessions_per_request`, which ingestion lowers to fit the model's token budget —
so it is an explicit argument to both builders and neither invents it.

The earlier version of this module had one builder taking a live extractor, and
preflight called it with the *class*. `model` is an instance attribute, so it
fingerprinted as `unknown`; no deduplicator was passed, so `dedup_threshold` was
absent. The expected fingerprint could not match any real store, and the check
that guards the formal evaluation could never pass. Hence the parity test: a
divergence between the two paths is now a test failure rather than a check that
silently stops being a check.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class IngestSpec:
    """The identity of an ingestion setup, however it was arrived at.

    `sessions_per_request` is in it because batch size is not a performance knob
    here: it changes how many sessions share one prompt, and so what the model sees
    and how it segments its answer.
    """

    extractor_version: str
    model: str
    schema_version: str
    prompts: str
    sessions_per_request: str
    dedup_threshold: str | None = None

    def as_dict(self) -> dict[str, str]:
        """The stored form. Kept as a flat string dict because it is what is written
        to `meta` and compared against stores written before this refactor — the
        clean P10 store among them, which must keep matching without a re-ingest.

        `dedup_threshold` is omitted rather than nulled when there is no
        deduplicator, preserving the shape of fingerprints already on disk.
        """
        fp = {
            "extractor_version": self.extractor_version,
            "schema_version": self.schema_version,
            "model": self.model,
            "sessions_per_request": self.sessions_per_request,
            "prompts": self.prompts,
        }
        if self.dedup_threshold is not None:
            fp["dedup_threshold"] = self.dedup_threshold
        return fp


def _threshold(value: Any) -> str | None:
    """Fixed four decimals, so 0.92 and 0.9200 are one value rather than two."""
    return None if value is None else f"{float(value):.4f}"


def from_runtime(extractor: Any, *, sessions_per_request: int, dedup: Any = None) -> IngestSpec:
    """The spec of the objects that are about to write rows.

    Reads the live instances rather than the configuration that produced them, so a
    runtime that was built differently from what the config says fingerprints as
    what it is. `run()` compares this against the config-derived spec for exactly
    that reason.
    """
    prompts = getattr(extractor, "prompt_texts", None)
    return IngestSpec(
        extractor_version=str(getattr(extractor, "version", "unversioned")),
        model=str(getattr(extractor, "model", "unknown")),
        schema_version=schema_version(),
        prompts=digest(*prompts()) if callable(prompts) else "unfingerprinted",
        sessions_per_request=str(sessions_per_request),
        dedup_threshold=_threshold(getattr(dedup, "threshold", None)),
    )


def from_config(cfg: Any, *, sessions_per_request: int) -> IngestSpec:
    """The same spec from resolved configuration — no client, no encoder, no network.

    `sessions_per_request` is passed in rather than read from `cfg`, because the
    effective batch size is the configured one capped by the model's token budget.
    Callers resolve it with `pipeline.resolved_sessions_per_request` so that the
    cap is applied identically on both sides.
    """
    from .extract import Extractor
    from .two_stage import TwoStageExtractor

    extractor = TwoStageExtractor if cfg.ingest.two_stage else Extractor
    return IngestSpec(
        extractor_version=str(extractor.version),
        model=str(cfg.models.extractor),
        schema_version=schema_version(),
        prompts=digest(*extractor.prompt_texts()),
        sessions_per_request=str(sessions_per_request),
        # Unconditional, because ingestion builds a Deduplicator unconditionally.
        # `ingest.dedupe_sessions` is a corpus-planning flag and does not gate it.
        dedup_threshold=_threshold(cfg.ingest.dedupe_similarity_threshold),
    )


def differences(previous: dict[str, str], current: dict[str, str]) -> list[str]:
    """Which components moved, named individually. Keys present on one side only
    count as differences — a component that did not exist before is a change."""
    return sorted(
        f"{k}: {previous.get(k, '(absent)')} -> {current.get(k, '(absent)')}"
        for k in set(previous) | set(current)
        if previous.get(k) != current.get(k)
    )


def dumps(fp: IngestSpec | dict[str, str]) -> str:
    return json.dumps(fp.as_dict() if isinstance(fp, IngestSpec) else fp, sort_keys=True)


def loads(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}
