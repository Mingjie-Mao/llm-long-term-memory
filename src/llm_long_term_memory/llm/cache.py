"""Reuse a provider response only when the request is byte-for-byte the same.

The point is development speed: after a change that touches one question type, the
questions whose requests did not change need not spend quota again. Two rules keep
that from turning into a measurement error:

- The key is everything the provider sees — model, role, prompt (which carries the
  whole context and the user's data), system prompt, schema, sampling parameters —
  plus `scope`, for whatever the request does not show (a store fingerprint, a judge
  configuration). Any change to any of it is a different key.
- `sample` is required and part of the key. A cached answer is one sample: repetition
  `rep2` never reuses `rep1`'s answers, so three cached runs are never mistaken for
  three independent draws. To measure variance, use three sample labels.

Failures are never cached; they propagate exactly as from the wrapped client.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

KEY_VERSION = "response-cache-v1"


def _schema_identity(schema: Any) -> Any:
    if schema is None:
        return None
    if hasattr(schema, "model_json_schema"):
        return schema.model_json_schema()
    return json.loads(json.dumps(schema, sort_keys=True, default=str))


def request_key(*, sample: str, scope: dict | None, **request: Any) -> str:
    """The cache key for one `generate()` call; `est_input_tokens` is not part of it."""
    request = {k: v for k, v in request.items() if k != "est_input_tokens"}
    request["schema"] = _schema_identity(request.get("schema"))
    payload = {
        "version": KEY_VERSION,
        "sample": sample,
        "scope": scope or {},
        "request": request,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class CachingClient:
    """Wraps a provider client; same `generate()` signature, other attributes pass through."""

    def __init__(
        self,
        client,
        path: str | Path,
        *,
        sample: str,
        scope: dict | None = None,
    ) -> None:
        if not sample:
            raise ValueError("a cache needs a sample label, e.g. 'rep1'")
        self._client = client
        self.sample = sample
        self.scope = dict(scope or {})
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.execute(
            "create table if not exists responses ("
            " key text primary key, sample text not null, model text not null,"
            " role text not null, completion text not null)"
        )
        self.hits = 0
        self.misses = 0

    def __getattr__(self, name: str):
        return getattr(self._client, name)

    def generate(self, **request: Any):
        from llm_long_term_memory.llm.client import Completion

        key = request_key(sample=self.sample, scope=self.scope, **request)
        row = self._db.execute("select completion from responses where key=?", (key,)).fetchone()
        if row is not None:
            self.hits += 1
            stored = json.loads(row[0])
            # No provider call was made: no attempts and no API latency to report.
            return Completion(**{**stored, "attempts": 0, "api_latency_ms": 0.0})
        completion = self._client.generate(**request)
        self.misses += 1
        self._db.execute(
            "insert or replace into responses values (?, ?, ?, ?, ?)",
            (
                key,
                self.sample,
                request.get("model", ""),
                request.get("role", ""),
                json.dumps(asdict(completion), ensure_ascii=False),
            ),
        )
        self._db.commit()
        return completion

    def close(self) -> None:
        self._db.close()
