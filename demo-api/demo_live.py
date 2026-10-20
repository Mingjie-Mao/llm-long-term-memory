"""Opt-in LLM adapter for the disposable public playground.

Product extraction/answering stay in MemoryService. This layer owns public-demo
admission, conservative capacity reservations and request-local inspector traces.
No prompts or API keys are written to the capacity ledger.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import HTTPException

ROOT = Path(__file__).resolve().parent.parent


def enabled() -> bool:
    value = os.environ.get("LLTM_DEMO_LLM_ENABLED", "0").strip().lower()
    if value not in {"0", "1", "false", "true"}:
        raise ValueError("LLTM_DEMO_LLM_ENABLED must be 0 or 1")
    return value in {"1", "true"}


def positive(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


class CapacityExceeded(RuntimeError):
    pass


class DemoClient:
    """Reserve before calling; failures and compatibility retries spend capacity.

    GeminiClient gets TWO attempts total, shared by transient API errors and the
    optional thinking_config compatibility retry. Disable SDK/transport retries,
    and never refund failed or unused capacity. Token capacity uses UTF-8 byte
    lengths plus schema/output bounds; this is deliberately not a billing meter.
    The playground lock serializes owners and ledger writes.
    """

    def __init__(self, base, store) -> None:
        self.base = base
        self.store = store
        self.usage = base.usage
        self.owner: str | None = None
        self.inputs: list[dict] = []
        self.session_calls = positive("LLTM_DEMO_SESSION_CALLS", 32)
        self.session_tokens = positive("LLTM_DEMO_SESSION_TOKENS", 300_000)
        self.global_calls = positive("LLTM_DEMO_DAILY_CALLS", 100)
        self.global_tokens = positive("LLTM_DEMO_DAILY_TOKENS", 1_000_000)
        with store._conn:
            store._conn.execute(
                "CREATE TABLE IF NOT EXISTS demo_llm_capacity ("
                "day TEXT NOT NULL, owner TEXT NOT NULL, calls INTEGER NOT NULL, "
                "tokens INTEGER NOT NULL, PRIMARY KEY(day, owner))"
            )

    def limits(self) -> dict:
        return {"calls": self.session_calls, "tokens": self.session_tokens}

    def reserved(self, owner: str, day: str | None = None) -> dict:
        row = self.store._conn.execute(
            "SELECT calls, tokens FROM demo_llm_capacity WHERE day=? AND owner=?",
            (day or datetime.now(UTC).date().isoformat(), owner),
        ).fetchone()
        return {"calls": row[0] if row else 0, "tokens": row[1] if row else 0}

    def _reserve(self, tokens: int) -> None:
        if self.owner is None:
            raise RuntimeError("LLM calls require a demo namespace")
        day = datetime.now(UTC).date().isoformat()
        rows = [
            (self.owner, self.session_calls, self.session_tokens),
            ("*", self.global_calls, self.global_tokens),
        ]
        with self.store._conn:
            for owner, calls_cap, token_cap in rows:
                spent = self.reserved(owner, day)
                if spent["calls"] + 2 > calls_cap or spent["tokens"] + tokens > token_cap:
                    raise CapacityExceeded(
                        "Demo capacity exhausted; try again after the UTC day resets."
                    )
            for owner, _, _ in rows:
                self.store._conn.execute(
                    "INSERT INTO demo_llm_capacity VALUES(?,?,2,?) "
                    "ON CONFLICT(day,owner) DO UPDATE SET "
                    "calls=calls+2, tokens=tokens+excluded.tokens",
                    (day, owner, tokens),
                )

    @contextmanager
    def operation(self, namespace: str):
        if self.owner is not None:
            raise RuntimeError("concurrent demo LLM operation")
        self.owner = namespace
        self.inputs = []
        self.deadline = time.monotonic() + 60
        self.usage.records.clear()
        try:
            yield
        finally:
            self.inputs.clear()
            self.usage.records.clear()
            self.owner = None

    def generate(self, **kwargs):
        if self.owner is None or time.monotonic() >= self.deadline:
            raise CapacityExceeded("Demo operation timed out; retry later.")
        # Keep t4's specified output limits; bound otherwise-unbounded extraction.
        if not kwargs.get("max_output_tokens"):
            kwargs["max_output_tokens"] = 4096
        schema = kwargs.get("schema")
        schema_text = str(schema.model_json_schema()) if schema else ""
        input_bound = (
            len((kwargs["prompt"] + (kwargs.get("system") or "") + schema_text).encode("utf-8"))
            + 2048
        )
        wait = self.base.quota.for_model(kwargs["model"]).check(input_bound)
        if wait is not None:
            raise CapacityExceeded("Model rate limit reached; retry later.")
        self._reserve(2 * (input_bound + kwargs["max_output_tokens"]))
        if kwargs["role"] == "answerer":
            self.inputs.append({"model": kwargs["model"], "prompt": kwargs["prompt"]})
        return self.base.generate(**kwargs)


def attach(playground, store_path: Path):
    """Replace the bootstrap store with one product-owned resource set."""
    from google.genai import types

    from llm_long_term_memory.api.service import MemoryService
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.llm import Limits, QuotaManager, UsageTracker
    from llm_long_term_memory.llm.client import GeminiClient

    settings = Settings(store_dir=store_path.parent)
    settings.require_api_key()
    playground.store.close()
    service = MemoryService(
        config_path=ROOT / "configs/fallback.yaml",
        store_name=store_path.name,
        settings=settings,
        encoder=playground.encoder,
    )
    base = None
    try:
        quota = QuotaManager(
            state_dir=store_path.parent / f"{store_path.name}-quota",
            default=Limits(rpm=10, tpm=250_000, rpd=500),
        )
        quota.load_learned()
        usage = UsageTracker()
        base = GeminiClient(
            settings.require_api_key(),
            quota,
            usage,
            max_retries=2,
            max_transport_retries=1,
            timeout_seconds=30,
        )
        base._client._api_client._http_options.retry_options = types.HttpRetryOptions(attempts=1)
        client = DemoClient(base, service.store)
        service._live_client_instance = client
        service._live_usage = usage
        service._lock = playground.lock
        playground.store = service.store
        playground.index = service.index
        playground.retriever = service.retriever
        playground.resolver = service.resolver
        return service, client
    except BaseException:
        if base is not None:
            base._client.close()
        service.close()
        raise


def unavailable() -> HTTPException:
    return HTTPException(
        503,
        detail={
            "error": "llm_unavailable",
            "message": "Live LLM service is unavailable. No rule-based answer will be substituted.",
        },
    )
