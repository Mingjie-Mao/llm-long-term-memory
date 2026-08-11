from .rate_limiter import Limits, QuotaManager, RateLimiter, Wait
from .usage import CallRecord, RoleStats, UsageTracker

__all__ = [
    "CallRecord",
    "Limits",
    "QuotaManager",
    "RateLimiter",
    "RoleStats",
    "UsageTracker",
    "Wait",
]


def __getattr__(name: str):
    # google-genai is an optional extra; importing it eagerly would make the
    # dataset/stats commands require a dependency they do not use.
    if name in ("GeminiClient", "Completion", "DailyQuotaExhausted"):
        from . import client

        return getattr(client, name)
    raise AttributeError(name)
