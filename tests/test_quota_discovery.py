"""Runtime quota discovery.

Free-tier per-model limits are undocumented and differ by orders of magnitude
(gemini-3.6-flash allows 20 requests/day; flash-lite allows thousands). The only
authoritative statement of the number is the QuotaFailure detail inside a 429, so
the limiter reads it from there. These tests pin the parsing against a real 429
body captured on 2026-08-10.
"""

from __future__ import annotations

from chronomem.llm import Limits, QuotaManager
from chronomem.llm.client import parse_quota_violations

_METRIC = "generativelanguage.googleapis.com/generate_content_free_tier_requests"

# Trimmed from a genuine response; structure preserved exactly.
REAL_429 = {
    "error": {
        "code": 429,
        "message": "You exceeded your current quota... limit: 20, model: gemini-3.6-flash",
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.Help", "links": []},
            {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [
                    {
                        "quotaMetric": _METRIC,
                        "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                        "quotaDimensions": {"location": "global", "model": "gemini-3.6-flash"},
                        "quotaValue": "20",
                    },
                    {
                        "quotaMetric": _METRIC,
                        "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
                        "quotaDimensions": {"location": "global", "model": "gemini-3.6-flash"},
                        "quotaValue": "5",
                    },
                ],
            },
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "58s"},
        ],
    }
}


class FakeAPIError(Exception):
    def __init__(self, details):
        super().__init__("429")
        self.code = 429
        self.details = details


def test_parses_per_day_and_per_minute_values():
    violations = parse_quota_violations(FakeAPIError(REAL_429))
    assert len(violations) == 2

    daily = next(v for v in violations if v.is_per_day)
    assert daily.value == 20
    assert not daily.is_per_minute

    minute = next(v for v in violations if v.is_per_minute)
    assert minute.value == 5


def test_pro_style_zero_limit_is_parsed_not_treated_as_missing():
    """`limit: 0` is how the API says a model is not on the free tier at all."""
    body = {
        "error": {
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [
                        {
                            "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                            "quotaValue": "0",
                        }
                    ],
                }
            ]
        }
    }
    (v,) = parse_quota_violations(FakeAPIError(body))
    assert v.value == 0


TOKEN_429 = {
    "error": {
        "details": [
            {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [
                    {
                        "quotaId": "GenerateContentInputTokensPerModelPerMinute-FreeTier",
                        "quotaValue": "250000",
                    }
                ],
            }
        ]
    }
}


def test_token_quota_is_not_mistaken_for_a_request_quota():
    """Both quota IDs end in PerMinute, but one counts tokens and reports 250,000.
    Adopting that as requests-per-minute disables rate limiting entirely."""
    (v,) = parse_quota_violations(FakeAPIError(TOKEN_429))
    assert v.is_tokens
    assert v.is_tokens_per_minute
    assert not v.is_per_minute
    assert not v.is_per_day


def test_token_violation_updates_tpm_and_leaves_rpm_alone(tmp_path):
    qm = QuotaManager(state_dir=tmp_path, default=Limits(rpm=10, tpm=10**9, rpd=1_500))
    qm.learn("m", parse_quota_violations(FakeAPIError(TOKEN_429)))

    limits = qm.for_model("m").limits
    assert limits.tpm == 250_000
    assert limits.rpm == 10, "a token limit must not overwrite the request limit"
    assert limits.rpd == 1_500


def test_unparseable_body_yields_no_violations():
    assert parse_quota_violations(FakeAPIError(None)) == []
    assert parse_quota_violations(Exception("plain")) == []
    assert parse_quota_violations(FakeAPIError({"error": {"details": ["junk", 42]}})) == []


def test_manager_adopts_learned_limits(tmp_path):
    qm = QuotaManager(state_dir=tmp_path, default=Limits(rpm=10, tpm=250_000, rpd=1_500))
    limiter = qm.for_model("gemini-3.6-flash")
    assert limiter.limits.rpd == 1_500  # the wrong default we started with

    qm.learn("gemini-3.6-flash", parse_quota_violations(FakeAPIError(REAL_429)))

    assert qm.for_model("gemini-3.6-flash").limits.rpd == 20
    assert qm.for_model("gemini-3.6-flash").limits.rpm == 5
    # The already-created limiter instance must see it too, not just new ones.
    assert limiter.limits.rpd == 20


def test_learned_limits_survive_to_the_next_run(tmp_path):
    qm = QuotaManager(state_dir=tmp_path, default=Limits())
    qm.learn("gemini-3.6-flash", parse_quota_violations(FakeAPIError(REAL_429)))

    fresh = QuotaManager(state_dir=tmp_path, default=Limits())
    learned = fresh.load_learned()
    assert learned["gemini-3.6-flash"].rpd == 20
    assert fresh.for_model("gemini-3.6-flash").limits.rpd == 20


def test_learning_only_other_models_leaves_this_one_alone(tmp_path):
    qm = QuotaManager(state_dir=tmp_path, default=Limits(rpd=1_500))
    qm.learn("gemini-3.6-flash", parse_quota_violations(FakeAPIError(REAL_429)))
    assert qm.for_model("gemini-3.5-flash").limits.rpd == 1_500


def test_quota_counters_are_per_model(tmp_path):
    """Free-tier metering is PerProjectPerModel, so one model's spend must not
    count against another's — that is what makes the three-role split buy
    throughput."""
    qm = QuotaManager(state_dir=tmp_path, default=Limits(rpm=100, tpm=10**9, rpd=3))
    for _ in range(3):
        qm.for_model("model-a").consume()

    assert qm.for_model("model-a").remaining_today == 0
    assert qm.for_model("model-b").remaining_today == 3
