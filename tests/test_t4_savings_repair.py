import sys
from pathlib import Path

import pytest

from llm_long_term_memory.evaluation.grounded.answering import EvidenceLedger, EvidenceSource

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from t4_savings_repair import quoted_savings


def source(sid, text, role="user"):
    return EvidenceSource(sid, "raw", sid, text, role, "2023-05-22", session_id=sid, turn_index=0)


Q = "How much will I save by taking the train from the airport to my hotel instead of a taxi?"


def test_planned_alternative_price_is_eligible_and_assistant_estimate_cannot_replace_it():
    ledger = EvidenceLedger(
        sources=[
            source(
                "E1",
                "I was thinking of taking the train from the airport to my hotel, "
                "my friend told me it's only around $10.",
            ),
            source("E2", "I was told taking a taxi from the airport to my hotel costs around $60."),
            source("E3", "The train from the airport to the hotel costs ¥3,000.", "assistant"),
        ]
    )
    result = quoted_savings(ledger, Q)
    assert result.answer.startswith("About $50")
    assert result.detail["quotes"]["train"][0]["source"] == "E1"
    assert result.detail["formula"] == "original minus chosen"


def test_direction_and_decimal_precision_are_not_absolute_or_float_arithmetic():
    ledger = EvidenceLedger(
        sources=[
            source("E1", "The train from the airport to my hotel costs $60.15."),
            source("E2", "The taxi from the airport to my hotel costs $10.10."),
        ]
    )
    assert quoted_savings(ledger, Q).detail["result"] == "-50.05"


@pytest.mark.parametrize(
    "bad",
    [
        "The train from the office to my house costs $10.",
        "The train from the airport to my hotel costs £10.",
        "Imagine the train from the airport to my hotel costs $10.",
        "The train from the airport to my hotel does not cost $10.",
        "The train and taxi from the airport to my hotel cost $10 and $60.",
    ],
)
def test_wrong_route_currency_hypothetical_negation_and_mixed_alternatives_are_not_operands(bad):
    ledger = EvidenceLedger(
        sources=[
            source("E1", bad),
            source("E2", "The taxi from the airport to my hotel costs $60."),
        ]
    )
    assert quoted_savings(ledger, Q) is None


def test_repeated_quotes_not_added_but_conflicting_prices_require_resolution():
    a = source("E1", "The train from the airport to my hotel costs $10.")
    b = source("E2", "The taxi from the airport to my hotel costs $60.")
    ledger = EvidenceLedger(sources=[a, b, source("E3", a.text)])
    assert quoted_savings(ledger, Q).detail["result"] == "50"
    ledger.sources.append(source("E4", "The train from the airport to my hotel costs $20."))
    assert quoted_savings(ledger, Q) is None


def test_historical_question_cannot_reuse_prices_without_temporal_selection():
    ledger = EvidenceLedger(
        sources=[
            source("E1", "The train from the airport to my hotel costs $10."),
            source("E2", "The taxi from the airport to my hotel costs $60."),
        ]
    )
    assert quoted_savings(ledger, Q.replace("will", "would") + " As of last year?") is None
