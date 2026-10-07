from dataclasses import replace

from llm_long_term_memory.conversation import AnswerRequest
from llm_long_term_memory.runtime.grounded_answering import (
    EvidenceLedger,
    EvidenceSource,
    consecutive_event_interval,
    latest_inventory_quantity,
    redemption_place,
)


def source(sid, text, date="2026-02-14", session="s", role="user"):
    return EvidenceSource(sid, "raw", sid, text, role, date, session_id=session, turn_index=0)


def test_redemption_context_is_labeled_inference_with_action_and_place_citations():
    ledger = EvidenceLedger(
        sources=[
            source("E1", "I shop at Shop One frequently."),
            source("E2", "I redeemed a $5 coupon on milk, which I found in my email inbox."),
        ]
    )
    result = redemption_place(ledger, "Where did I redeem the coupon?")
    assert result.computed and result.answer.startswith("Likely Shop One")
    assert result.detail["inferred"]
    assert {x["source"] for x in result.detail["location_citations"]} == {"E1", "E2"}


def test_redemption_context_rejects_other_session_assistant_negation_and_ambiguity():
    for neighbor in [
        source("E1", "I shop at Shop One frequently.", session="other"),
        source("E1", "I shop at Shop One frequently.", role="assistant"),
        source("E1", "I don't shop at Shop One."),
        source("E1", "I shop at Shop One. I shop at Shop Two."),
    ]:
        ledger = EvidenceLedger(
            sources=[
                neighbor,
                source("E2", "I redeemed a coupon on milk, which came from my email inbox."),
            ]
        )
        assert not redemption_place(ledger, "Where did I redeem the coupon?").computed


def test_explicit_redemption_place_has_priority_over_other_shopping_context():
    ledger = EvidenceLedger(
        sources=[
            source("E1", "I shop at Shop One."),
            source("E2", "I redeemed a coupon at Shop Two."),
        ]
    )
    result = redemption_place(ledger, "Where did I redeem the coupon?")
    assert result.computed and result.answer == "Shop Two" and not result.detail["inferred"]


def event_ledger():
    return EvidenceLedger(
        sources=[
            source("E1", 'I just got back from the "River Ride" charity event today.'),
            source(
                "E2", 'I volunteered at the "Library Drive" charity event today.', date="2026-02-15"
            ),
            source("E3", 'I did the "Town Walk" charity event today.', date="2026-03-19"),
        ]
    )


def request():
    return AnswerRequest(
        "How many months since I participated in two charity events on consecutive days?",
        "2026-04-18",
        "alice",
    )


def test_consecutive_sequence_uses_end_not_start_or_isolated_later_event():
    result = consecutive_event_interval(event_ledger(), request())
    assert result.computed and result.answer == "2 months and 3 days"
    assert {x["source"] for x in result.detail["sequence_citations"]} == {"E1", "E2"}


def test_consecutive_sequence_rejects_role_plan_duplicate_dates_and_multiple_sequences():
    for change in [
        {"role": "assistant"},
        {"text": 'I will attend the "Library Drive" charity event today.'},
        {"date": "2026-02-14"},
    ]:
        ledger = event_ledger()
        date = change.pop("date", None)
        if date:
            change["conversation_date"] = date
        ledger.sources[1] = replace(ledger.sources[1], **change)
        assert not consecutive_event_interval(ledger, request()).computed
    ledger = event_ledger()
    ledger.sources += [
        source("E4", 'I did the "Town Sale" charity event today.', date="2026-03-20")
    ]
    assert not consecutive_event_interval(ledger, request()).computed


def inventory_ledger():
    return EvidenceLedger(
        sources=[
            source(
                "E1",
                "We have jars stocked up in the pantry - 10 boxes at the moment.",
                date="2026-01-10",
            ),
            source(
                "E2",
                "We have jars stocked up in the pantry - 7 boxes right now.",
                date="2026-02-10",
            ),
        ]
    )


def test_current_inventory_uses_latest_matching_report_not_earlier_larger_quantity():
    result = latest_inventory_quantity(
        inventory_ledger(), "How many boxes of jars do we currently have stocked up in our pantry?"
    )
    assert result.computed and result.answer == "7 boxes jars"
    assert result.detail["quantity_citations"][0]["source"] == "E2"


def test_current_inventory_rejects_wrong_location_role_and_latest_time_conflict():
    question = "How many boxes of jars do we currently have stocked up in our pantry?"
    for change in [
        {"role": "assistant"},
        {"text": "We have jars stocked up in the cellar - 7 boxes right now."},
    ]:
        ledger = inventory_ledger()
        ledger.sources[1] = replace(ledger.sources[1], **change)
        assert latest_inventory_quantity(ledger, question).answer == "10 boxes jars"
    ledger = inventory_ledger()
    ledger.sources.append(
        source("E3", "We have jars in the pantry - 8 boxes right now.", date="2026-02-10")
    )
    assert not latest_inventory_quantity(ledger, question).computed


def test_consecutive_titles_do_not_treat_contractions_as_quote_delimiters():
    ledger = event_ledger()
    ledger.sources[0] = replace(
        ledger.sources[0],
        text=(
            'I\'m feeling tired today, just got back from the "River Ride" '
            "charity event for the children's hospital."
        ),
    )
    result = consecutive_event_interval(ledger, request())
    assert result.computed and result.answer == "2 months and 3 days"
