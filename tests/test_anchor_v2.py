"""provenance-v2: weight what is distinctive in a fact, not what it shares.

Registered in `results/prereg-anchor-v2-offline-v1.md`: 87.8% -> 96.5% on train150's
no-gold reference, 87.8% -> 96.2% on dev100. v3 (speaker-restricted) failed the keep
rule and is not used by ingestion.
"""

from __future__ import annotations

from llm_long_term_memory.conversation import ConversationSession, ConversationTurn
from llm_long_term_memory.ingest.provenance import (
    ANCHOR_VERSION,
    attach_source_span,
    source_span_for,
    source_span_for_v2,
)
from llm_long_term_memory.store import Memory

SESSION = ConversationSession(
    session_id="s",
    date="2023/05/20",
    turns=[
        ConversationTurn("user", "The user of the park said the trees in the city are the best."),
        ConversationTurn("assistant", "Sure, tell me more."),
        ConversationTurn("user", "I'm allergic to birch pollen."),
    ],
)
FACT = "The user is allergic to the pollen of the birch trees in the park."


def test_v1_is_led_by_common_words_to_the_earlier_turn():
    assert source_span_for(FACT, SESSION)[0] == 0


def test_v2_follows_the_distinctive_words_to_the_turn_that_says_it():
    assert source_span_for_v2(FACT, SESSION)[0] == 2


def test_new_ingests_use_v2():
    memory = attach_source_span(
        Memory(id="m", user_id="u", type="semantic", content=FACT, token_count=5), SESSION
    )
    assert memory.source_turn_index == 2
    assert ANCHOR_VERSION == "provenance-v2"


def test_a_fact_sharing_nothing_with_the_session_has_no_anchor():
    assert source_span_for_v2("Kayaks float.", SESSION) is None
