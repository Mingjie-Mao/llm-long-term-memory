"""The two-stage write path.

Most of these pin the index mapping. Stage A returns facts grouped by session,
Stage B keys a flattened list positionally, and the results are zipped back — an
off-by-one anywhere in that round trip attaches a fact to the wrong session or the
wrong attribute, and nothing downstream would notice: the counts stay right and the
sentences stay true.
"""

from __future__ import annotations

import json

from llm_long_term_memory.evaluation.datasets.longmemeval import HaystackSession, HaystackTurn
from llm_long_term_memory.ingest.extract_facts import FactExtractor
from llm_long_term_memory.ingest.keying import DEFAULT, FactKeyer, UpdateOp, _normalise
from llm_long_term_memory.ingest.two_stage import TwoStageExtractor


class ScriptedClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.prompts: list[str] = []

    def generate(self, *, role, model, prompt, **kw):
        self.prompts.append(prompt)
        payload = self.payloads.pop(0)

        class C:
            text = payload if isinstance(payload, str) else json.dumps(payload)
            input_tokens = 100
            output_tokens = 30
            thinking_tokens = 0
            api_latency_ms = 5.0

        return C()


def session(sid: str, date: str, *lines: str) -> HaystackSession:
    return HaystackSession(
        session_id=sid,
        date=date,
        turns=[HaystackTurn(role="user", content=line) for line in lines],
    )


# ------------------------------------------------------------------ stage A


def test_facts_are_attributed_to_the_session_they_are_grouped_under():
    """Grouping makes attribution structural. The single-stage extractor carried a
    `session_index` field per fact and could mis-assign it; here the model cannot."""
    sessions = [session("s0", "2023/01/05", "a"), session("s1", "2023/03/12", "b")]
    client = ScriptedClient(
        [{"sessions": [{"session_index": 1, "facts": [{"content": "The user moved to Sydney."}]}]}]
    )
    outcome = FactExtractor(client, "m").extract(sessions)

    assert [f.content for f in outcome.by_session["s1"]] == ["The user moved to Sydney."]
    assert outcome.by_session["s0"] == []
    assert outcome.dropped_bad_index == 0


def test_an_invented_session_index_is_counted_not_misfiled():
    sessions = [session("s0", "2023/01/05", "a")]
    client = ScriptedClient(
        [
            {
                "sessions": [
                    {"session_index": 7, "facts": [{"content": "orphan"}, {"content": "orphan2"}]}
                ]
            }
        ]
    )
    outcome = FactExtractor(client, "m").extract(sessions)

    assert outcome.total == 0
    assert outcome.dropped_bad_index == 2


def test_blank_facts_are_dropped():
    sessions = [session("s0", "2023/01/05", "a")]
    client = ScriptedClient(
        [
            {
                "sessions": [
                    {"session_index": 0, "facts": [{"content": "  "}, {"content": "real fact"}]}
                ]
            }
        ]
    )
    facts = FactExtractor(client, "m").extract(sessions).by_session["s0"]
    assert [f.content for f in facts] == ["real fact"]


def test_an_empty_batch_costs_no_request():
    client = ScriptedClient([])
    assert FactExtractor(client, "m").extract([]).total == 0
    assert client.prompts == []


# ------------------------------------------------------------------ stage B


def test_keys_are_assigned_by_index_not_by_order_of_reply():
    """The model may return assignments in any order, or skip one. Positional
    placement means a skipped index leaves a safe default rather than shifting every
    later fact onto someone else's attribute."""
    client = ScriptedClient(
        [
            {
                "facts": [
                    {
                        "index": 2,
                        "temporal_key": "home_city",
                        "update_op": "replaces",
                        "object": "Sydney",
                    },
                    {
                        "index": 0,
                        "temporal_key": "houseplants",
                        "update_op": "coexists",
                        "object": "fern",
                    },
                ]
            }
        ]
    )
    out = FactKeyer(client, "m").key(["owns a fern", "unmentioned", "moved to Sydney"])

    assert out[0].temporal_key == "houseplants"
    assert out[1] == DEFAULT, "the skipped one gets the safe default, not a neighbour's key"
    assert out[2].temporal_key == "home_city"
    assert out[2].update_op is UpdateOp.REPLACES


def test_an_out_of_range_index_cannot_corrupt_a_real_fact():
    client = ScriptedClient(
        [{"facts": [{"index": 99, "temporal_key": "bogus", "update_op": "replaces"}]}]
    )
    out = FactKeyer(client, "m").key(["one fact"])
    assert out == [DEFAULT]


def test_keys_are_normalised_so_wording_does_not_split_an_attribute():
    assert _normalise("Home City") == "home_city"
    assert _normalise("ml-framework") == "ml_framework"
    assert _normalise("  ML   Framework  ") == "ml_framework"
    assert _normalise("") == "states"


def test_coexists_is_the_default_when_the_model_omits_the_op():
    client = ScriptedClient([{"facts": [{"index": 0, "temporal_key": "houseplants"}]}])
    out = FactKeyer(client, "m").key(["owns a fern"])
    assert out[0].update_op is UpdateOp.COEXISTS, "the safe default, never replaces"


def test_keying_an_empty_list_costs_no_request():
    client = ScriptedClient([])
    assert FactKeyer(client, "m").key([]) == []
    assert client.prompts == []


# ------------------------------------------------------------- both stages


def _two_stage_client(facts_by_index, keyings):
    return ScriptedClient([{"sessions": facts_by_index}, {"facts": keyings}])


def test_a_fact_keeps_its_session_across_the_flatten_and_regroup():
    """The round trip that could silently misfile everything: facts are grouped by
    session, flattened for keying, then zipped back."""
    sessions = [session("s0", "2023/01/05", "a"), session("s1", "2023/08/02", "b")]
    client = _two_stage_client(
        [
            {"session_index": 0, "facts": [{"content": "The user owns a fern."}]},
            {"session_index": 1, "facts": [{"content": "The user moved to Sydney."}]},
        ],
        [
            {"index": 0, "temporal_key": "houseplants", "update_op": "coexists", "object": "fern"},
            {"index": 1, "temporal_key": "home_city", "update_op": "replaces", "object": "Sydney"},
        ],
    )
    memories = TwoStageExtractor(client, "m", user_id="u1").extract(sessions).memories
    by_content = {m.content: m for m in memories}

    fern = by_content["The user owns a fern."]
    assert fern.source_session_id == "s0"
    assert fern.predicate == "houseplants"
    assert fern.replaces_previous is False

    moved = by_content["The user moved to Sydney."]
    assert moved.source_session_id == "s1"
    assert moved.observed_at.month == 8, "the date comes from its own session, not the batch"
    assert moved.event_time is None, "'The user moved to Sydney.' names no date"
    assert moved.replaces_previous is True


def test_removes_also_closes_the_earlier_fact():
    """`removes` and `replaces` both end an attribute; only the successor differs.
    The resolver acts on the boolean, so both must set it or a sale is not recorded
    as ending ownership."""
    sessions = [session("s0", "2023/07/01", "a")]
    client = _two_stage_client(
        [{"session_index": 0, "facts": [{"content": "The user sold their Honda Civic."}]}],
        [
            {
                "index": 0,
                "temporal_key": "car",
                "update_op": "removes",
                "object": "",
                "target_object": "Honda Civic",
            }
        ],
    )
    (memory,) = TwoStageExtractor(client, "m").extract(sessions).memories

    assert memory.update_op == "removes"
    assert memory.replaces_previous is True
    assert memory.object == ""
    assert memory.target_object == "Honda Civic"


def test_coexists_never_sets_the_closing_flag():
    sessions = [session("s0", "2023/07/01", "a")]
    client = _two_stage_client(
        [{"session_index": 0, "facts": [{"content": "The user bought a snake plant."}]}],
        [
            {
                "index": 0,
                "temporal_key": "houseplants",
                "update_op": "coexists",
                "object": "snake plant",
            }
        ],
    )
    (memory,) = TwoStageExtractor(client, "m").extract(sessions).memories
    assert memory.replaces_previous is False


def test_the_reported_request_count_matches_what_was_spent():
    """The pipeline budgets from this number. It read one per batch while the
    two-stage path spent two, so a full ingest was planned at half its true size."""
    sessions = [session("s0", "2023/07/01", "a")]
    client = _two_stage_client(
        [{"session_index": 0, "facts": [{"content": "The user owns a fern."}]}],
        [{"index": 0, "temporal_key": "houseplants", "update_op": "coexists"}],
    )
    outcome = TwoStageExtractor(client, "m").extract(sessions)

    assert outcome.requests == 2
    assert len(client.prompts) == 2


def test_an_empty_stage_a_result_does_not_charge_a_stage_b_request():
    sessions = [session("s0", "2023/07/01", "a")]
    client = _two_stage_client([], [])

    outcome = TwoStageExtractor(client, "m").extract(sessions)

    assert outcome.memories == []
    assert outcome.requests == 1
    assert len(client.prompts) == 1


def test_an_assistant_fact_is_attributed_to_the_assistant():
    sessions = [session("s0", "2023/07/01", "a")]
    client = _two_stage_client(
        [
            {
                "session_index": 0,
                "facts": [
                    {
                        "content": "The assistant recommended Mod Podge.",
                        "source_role": "assistant",
                        "subject": "assistant",
                        "scope": "recommendation",
                    }
                ],
            }
        ],
        [{"index": 0, "temporal_key": "assistant_recommendation", "update_op": "coexists"}],
    )
    (memory,) = TwoStageExtractor(client, "m").extract(sessions).memories
    assert memory.subject == "assistant"
    assert memory.source_role == "assistant"
    assert memory.scope == "recommendation"


def test_the_speaker_and_the_subject_are_independent():
    """The defect this separation exists to fix.

    "Andy wore a blue shirt", said by the user, is a fact about Andy. The previous
    implementation derived `subject` from whether the sentence started with "the
    assistant", so every third-party fact was filed under `user` and questions about
    Andy could not be answered from the store (results/assistant-gap.md).
    """
    sessions = [session("s0", "2023/09/20", "a")]
    client = _two_stage_client(
        [
            {
                "session_index": 0,
                "facts": [
                    {
                        "content": "Andy wears an untidy, stained white shirt in the script.",
                        "source_role": "user",
                        "subject": "andy",
                        "scope": "shared_context",
                    }
                ],
            }
        ],
        [{"index": 0, "temporal_key": "clothing", "update_op": "coexists", "object": "shirt"}],
    )

    (memory,) = TwoStageExtractor(client, "m").extract(sessions).memories

    assert memory.source_role == "user", "the user said it"
    assert memory.subject == "andy", "but it is not about the user"
    assert memory.scope == "shared_context"


def test_an_unrecognised_scope_is_dropped_rather_than_stored():
    """A made-up scope is indistinguishable from a real one once it is in the
    column, and the field is meant to be filterable."""
    sessions = [session("s0", "2023/07/01", "a")]
    client = _two_stage_client(
        [
            {
                "session_index": 0,
                "facts": [{"content": "The user owns a fern.", "scope": "vibes"}],
            }
        ],
        [{"index": 0, "temporal_key": "houseplants", "update_op": "coexists"}],
    )

    (memory,) = TwoStageExtractor(client, "m").extract(sessions).memories

    assert memory.scope is None


def test_stage_a_defaults_keep_a_bare_fact_usable():
    """Stage A may omit the new fields; a fact with no attribution is a user fact,
    which is what every pre-P10 memory in the store actually is."""
    sessions = [session("s0", "2023/07/01", "a")]
    client = _two_stage_client(
        [{"session_index": 0, "facts": [{"content": "The user owns a fern."}]}],
        [{"index": 0, "temporal_key": "houseplants", "update_op": "coexists"}],
    )

    (memory,) = TwoStageExtractor(client, "m").extract(sessions).memories

    assert memory.source_role == "user"
    assert memory.subject == "user"
    assert memory.scope is None


def test_a_fact_repeated_in_one_batch_is_stored_once():
    """Ids are content-hashed, so duplicates would collapse on write and make the
    reported count disagree with the store."""
    sessions = [session("s0", "2023/07/01", "a")]
    client = _two_stage_client(
        [
            {
                "session_index": 0,
                "facts": [
                    {"content": "The user owns a fern."},
                    {"content": "The user owns a fern."},
                ],
            }
        ],
        [
            {"index": 0, "temporal_key": "houseplants", "update_op": "coexists"},
            {"index": 1, "temporal_key": "houseplants", "update_op": "coexists"},
        ],
    )
    assert len(TwoStageExtractor(client, "m").extract(sessions).memories) == 1
