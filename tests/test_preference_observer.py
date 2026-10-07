"""The opt-in preference-observer instruction: off by default, visible when on."""

from __future__ import annotations

import json

from llm_long_term_memory.conversation import ConversationSession, ConversationTurn
from llm_long_term_memory.ingest.extract_facts import (
    EXTRACTOR_VERSION,
    PREFERENCE_OBSERVER,
    FactExtractor,
)
from llm_long_term_memory.ingest.two_stage import TwoStageExtractor


class _Client:
    def __init__(self):
        self.prompts = []

    def generate(self, *, prompt, **kw):
        self.prompts.append(prompt)

        class C:
            text = json.dumps({"sessions": []})

        return C()


SESSION = ConversationSession(
    session_id="s", date="2023/05/20", turns=[ConversationTurn("user", "I'm in bed by 9:30.")]
)


def test_the_production_prompt_is_unchanged_by_default():
    client = _Client()
    FactExtractor(client, "m").extract([SESSION])
    assert PREFERENCE_OBSERVER not in client.prompts[0]


def test_the_instruction_is_added_when_asked_for():
    client = _Client()
    FactExtractor(client, "m", preference_observer=True).extract([SESSION])
    assert client.prompts[0].endswith(PREFERENCE_OBSERVER)


def test_turning_it_on_changes_what_the_fingerprint_sees():
    from llm_long_term_memory.ingest import fingerprint

    plain = TwoStageExtractor(_Client(), "m")
    observing = TwoStageExtractor(_Client(), "m", preference_observer=True)
    assert plain.version == EXTRACTOR_VERSION
    assert observing.version != plain.version
    a = fingerprint.from_runtime(plain, sessions_per_request=8).as_dict()
    b = fingerprint.from_runtime(observing, sessions_per_request=8).as_dict()
    assert {line.split(":")[0] for line in fingerprint.differences(a, b)} == {
        "extractor_version",
        "prompts",
    }
