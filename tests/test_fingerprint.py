"""The ingest fingerprint, and the parity its two builders owe each other.

The bug these pin: preflight computed its expected fingerprint by calling the
builder with the extractor *class* and no deduplicator. `model` is an instance
attribute and `dedup_threshold` came from the deduplicator, so the expected value
carried `model: unknown` and no threshold at all, and could not match any store a
real ingest had written. The check that guards the formal A2 could never pass —
and, having never passed, had never checked anything.

A single builder with two calling conventions is what allowed that. There are now
two named builders and a test that they agree, so the next divergence is a red
test rather than a check that quietly stops being one.
"""

from __future__ import annotations

import pytest

from llm_long_term_memory.config import ExperimentConfig
from llm_long_term_memory.ingest import fingerprint
from llm_long_term_memory.ingest.dedup import Deduplicator
from llm_long_term_memory.ingest.extract import Extractor
from llm_long_term_memory.ingest.pipeline import resolved_sessions_per_request
from llm_long_term_memory.ingest.two_stage import TwoStageExtractor

BASELINES = "configs/baselines.yaml"


def cfg(**overrides):
    """The real config, so parity is asserted against what actually ships."""
    config = ExperimentConfig.from_yaml(BASELINES)
    for dotted, value in overrides.items():
        section, _, field = dotted.partition(".")
        setattr(getattr(config, section), field, value)
    return config


def runtime_from(config, sessions_per_request: int) -> fingerprint.IngestSpec:
    """Build the live objects the way `lltm ingest run` does, then fingerprint them.

    `None` for the client and encoder: constructing them would need an API key and
    a 2GB model download, and neither reaches the fingerprint. What does reach it —
    version, model, prompts, threshold — is set in `__init__` from these same
    arguments, which is exactly the wiring under test.
    """
    extractor = (
        TwoStageExtractor(None, config.models.extractor)
        if config.ingest.two_stage
        else Extractor(None, config.models.extractor)
    )
    dedup = Deduplicator(
        None,
        config.models.extractor,
        None,
        threshold=config.ingest.dedupe_similarity_threshold,
    )
    return fingerprint.from_runtime(
        extractor, sessions_per_request=sessions_per_request, dedup=dedup
    )


# ------------------------------------------------------------------ parity


def test_config_and_runtime_fingerprints_are_identical():
    """The regression. Before the split these could not agree: the config side had
    no way to reach an instance attribute, and the caller that mattered passed a
    class."""
    config = cfg()
    assert fingerprint.from_config(config, sessions_per_request=15) == runtime_from(config, 15)


def test_parity_holds_for_the_single_stage_extractor_too():
    """`two_stage` selects the class on both sides, so it is a way for them to
    diverge. The config builder must follow the flag rather than assume the
    generation that happens to be current."""
    config = cfg(**{"ingest.two_stage": False})
    spec = fingerprint.from_config(config, sessions_per_request=10)

    assert spec == runtime_from(config, 10)
    assert spec.extractor_version == Extractor.version


def test_the_config_builder_names_the_configured_model_not_unknown():
    """The exact shape of the bug, stated as a value rather than a comparison: the
    old call produced `model: unknown` and omitted `dedup_threshold`, which is what
    made every store look like a mismatch."""
    spec = fingerprint.from_config(cfg(), sessions_per_request=15).as_dict()

    assert spec["model"] == cfg().models.extractor
    assert spec["model"] != "unknown"
    assert spec["dedup_threshold"] == "0.9200"


def test_the_config_builder_touches_no_client_or_encoder(monkeypatch):
    """Preflight validates without starting the system. If computing the expected
    fingerprint constructed a Gemini client, the check would fail on a machine
    with no API key for reasons unrelated to the store."""
    import llm_long_term_memory.llm.client as client_module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the config fingerprint must not construct a client")

    monkeypatch.setattr(client_module, "GeminiClient", forbidden)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    assert fingerprint.from_config(cfg(), sessions_per_request=15).model


# ------------------------------------------- every component actually moves


def test_a_reworded_prompt_changes_the_fingerprint(monkeypatch):
    """The failure `extractor_version` alone cannot see."""
    before = fingerprint.from_config(cfg(), sessions_per_request=15)
    monkeypatch.setattr(
        TwoStageExtractor, "prompt_texts", staticmethod(lambda: ("reworded",)), raising=True
    )

    assert fingerprint.from_config(cfg(), sessions_per_request=15).prompts != before.prompts


def test_one_changed_character_is_enough():
    """A digest that only noticed large edits would pass the realistic case."""
    assert fingerprint.digest("extract the facts") != fingerprint.digest("extract the fact")


def test_a_changed_model_changes_the_fingerprint():
    a = fingerprint.from_config(cfg(), sessions_per_request=15)
    b = fingerprint.from_config(
        cfg(**{"models.extractor": "gemma-3-27b-it"}), sessions_per_request=15
    )

    assert a != b
    assert "model" in " ".join(fingerprint.differences(a.as_dict(), b.as_dict()))


def test_a_changed_dedup_threshold_changes_the_fingerprint():
    a = fingerprint.from_config(cfg(), sessions_per_request=15)
    b = fingerprint.from_config(
        cfg(**{"ingest.dedupe_similarity_threshold": 0.93}), sessions_per_request=15
    )

    assert a.dedup_threshold == "0.9200"
    assert b.dedup_threshold == "0.9300"
    assert a != b


def test_a_changed_schema_changes_the_fingerprint(monkeypatch):
    """The schema is part of what produced the rows: a column added mid-corpus
    means earlier memories have it NULL and later ones do not."""
    before = fingerprint.from_config(cfg(), sessions_per_request=15)
    monkeypatch.setattr(fingerprint, "schema_version", lambda: "deadbeef0000")

    assert fingerprint.from_config(cfg(), sessions_per_request=15).schema_version != (
        before.schema_version
    )


def test_a_changed_batch_size_changes_the_fingerprint():
    a = fingerprint.from_config(cfg(), sessions_per_request=15)
    b = fingerprint.from_config(cfg(), sessions_per_request=10)

    assert a != b
    assert fingerprint.differences(a.as_dict(), b.as_dict()) == ["sessions_per_request: 15 -> 10"]


# ------------------------------------------- the effective batch size, shared


def test_the_batch_size_in_the_fingerprint_is_the_effective_one():
    """The configured value is not necessarily what runs. A store fingerprinted
    under a lowered batch size must be compared against the lowered number, or the
    check reports a mismatch that only exists in the comparison."""
    assert resolved_sessions_per_request(15, tpm=250_000) == 15
    assert resolved_sessions_per_request(15, tpm=16_000) == 5, "Gemma's free-tier allowance"

    lowered = resolved_sessions_per_request(15, tpm=16_000)
    assert fingerprint.from_config(cfg(), sessions_per_request=lowered).sessions_per_request == "5"


# ---------------------------------------------------- the stored form is stable


def test_the_stored_form_keeps_the_keys_already_on_disk():
    """The clean P10 store carries these six keys. Renaming or adding one would
    make a healthy store read as a mismatch and cost a re-ingest to fix."""
    stored = fingerprint.from_config(cfg(), sessions_per_request=15).as_dict()

    assert set(stored) == {
        "extractor_version",
        "schema_version",
        "model",
        "sessions_per_request",
        "prompts",
        "dedup_threshold",
    }
    assert all(isinstance(v, str) for v in stored.values())


def test_the_threshold_is_omitted_rather_than_nulled_without_a_deduplicator():
    """Matches the shape of fingerprints written before this refactor, so a store
    from a run with no deduplicator still compares equal to one."""
    spec = fingerprint.from_runtime(TwoStageExtractor(None, "m"), sessions_per_request=15)

    assert spec.dedup_threshold is None
    assert "dedup_threshold" not in spec.as_dict()


def test_dumps_round_trips_through_loads():
    spec = fingerprint.from_config(cfg(), sessions_per_request=15)

    assert fingerprint.loads(fingerprint.dumps(spec)) == spec.as_dict()


@pytest.mark.parametrize("raw", [None, "", "not json", "[1, 2]"])
def test_loads_survives_anything_a_meta_row_might_hold(raw):
    assert fingerprint.loads(raw) == {}
