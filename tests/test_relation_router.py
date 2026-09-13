"""The router's most important behaviour is refusing to route.

A wrong route is worse than no route: an exhaustive scan returns a *complete* set of the
wrong relation, and completeness is exactly what makes an answer look trustworthy. On 72
real questions the confident routes were confidently wrong about what was being counted
— hours instead of films, days instead of books, and a `babies` question routed to
`knows_person` because no `babies` relation exists.
"""

from __future__ import annotations

import numpy as np

from llm_long_term_memory.retrieve.relation_router import (
    CLASS_GLOSS,
    DEFAULT_MARGIN,
    ROUTING_CLASSES,
    RelationRouter,
)


class _Encoder:
    """Returns vectors chosen by the test, so routing behaviour is asserted without
    depending on what a real embedding model happens to think."""

    def __init__(self, mapping: dict[str, list[float]], dim: int = 3) -> None:
        self.mapping = mapping
        self.dim = dim

    def encode(self, texts):
        return np.asarray([self.mapping.get(t, [0.0] * self.dim) for t in texts], dtype=np.float32)


def _router(question_vector, margin=DEFAULT_MARGIN):
    """Two classes only, positioned so the test controls the margin exactly."""
    mapping = {CLASS_GLOSS[name]: [0.0, 0.0, 0.0] for name in ROUTING_CLASSES}
    mapping[CLASS_GLOSS["read"]] = [1.0, 0.0, 0.0]
    mapping[CLASS_GLOSS["watched"]] = [0.0, 1.0, 0.0]
    mapping["q"] = question_vector
    return RelationRouter(_Encoder(mapping), margin=margin)


def test_a_clear_winner_is_routed():
    route = _router([1.0, 0.0, 0.0])
    decision = route.route("q")
    assert decision.routed
    assert decision.predicted_class == "read"


def test_a_tie_abstains_rather_than_guessing():
    """Two relations equally close is a coin flip wearing a decision."""
    decision = _router([1.0, 1.0, 0.0]).route("q")
    assert not decision.routed
    assert decision.relations == ()


def test_an_abstention_still_reports_what_it_would_have_said():
    """A router that hides its near-misses cannot be tuned or audited later."""
    decision = _router([1.0, 0.98, 0.0]).route("q")
    assert not decision.routed
    assert decision.predicted_class is not None
    assert 0.0 <= decision.margin < DEFAULT_MARGIN


def test_the_margin_is_configurable_and_actually_binds():
    vector = [1.0, 0.95, 0.0]
    assert not _router(vector, margin=0.5).route("q").routed
    assert _router(vector, margin=0.0).route("q").routed


def test_owns_and_acquired_are_one_routing_target():
    """Every routing error in the templated upper bound was that pair, and buying
    something makes you its owner."""
    assert set(ROUTING_CLASSES["acquired"]) == {"acquired", "owns"}
    assert "owns" not in ROUTING_CLASSES, "owns must not also be its own class"


def test_every_class_is_glossed_by_description_not_by_its_own_name():
    """Glossing `visited` as "visited" would route by string identity and measure
    nothing.

    The check is that a gloss is a *description*, not that the name never appears in
    it: "reading a book, article or publication" shares a stem with `read`, and that is
    the natural way a question would put it too. What is forbidden is a gloss that is
    the identifier itself.
    """
    assert set(CLASS_GLOSS) == set(ROUTING_CLASSES)
    for name, gloss in CLASS_GLOSS.items():
        assert gloss.strip().lower() != name.replace("_", " "), f"{name} glosses itself"
        assert len(gloss.split()) >= 4, f"{name} gloss is too thin to route on"
