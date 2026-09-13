"""Route a question to the relation an exhaustive scan should scan — or abstain.

Scanning one relation returns every member of a set: measured at 100% coverage on 68.5
median tokens, against top-20 similarity's 58.3% on 439
(`results/analysis/exhaustive-scan.json`). That is the only mechanism measured so far
that closes the `count` gap, where 43% of probes cannot be answered from the retrieved
context at all.

The catch is that the measured scan was **told** which relation to scan. This module is
the missing router, and its most important behaviour is refusing to route.

**Why abstention is the point, not a shortfall.** On 72 real `train150` aggregation
questions, 31 route with a margin below 0.05 — the top two relations effectively tied.
Worse, the *confident* routes are confidently wrong about what is being counted:

    watched   margin 0.26   "How many hours did I spend watching documentaries?"   counts hours
    read      margin 0.23   "How many days had passed since I finished reading X?" counts days
    know_p    margin 0.20   "How many babies were born to friends recently?"       no such relation

A wrong route is worse than no route, because a scan returns a *complete* set of the
wrong relation, and completeness is exactly what makes an answer look trustworthy. So
below the margin this returns nothing and the caller keeps today's similarity retrieval.
At 0.15 that is 25% routed and 75% abstaining, which is the correct behaviour given how
little of real question space the vocabulary covers.

**`owns` and `acquired` are one routing target.** Every routing error in the templated
upper bound was that pair — twelve `owns → acquired`, two the other way, and nothing else
confused with anything. Buying something makes you its owner, so the vocabulary split one
relation in two. The merge is applied *here* rather than in `predicate-map.csv`: the map
is hashed into the probe set, which is hashed into the held-out split, and rewriting it
would silently destroy the only unseen check v4 has.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# One routing class per line, mapping to the stored relation names it should scan.
# A class with several names is scanned as their union.
ROUTING_CLASSES: dict[str, tuple[str, ...]] = {
    "acquired": ("acquired", "owns"),
    "ate_at": ("ate_at",),
    "attended": ("attended",),
    "completed": ("completed",),
    "cooked": ("cooked",),
    "financial_activity": ("financial_activity",),
    "grows": ("grows",),
    "knows_person": ("knows_person",),
    "listened_to": ("listened_to",),
    "practises_hobby": ("practises_hobby",),
    "read": ("read",),
    "used_service": ("used_service",),
    "visited": ("visited",),
    "watched": ("watched",),
}

# Described by what a question about them would ask, never by the relation's own name:
# glossing `visited` as "visited" would route by string identity and measure nothing.
CLASS_GLOSS: dict[str, str] = {
    "acquired": "buying, getting, receiving or already possessing an item",
    "ate_at": "eating at a restaurant, cafe or food place",
    "attended": "going to a concert, wedding, workshop or other event",
    "completed": "finishing a task, project or course",
    "cooked": "preparing or making a dish or meal at home",
    "financial_activity": "an investment, payment, saving or other money transaction",
    "grows": "keeping or growing a plant or garden",
    "knows_person": "a friend, relative, colleague or other person the user knows",
    "listened_to": "listening to music, an album, a song or a podcast",
    "practises_hobby": "a hobby, sport or leisure activity the user does",
    "read": "reading a book, article or publication",
    "used_service": "using an app, subscription or delivery service",
    "visited": "going to a place, city, country, museum or shop",
    "watched": "watching a film, television series or video",
}

# 0.15 routes 25% of real aggregation questions and abstains on 75%. Chosen from the
# margin distribution rather than tuned for a score: below 0.05 the top two relations
# are tied, and every hand-read route between 0.05 and 0.15 was wrong about what the
# question counted.
DEFAULT_MARGIN = 0.15


@dataclass(frozen=True, slots=True)
class Route:
    """Where to scan, or `relations=()` meaning: do not scan, use similarity retrieval."""

    relations: tuple[str, ...]
    predicted_class: str | None
    similarity: float
    margin: float

    @property
    def routed(self) -> bool:
        return bool(self.relations)


class RelationRouter:
    def __init__(self, encoder, *, margin: float = DEFAULT_MARGIN) -> None:
        self.encoder = encoder
        self.margin = margin
        self._classes = sorted(ROUTING_CLASSES)
        matrix = np.asarray(
            encoder.encode([CLASS_GLOSS[name] for name in self._classes]), dtype=np.float32
        )
        self._gloss = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)

    def route(self, question: str) -> Route:
        vector = np.asarray(self.encoder.encode([question]), dtype=np.float32)[0]
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        scores = self._gloss @ vector
        order = np.argsort(-scores)
        best, second = int(order[0]), int(order[1])
        margin = float(scores[best] - scores[second])
        name = self._classes[best]
        if margin < self.margin:
            # Deliberately reports what it *would* have said. A router that hides its
            # near-miss cannot be tuned or audited later.
            return Route((), name, float(scores[best]), margin)
        return Route(ROUTING_CLASSES[name], name, float(scores[best]), margin)
