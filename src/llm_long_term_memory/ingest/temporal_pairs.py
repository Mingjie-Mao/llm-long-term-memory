"""Fixed, package-owned benchmark for deciding whether Stage B may build timelines.

The CLI uses this fixture before a costly ingest. Keeping it in the installed
package, rather than importing from ``tests/``, makes that operational check work
from a wheel and makes the gate artifact reproducible outside the source checkout.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TemporalPair:
    before: str
    after: str
    attribute: str
    expected: str  # replaces | coexists | removes
    note: str = ""


REPLACEMENT: tuple[TemporalPair, ...] = (
    TemporalPair(
        "The user lives in Canberra.",
        "The user moved to Sydney in August 2023.",
        "home city",
        "replaces",
    ),
    TemporalPair(
        "The user uses TensorFlow for machine learning.",
        "The user switched completely to PyTorch in August 2023.",
        "ML framework",
        "replaces",
    ),
    TemporalPair(
        "The user works as a software engineer.",
        "The user now works as a solicitor.",
        "job title",
        "replaces",
    ),
    TemporalPair(
        "The user's bedtime is 11:30 pm.",
        "The user's bedtime is now 10:30 pm.",
        "bedtime",
        "replaces",
    ),
    TemporalPair(
        "The user's phone is an iPhone 12.",
        "The user upgraded to an iPhone 15 in September 2023.",
        "phone",
        "replaces",
    ),
    TemporalPair(
        "The user weighs 82 kg.",
        "The user weighs 76 kg after three months at the gym.",
        "weight",
        "replaces",
    ),
    TemporalPair(
        "The user is training for a 10k.",
        "The user has switched their goal to a half marathon.",
        "running goal",
        "replaces",
    ),
)

COEXISTENCE: tuple[TemporalPair, ...] = (
    TemporalPair(
        "The user owns a peace lily.",
        "The user bought a snake plant in March 2023.",
        "houseplants",
        "coexists",
    ),
    TemporalPair(
        "The user read 'Dune' by Frank Herbert.",
        "The user read 'Project Hail Mary' in April 2023.",
        "books read",
        "coexists",
    ),
    TemporalPair(
        "The user visited Kyoto in May 2023.",
        "The user visited Osaka in May 2023.",
        "trips",
        "coexists",
    ),
    TemporalPair(
        "The user has a sister named Mei.",
        "The user has a brother named Ken.",
        "siblings",
        "coexists",
    ),
    TemporalPair(
        "The user spent $45 on running shoes.",
        "The user spent $120 on a jacket in June 2023.",
        "purchases",
        "coexists",
    ),
    TemporalPair(
        "The user is allergic to peanuts.",
        "The user is allergic to shellfish.",
        "allergies",
        "coexists",
    ),
    TemporalPair(
        "The user has a goal of learning Japanese.",
        "The user has a goal of learning to sail.",
        "goals",
        "coexists",
    ),
    TemporalPair(
        "The user attended a pottery class on 12 March.",
        "The user attended a cooking class on 19 March.",
        "classes attended",
        "coexists",
    ),
    TemporalPair(
        "The user has a cooking class on Wednesday evenings.",
        "The user has game nights on Fridays.",
        "recurring commitments",
        "coexists",
    ),
    TemporalPair(
        "The user owns a Fitbit Charge 3.",
        "The user owns a set of resistance bands.",
        "fitness equipment",
        "coexists",
    ),
)

REMOVAL: tuple[TemporalPair, ...] = (
    TemporalPair(
        "The user drinks two cups of coffee each morning.",
        "The user no longer drinks coffee.",
        "coffee habit",
        "removes",
    ),
    TemporalPair(
        "The user owns a Honda Civic.",
        "The user sold their Honda Civic in July 2023.",
        "car",
        "removes",
    ),
    TemporalPair(
        "The user subscribes to Netflix.",
        "The user cancelled their Netflix subscription.",
        "streaming subscription",
        "removes",
    ),
    TemporalPair(
        "The user is taking a Spanish course.",
        "The user finished the Spanish course in June 2023.",
        "Spanish course",
        "removes",
    ),
)

AMBIGUOUS: tuple[TemporalPair, ...] = (
    TemporalPair(
        "The user is considering buying a Sonos One.",
        "The user is considering buying a Google Home.",
        "speaker being considered",
        "coexists",
    ),
    TemporalPair(
        "The user enjoys hiking.", "The user enjoys rock climbing.", "hobbies", "coexists"
    ),
    TemporalPair(
        "The user mentioned feeling tired on 3 March.",
        "The user mentioned feeling tired on 15 April.",
        "tiredness",
        "coexists",
    ),
    TemporalPair(
        "The user's favourite restaurant is Nobu.",
        "The user tried a new restaurant, Sokyo, and liked it.",
        "restaurants",
        "coexists",
    ),
)

ALL_PAIRS: tuple[TemporalPair, ...] = REPLACEMENT + COEXISTENCE + REMOVAL + AMBIGUOUS
