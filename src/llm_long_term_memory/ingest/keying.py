"""Stage B, LLM-only: give each fact a temporal key and say what it does to history.

The rule-based structurer reached 43% predicate accuracy and 37.5% resolvable
key-consistency, and produced zero supersessions over 148 real memories. Regexes
cannot key open-domain relations; that was measured rather than assumed.

**`update_op` replaces arity as the gatekeeper.** Previously a supersede could only
fire when the predicate appeared in a hand-maintained single-valued list. That list
was wrong twice (D23), and its current contents contain a predicate the extractor
never emits — the abstraction needs rebuilding, not patching. Asking the model
directly what a fact *does* to what came before sidesteps it: the fact itself says
whether it replaces, coexists, or removes, and the resolver acts on that.

The four operations are deliberately not symmetric in cost:

    REPLACES   closes an earlier fact's validity window. The expensive mistake —
               it retires something still true and removes it from every later
               query, so the prompt demands an explicit signal from the user.
    COEXISTS   the default. Two possessions, two goals, two events.
    REMOVES    the fact ends without a successor: sold, cancelled, finished.
    NONE       not about a trackable attribute at all.

Defaulting to COEXISTS rather than REPLACES is the whole safety property. A missing
supersede leaves the store no worse than a flat list; a wrong one makes it
confidently wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field

from llm_long_term_memory.llm.client import GeminiClient


class UpdateOp(StrEnum):
    REPLACES = "replaces"
    COEXISTS = "coexists"
    REMOVES = "removes"
    NONE = "none"


KEYING_SYSTEM = """\
You assign each fact about a user the attribute it belongs to, and say what it does \
to what the user said before.

Getting `replaces` wrong is expensive: it retires an earlier fact permanently. When \
in doubt, say `coexists`.\
"""

_PROMPT = """\
For each numbered fact, return three things.

**temporal_key** — a short snake_case name for the *attribute* this fact is about. \
It is a key, so two facts about the same attribute must get the same one however \
differently they are worded:

    "The user uses TensorFlow"        -> ml_framework
    "The user switched to PyTorch"    -> ml_framework
    "The user lives in Canberra"      -> home_city
    "The user moved to Sydney"        -> home_city
    "The user's bedtime is 11:30pm"   -> bedtime
    "The user's bedtime is now 10:30" -> bedtime

Name the attribute, not the verb. "bought a Fitbit" and "owns a Garmin" are both \
`fitness_tracker` if the user has one at a time, but two separate books the user \
read are both `books_read` — a key can hold many values.

**update_op** — what this fact does to earlier facts on the same key:

- `replaces` — the user signalled that this supersedes an earlier value: "switched \
to", "no longer", "moved to", "stopped", "now", "instead of", "used to". Only with \
such a signal, or when the attribute plainly holds one value at a time (where you \
live, who employs you).
- `coexists` — **the default**. Both can be true at once: two possessions, two \
goals, two trips, two purchases. Use this whenever you are not sure.
- `removes` — the attribute ends with no successor: "sold my car", "cancelled the \
subscription", "finished the course".
- `none` — not a trackable attribute: a one-off observation, something about the \
assistant, an opinion.

**object** — the value this fact assigns to the key, in a few words. Empty for \
`none`.

## Worked examples

    "The user moved to Sydney in August 2023."
        home_city / replaces / Sydney
    "The user bought a peace lily."
        houseplants / coexists / peace lily
    "The user also bought a snake plant."
        houseplants / coexists / snake plant
    "The user sold their Honda Civic."
        car / removes / Honda Civic
    "The user has been feeling tired lately."
        none / none /
    "The assistant recommended Mod Podge as a sealant."
        assistant_recommendation / coexists / Mod Podge

Two possessions on one key with `coexists` is correct and important: keying them \
together lets "what houseplants do I own?" find both, while `replaces` would have \
thrown the first away.

## Facts

{facts}
"""


class KeyedFact(BaseModel):
    index: int
    temporal_key: str = Field(description="snake_case attribute name")
    update_op: UpdateOp = UpdateOp.COEXISTS
    object: str = ""


class KeyingResult(BaseModel):
    facts: list[KeyedFact] = Field(default_factory=list)


@dataclass(slots=True)
class Keying:
    temporal_key: str
    update_op: UpdateOp
    object: str

    @property
    def replaces_previous(self) -> bool:
        return self.update_op is UpdateOp.REPLACES

    @property
    def trackable(self) -> bool:
        return self.update_op is not UpdateOp.NONE and self.temporal_key not in ("", "none")


DEFAULT = Keying(temporal_key="states", update_op=UpdateOp.NONE, object="")


class FactKeyer:
    """One request per batch of facts, not per fact.

    Facts are one sentence each, so a whole ingestion batch fits comfortably in a
    single call. That keeps Stage B at +1 request per batch — roughly 190 extra
    requests for the dev-subset ingest, inside one day's budget.
    """

    def __init__(self, client: GeminiClient, model: str, chars_per_token: float = 4.6) -> None:
        self.client = client
        self.model = model
        self.chars_per_token = chars_per_token

    def key(self, facts: list[str]) -> list[Keying]:
        if not facts:
            return []

        numbered = "\n".join(f"{i}. {f}" for i, f in enumerate(facts))
        prompt = _PROMPT.format(facts=numbered)
        completion = self.client.generate(
            role="extractor",
            model=self.model,
            prompt=prompt,
            system=KEYING_SYSTEM,
            schema=KeyingResult,
            temperature=0.0,
            est_input_tokens=int(len(prompt) / self.chars_per_token),
        )
        result = KeyingResult.model_validate_json(completion.text)

        # Positional, so a model that skips or invents an index must not shift every
        # later fact onto the wrong key — an off-by-one here would silently attach
        # facts to unrelated attributes.
        out = [DEFAULT] * len(facts)
        for item in result.facts:
            if 0 <= item.index < len(facts):
                out[item.index] = Keying(
                    temporal_key=_normalise(item.temporal_key),
                    update_op=item.update_op,
                    object=item.object.strip(),
                )
        return out


def _normalise(key: str) -> str:
    cleaned = "".join(c if c.isalnum() else "_" for c in key.strip().lower()).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or "states"
