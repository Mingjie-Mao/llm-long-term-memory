"""Extraction that has to show its work: atomic facts, each quoted from a real turn.

Detail retention is 36.6% overall, 51.9% on quantities and 57.1% on durations, and it
is the first loss point for 10 of 14 analysed failures. Everything downstream inherits
that: a memory that dropped the number will be retrieved perfectly and answer wrongly.

The instruction to stay atomic is **not** the missing piece — both shipped extractors
already say "one record per fact, split compound sentences", and 36.6% is the number
*with* that instruction. D30 measured where the remaining headroom is and named it: the
next lever is the output schema, not another prompt edit.

So this changes what a memory has to carry rather than what the model is asked to try:

* **`verbatim_span`** — the words from the source turn that support the fact. The model
  returns the *text*, never offsets: asking a language model for `start=124, end=157` is
  asking it to do arithmetic it is bad at, and a wrong offset is indistinguishable from a
  right one until someone reads it. Python finds the offset, and a span that is not in
  the turn is **rejected** rather than stored. Extraction stops being "the model believes
  it extracted correctly" and becomes "every memory cites a turn".

* **Typed fields for what goes missing.** `value` and `unit` beside the sentence, so "12
  days" is a field rather than four characters inside prose that a paraphrase can drop.

* **`event_time` separate from `observed_time`.** Today `event_time` is the *session's*
  date — 4,714 sessions share 3,776 distinct timestamps, so every memory from one
  conversation is stamped alike and "I moved last July" is dated to the day it was
  mentioned. When the expression cannot be resolved, `event_time` is None: the temporal
  resolver already reports undated facts and leaves them alone, which is the correct
  outcome, and a guessed date is worse than an absent one because it ranks.

**Nothing here is wired into the running pipeline.** Changing extraction changes the
ingest fingerprint, and a resumed ingest into a store written by a different extractor is
the defect that cost this project 4,843 memories once already. This is a candidate,
prepared and testable offline. Measured 2026-09-16 against the shipped extractor at a
matched batch size: 64.2% against 78.4%, because requiring a citation cut what got
written to a third. Kept for auditability, which is a different claim; it does not
improve fidelity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

# Deliberately not `re.sub`-normalised beyond case and whitespace. A span that only
# matches after aggressive rewriting is not evidence that the model quoted the turn.
_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Case and whitespace only, so a span may differ in spacing but not in words."""
    return _WS.sub(" ", text).strip().lower()


class GroundedFact(BaseModel):
    """One atomic fact, with the words that support it."""

    content: str = Field(description="One self-contained sentence")
    verbatim_span: str = Field(
        description="The words from the turn that support this fact, copied exactly"
    )
    turn_index: int = Field(description="Which turn of the session the span is in")

    subject: str = Field(default="user")
    attribute: str = Field(default="", description="snake_case, e.g. duration, lives_in")
    value: str = Field(default="", description="The value alone, e.g. 12")
    unit: str = Field(default="", description="The unit alone, e.g. days")

    # Two different times. `time_expression` is what the user actually said, kept so a
    # later reader can tell a resolved date from an invented one.
    time_expression: str = Field(default="")
    event_time: str | None = Field(default=None, description="ISO; None when unresolvable")


class GroundedResult(BaseModel):
    facts: list[GroundedFact] = Field(default_factory=list)


@dataclass(slots=True)
class Anchored:
    """A fact whose span was found in the turn it named."""

    fact: GroundedFact
    turn_index: int
    char_start: int
    char_end: int


@dataclass(slots=True)
class GroundingReport:
    anchored: list[Anchored] = field(default_factory=list)
    rejected: list[tuple[GroundedFact, str]] = field(default_factory=list)

    @property
    def grounded_rate(self) -> float | None:
        total = len(self.anchored) + len(self.rejected)
        return len(self.anchored) / total if total else None


def locate(span: str, turn_text: str) -> tuple[int, int] | None:
    """Where `span` sits in `turn_text`, or None.

    Tried exactly first, then case- and whitespace-insensitively, because a model that
    copies a span correctly may still normalise a double space. Anything looser would
    accept a paraphrase, which is the thing being guarded against.
    """
    if not span.strip():
        return None
    direct = turn_text.find(span)
    if direct != -1:
        return direct, direct + len(span)

    target = normalise(span)
    if not target:
        return None
    # Walk word starts so the returned offsets index the original text, not the
    # normalised copy — an offset into a string nobody stored is not provenance.
    starts = [m.start() for m in re.finditer(r"\S+", turn_text)]
    for start in starts:
        for end in range(len(turn_text), start, -1):
            if end - start < len(target):
                break
            if normalise(turn_text[start:end]) == target:
                return start, end
    return None


def ground(result: GroundedResult, turns: list[str]) -> GroundingReport:
    """Anchor every fact to its turn, refusing the ones that cannot be.

    A rejected fact is not repaired here. Repair costs a request and belongs to whoever
    is willing to spend one; this reports what would need repairing.
    """
    report = GroundingReport()
    for fact in result.facts:
        if not 0 <= fact.turn_index < len(turns):
            report.rejected.append((fact, f"turn {fact.turn_index} is not in this session"))
            continue
        found = locate(fact.verbatim_span, turns[fact.turn_index])
        if found is None:
            report.rejected.append((fact, "the span is not in the turn it names"))
            continue
        report.rejected.extend(_typed_field_problems(fact, turns[fact.turn_index]))
        if any(f is fact for f, _ in report.rejected):
            continue
        report.anchored.append(Anchored(fact, fact.turn_index, *found))
    return report


def _typed_field_problems(fact: GroundedFact, turn_text: str) -> list[tuple[GroundedFact, str]]:
    """The value must be in its own span; the unit only has to be in the turn.

    The value is the thing that can be silently wrong, so it is held to the span: a typed
    number that disagrees with its own evidence is worse than prose, because it arrives
    looking checked.

    The unit is held to the turn instead, because units are routinely anaphoric. "I
    stayed 12 days, and 3 of them I was sick" states a duration of 3 days in a clause
    that never says "days". Requiring the unit inside the span would leave the model two
    options on a fact like that — invent a span, or drop the fact — and both are worse
    than reading the unit from the sentence that supplied it. Found by a fixture built
    from a real example rather than by argument.
    """
    if not fact.value:
        return []
    if normalise(fact.value) not in normalise(fact.verbatim_span):
        return [(fact, f"value {fact.value!r} is not in its own span")]
    if fact.unit and normalise(fact.unit) not in normalise(turn_text):
        return [(fact, f"unit {fact.unit!r} is nowhere in the turn")]
    return []


def uncovered_specifics(source_text: str, contents: list[str]) -> dict[str, set[str]]:
    """Specifics stated in the source that no extracted fact repeats. Zero model calls.

    The obvious design is a second model call asking "did you miss anything". This
    project has been burned three times by an instrument that judged its own subject
    (D30: the low score was the metric, not the system, three times running), and a
    verifier costs a request per batch whether or not anything was missed.

    `fidelity.PATTERNS` already finds the numbers, durations, dates, money and names in a
    turn, deterministically and for free. So the coverage check is free, and a repair
    request is spent only when this finds something.
    """
    from .fidelity import _extract_facets

    stated = _extract_facets(source_text)
    kept = _extract_facets(" ".join(contents))
    return {
        facet: values - kept.get(facet, set())
        for facet, values in stated.items()
        if values - kept.get(facet, set())
    }


#   grounded-v1  2026-09-15: atomic facts carrying the span that supports them, typed
#                value/unit, and event time separated from observation time
GROUNDED_EXTRACTOR_VERSION = "grounded-v1"

GROUNDED_SYSTEM = """\
You read one chat session and write down every fact someone might need to look up \
later, each with the exact words from the conversation that support it.

You are not summarising. A record that reads as true but has dropped the number, the \
duration or the date has failed, because it will be recalled later and answer the \
question wrongly.\
"""

_GROUNDED_PROMPT = """\
Below is one chat session. Turns are numbered from 0.

{turns}

Write one record per fact. A sentence containing three facts produces three records, \
not one that mentions all three.

For each record:

- `content` — one self-contained sentence. Resolve pronouns. Keep every specific \
exactly as given.
- `verbatim_span` — **copy the words from the turn that support this fact, character \
for character.** Do not paraphrase, do not tidy, do not translate. If you cannot copy \
a span that supports the fact, do not write the record.
- `turn_index` — the number of the turn the span came from.
- `subject` — who or what the fact is about. `user` for facts about the user.
- `attribute` — a snake_case name for what the fact states: `duration`, `lives_in`, \
`owns`, `cost`, `count`.
- `value` and `unit` — when the fact carries a number, split it out: value `12`, unit \
`days`. **Both must appear inside `verbatim_span`.** Leave them empty otherwise.
- `time_expression` — the words the user used for when it happened: "last July", \
"three months ago", "on March 15". Empty when they gave none.
- `event_time` — when it happened, as ISO (`2025-07`, `2025-07-14`), resolved against \
the session date {session_date}. **When you cannot resolve it, leave it null.** Null is \
correct and a guessed date is not: the date is used for ordering, so a wrong one \
reorders a timeline.

Never use the session's date as `event_time` merely because the fact was mentioned \
then. That is when it was *said*, which is recorded separately.

## Example

Session dated 2026-03-02, turn 7:
  user: "I stayed in Xinjiang for 12 days last July, and 3 of them I was sick and \
didn't leave the room."

Records:
  1. content: "The user's Xinjiang trip lasted 12 days."
     verbatim_span: "I stayed in Xinjiang for 12 days"
     turn_index: 7, subject: user, attribute: duration, value: "12", unit: "days"
     time_expression: "last July", event_time: "2025-07"
  2. content: "The user was sick for 3 days of the Xinjiang trip."
     verbatim_span: "3 of them I was sick"
     turn_index: 7, subject: user, attribute: illness_duration, value: "3", unit: "days"
     time_expression: "last July", event_time: "2025-07"
  3. content: "The user did not leave the room on the 3 days they were sick in Xinjiang."
     verbatim_span: "didn't leave the room"
     turn_index: 7, subject: user, attribute: stayed_indoors
     time_expression: "last July", event_time: "2025-07"

One sentence, three records. "12 days", "3" and "didn't leave the room" each survive in \
one, and each cites the words it came from. A single record reading "The user travelled \
to Xinjiang last July" would have lost all three.

## Skip

- questions, greetings, thanks and acknowledgements
- the assistant describing its own capabilities
- anything the user only asked about rather than asserted: "I'd like to know more about \
the Sonos One" states nothing about the user
"""


def render_session(turns: list[tuple[str, str]]) -> str:
    """Numbered turns, because `turn_index` has to mean something the model can see."""
    return "\n".join(f"[{i}] {role}: {content}" for i, (role, content) in enumerate(turns))


class GroundedExtractor:
    """Stage A, with every fact required to cite the turn it came from.

    Returns the anchored facts and the report of what was refused, rather than silently
    dropping the refusals: the rejection rate is the measurement that says whether the
    model can do this at all, and a pipeline that hides it cannot be tuned.
    """

    version = GROUNDED_EXTRACTOR_VERSION

    def __init__(self, client, model: str, chars_per_token: float = 4.6) -> None:
        self.client = client
        self.model = model
        self.chars_per_token = chars_per_token

    @staticmethod
    def prompt_texts() -> tuple[str, ...]:
        return (GROUNDED_SYSTEM, _GROUNDED_PROMPT)

    def extract(self, turns: list[tuple[str, str]], session_date: str) -> GroundingReport:
        prompt = _GROUNDED_PROMPT.format(
            turns=render_session(turns), session_date=session_date or "unknown"
        )
        completion = self.client.generate(
            role="extractor",
            model=self.model,
            prompt=prompt,
            system=GROUNDED_SYSTEM,
            schema=GroundedResult,
            temperature=0.0,
            est_input_tokens=int(len(prompt) / self.chars_per_token),
        )
        result = GroundedResult.model_validate_json(completion.text)
        return ground(result, [content for _, content in turns])
