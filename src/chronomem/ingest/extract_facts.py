"""Stage A: sessions in, bare fact strings out.

The single-stage extractor asks for nine fields per memory — type, subject,
predicate, object, entities, importance, session_index, replaces_previous, and the
sentence itself. Mem0's, by contrast, returns a flat list of strings. That is the
last untested structural difference between the two, and the hypothesis is that the
per-fact output cost suppresses how many facts get written at all: a model with a
fixed appetite for output tokens produces fewer records when each one is expensive.

So this stage asks for nothing but the sentences. Structuring moves to Stage B,
where most of it is rules.

Two secondary gains fall out of the split:

  * `session_index` stops being a field the model has to carry correctly. Facts are
    grouped under their session, so attribution is structural. The
    `dropped_bad_index` failure mode disappears rather than being monitored.
  * Extraction failures and structuring failures become separately attributable.
    Today a missing `predicate` and a missing fact look the same from the outside.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from chronomem.evaluation.datasets.longmemeval import HaystackSession
from chronomem.llm.client import GeminiClient

FACTS_SYSTEM = """\
You read a user's chat sessions and write down, one line each, everything the user \
said about themselves.

You are transcribing, not summarising. A line that reads as true but has lost the \
number, the date, or the name has failed — it will be recalled later and answer the \
question wrongly.\
"""

_PROMPT = """\
Below are {n} chat sessions with the same user, each with its date and index.

For each session, list the facts the user stated about themselves — one short \
sentence per fact. Nothing else: no categories, no scores, no JSON beyond the shape \
asked for.

Each line must:

- stand alone when read with no context, so resolve every pronoun — "The user's \
sister Mei lives in Osaka", never "She lives there"
- keep every specific exactly as the user gave it: numbers ("25 postcards", "16GB"), \
durations ("three months", "45 minutes"), dates ("March 15", "in 2019"), relative \
time ("last week", "two months ago" — keep their words and add the resolved date \
where the session date allows), names ("The Glass Menagerie", "Mod Podge"), \
negations ("no longer drinks coffee"), and qualifiers ("about", "usually", "at least")
- carry one fact only. Split compound sentences.

Also record concrete things the *assistant* told this user that they might ask about \
later — a named product it recommended, a figure it gave. Write those as "The \
assistant recommended …". Not its generic advice.

Skip: questions, greetings, thanks, and anything the user only asked about rather \
than asserted. "I'd like to know more about the Sonos One" states nothing about the \
user.

## Example

Session dated 2023-03-15:
  user: "I finally counted my collection last weekend — 25 postcards now, up from 12 \
when I restarted in December. Took me about three months to find the rare Kyoto one."

Facts:
  - "The user's postcard collection reached 25 postcards as of March 2023."
  - "The user had 12 postcards when they restarted collecting in December 2022."
  - "The user restarted collecting postcards in December 2022."
  - "The user spent about three months finding a rare Kyoto postcard."
  - "The user counted their postcard collection last weekend (around 2023-03-11)."

Two sentences, five lines. "25", "12", "December", "three months" and "Kyoto" each \
survive in one. A single line reading "The user collects postcards" would have lost \
all five.

{sessions}
"""


class SessionFacts(BaseModel):
    session_index: int = Field(description="0-based index of the session in the batch")
    facts: list[str] = Field(default_factory=list, description="One self-contained sentence each")


class FactsResult(BaseModel):
    sessions: list[SessionFacts] = Field(default_factory=list)


def render_batch(sessions: list[HaystackSession]) -> str:
    blocks = []
    for i, sess in enumerate(sessions):
        turns = "\n".join(f"{t.role}: {t.content}" for t in sess.turns)
        blocks.append(f"=== Session index {i} | date: {sess.date} ===\n{turns}")
    return "\n\n".join(blocks)


@dataclass(slots=True)
class FactExtractionOutcome:
    by_session: dict[str, list[str]]
    dropped_bad_index: int
    """Kept for parity with the single-stage extractor, but grouping makes it
    structural rather than a field the model can get wrong — a nonzero value here
    means the model invented a session number, not that it mis-attributed a fact."""

    @property
    def total(self) -> int:
        return sum(len(v) for v in self.by_session.values())


class FactExtractor:
    def __init__(self, client: GeminiClient, model: str, chars_per_token: float = 4.6) -> None:
        self.client = client
        self.model = model
        self.chars_per_token = chars_per_token

    def extract(self, sessions: list[HaystackSession]) -> FactExtractionOutcome:
        if not sessions:
            return FactExtractionOutcome({}, 0)

        prompt = _PROMPT.format(n=len(sessions), sessions=render_batch(sessions))
        completion = self.client.generate(
            role="extractor",
            model=self.model,
            prompt=prompt,
            system=FACTS_SYSTEM,
            schema=FactsResult,
            temperature=0.0,
            est_input_tokens=int(len(prompt) / self.chars_per_token),
        )
        result = FactsResult.model_validate_json(completion.text)

        by_session: dict[str, list[str]] = {s.session_id: [] for s in sessions}
        bad = 0
        for group in result.sessions:
            if not 0 <= group.session_index < len(sessions):
                bad += len(group.facts)
                continue
            session_id = sessions[group.session_index].session_id
            by_session[session_id].extend(f.strip() for f in group.facts if f.strip())
        return FactExtractionOutcome(by_session, bad)
