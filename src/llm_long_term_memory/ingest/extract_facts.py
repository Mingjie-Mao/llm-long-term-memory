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
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from llm_long_term_memory.conversation import ConversationSession
from llm_long_term_memory.llm.client import GeminiClient

#   two-stage-v1   user-profile framing; subject derived from a string prefix
#   two-stage-p10-v2  2026-08-14: evidence-bound framing, source_role separated from
#                     subject, scope emitted, worked examples for assistant facts and
#                     third-party subjects
EXTRACTOR_VERSION = "two-stage-p10-v2"

FACTS_SYSTEM = """\
You read chat sessions and write down, one line each, everything that might need to \
be referred back to in a later conversation — no matter who said it.

You are transcribing, not summarising. A line that reads as true but has lost the \
number, the date, the URL, or the product name has failed — it will be recalled \
later and answer the question wrongly.\
"""

_PROMPT = """\
Below are {n} chat sessions between a user and an assistant, each with its date and \
index.

For each session, list the facts, events, preferences, plans, recommendations and \
decisions that someone might need to look up later. One short sentence per fact.

**Record them regardless of who said them.** The user asking "what did you recommend \
for the cork board?" three sessions later needs the assistant's answer to have been \
written down. A session where the user only asked a question and the assistant \
supplied the substance is a session whose substance came from the assistant.

For every line, also report:

- `source_role` — who said it: `user`, `assistant`, or `system`
- `subject` — who or what the fact is ABOUT. **This is not the speaker.** If the \
user says "Andy wore a blue shirt", the source_role is `user` and the subject is \
`andy`. Use `user` for facts about the user, `assistant` for what the assistant \
recommended or committed to, and the entity's own name otherwise.
- `scope` — one of: `profile` (stable traits, possessions, relationships), \
`preference` (likes, dislikes, choices), `plan` (intentions, bookings, deadlines), \
`recommendation` (something the assistant suggested), `commitment` (something the \
assistant agreed to do), `event` (something that happened at a time), \
`shared_context` (facts about third parties, documents, or artifacts created in the \
conversation)

Each line must:

- stand alone when read with no context, so resolve every pronoun — "The user's \
sister Mei lives in Osaka", never "She lives there"
- keep every specific exactly as it was given: numbers ("25 postcards", "16GB"), \
durations ("three months", "45 minutes"), dates ("March 15", "in 2019"), relative \
time ("last week", "two months ago" — keep their words and add the resolved date \
where the session date allows), names and URLs ("The Glass Menagerie", "Mod Podge", \
"https://youtube.com/watch?v=abc123"), negations ("no longer drinks coffee"), and \
qualifiers ("about", "usually", "at least")
- carry one fact only. Split compound sentences.

**Never drop a URL, a product or brand name, a figure, or a title** because it \
appeared inside a long list. A numbered list of suggestions is not "generic advice" \
— each named item in it is a separate recordable fact.

## Skip these

- questions, greetings, thanks, and acknowledgements ("Great question!", "Sure!", \
"Happy to help!", "Let me know if you need anything else")
- the assistant describing itself or its own capabilities
- advice with no named artifact: "The assistant suggested staying hydrated" records \
nothing anyone will ask about later
- anything the user only asked about rather than asserted. "I'd like to know more \
about the Sonos One" states nothing about the user.

## Example 1 — detail survival

Session dated 2023-03-15:
  user: "I finally counted my collection last weekend — 25 postcards now, up from 12 \
when I restarted in December. Took me about three months to find the rare Kyoto one."

Facts:
  - "The user's postcard collection reached 25 postcards as of March 2023."
    source_role=user, subject=user, scope=profile
  - "The user had 12 postcards when they restarted collecting in December 2022."
    source_role=user, subject=user, scope=profile
  - "The user restarted collecting postcards in December 2022."
    source_role=user, subject=user, scope=event
  - "The user spent about three months finding a rare Kyoto postcard."
    source_role=user, subject=user, scope=event
  - "The user counted their postcard collection last weekend (around 2023-03-11)."
    source_role=user, subject=user, scope=event

Two sentences, five lines. "25", "12", "December", "three months" and "Kyoto" each \
survive in one. A single line reading "The user collects postcards" would have lost \
all five.

## Example 2 — the substance came from the assistant

Session dated 2023-06-02:
  user: "Can you suggest some DIY decor using recycled materials?"
  assistant: "Sure! 1. Wine Cork Bulletin Board — collect corks and glue them to a \
board. 2. Newspaper Flower Vase — roll newspaper strips, then seal with Mod Podge or \
another sealant so it holds its shape. 3. Tin Can Planters — clean and paint them."
  user: "Love the cork one, I have loads of corks."

Facts:
  - "The assistant recommended making a wine cork bulletin board from collected corks."
    source_role=assistant, subject=assistant, scope=recommendation
  - "The assistant recommended sealing a newspaper flower vase with Mod Podge or \
another sealant."
    source_role=assistant, subject=assistant, scope=recommendation
  - "The assistant recommended making planters from cleaned and painted tin cans."
    source_role=assistant, subject=assistant, scope=recommendation
  - "The user has a large number of wine corks."
    source_role=user, subject=user, scope=profile

The user asserted one thing; the assistant supplied three named projects and a named \
product. "Mod Podge" is exactly what gets asked about later. Writing only "The \
assistant suggested some DIY projects" would lose all of it.

## Example 3 — the fact is about a third party

Session dated 2023-09-20:
  user: "Write a scene where Andy, a 40-something teacher in an untidy, stained \
white shirt, tries to impress the head of department."
  assistant: "INT. STAFF ROOM - DAY. Andy adjusts his collar..."

Facts:
  - "Andy is a man in his 40s who wears an untidy, stained white shirt in the user's \
script."
    source_role=user, subject=andy, scope=shared_context
  - "The user asked the assistant to write a comedy scene featuring a teacher named \
Andy."
    source_role=user, subject=user, scope=event

The first fact was stated by the user but is not about the user. Its subject is \
`andy`. Filing it under `user` would make "what was Andy wearing?" unanswerable.

## Example 4 — what not to record

Session dated 2023-04-01:
  user: "Thanks, that's helpful!"
  assistant: "Great question! I'm happy to help. Remember to stay hydrated and get \
plenty of rest. Let me know if there's anything else!"

Facts: (none)

No named artifact, no assertion, nothing anyone will ask about later.

{sessions}
"""


SourceRole = Literal["user", "assistant", "system"]
SCOPES = (
    "profile",
    "preference",
    "plan",
    "recommendation",
    "commitment",
    "event",
    "shared_context",
)


class FactLine(BaseModel):
    """One extracted fact, plus the two fields only Stage A can determine.

    Stage B sees fact strings with no conversation attached, so `source_role` is
    unrecoverable after this point — if Stage A does not emit it, "what did you
    recommend?" is permanently unanswerable. `subject` is here for the same reason:
    deciding that "Andy wore a blue shirt" is about Andy needs the sentence, not the
    temporal key Stage B computes.
    """

    content: str = Field(description="One self-contained sentence")
    source_role: SourceRole = Field(
        default="user", description="Who said it — NOT who the fact is about"
    )
    subject: str = Field(
        default="user",
        description="Who or what the fact is about: 'user', 'assistant', or an entity name",
    )
    scope: str = Field(default="", description=f"One of: {', '.join(SCOPES)}")

    @field_validator("subject")
    @classmethod
    def _normalize_subject(cls, v: str) -> str:
        return v.strip().lower() or "user"

    @field_validator("scope")
    @classmethod
    def _known_scope(cls, v: str) -> str:
        # An unrecognised scope is stored as NULL rather than kept: a made-up value
        # is indistinguishable from a real one once it is in the column, and this
        # field is meant to be filterable.
        cleaned = v.strip().lower().replace(" ", "_")
        return cleaned if cleaned in SCOPES else ""


class SessionFacts(BaseModel):
    session_index: int = Field(description="0-based index of the session in the batch")
    facts: list[FactLine] = Field(default_factory=list)


class FactsResult(BaseModel):
    sessions: list[SessionFacts] = Field(default_factory=list)


def render_batch(sessions: list[ConversationSession]) -> str:
    blocks = []
    for i, sess in enumerate(sessions):
        turns = "\n".join(f"{t.role}: {t.content}" for t in sess.turns)
        blocks.append(f"=== Session index {i} | date: {sess.date} ===\n{turns}")
    return "\n\n".join(blocks)


@dataclass(slots=True)
class FactExtractionOutcome:
    by_session: dict[str, list[FactLine]]
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

    def extract(self, sessions: list[ConversationSession]) -> FactExtractionOutcome:
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

        by_session: dict[str, list[FactLine]] = {s.session_id: [] for s in sessions}
        bad = 0
        for group in result.sessions:
            if not 0 <= group.session_index < len(sessions):
                bad += len(group.facts)
                continue
            session_id = sessions[group.session_index].session_id
            for line in group.facts:
                content = line.content.strip()
                if content:
                    by_session[session_id].append(line.model_copy(update={"content": content}))
        return FactExtractionOutcome(by_session, bad)
