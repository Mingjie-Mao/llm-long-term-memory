"""Turn raw sessions into candidate memories.

Sessions are extracted in batches (D5: ten per request) because the free tier caps
requests per day, and one-request-per-session costs 12.8 days for a single pass over
LongMemEval-S. Batching creates one problem the prompt has to solve: with ten
conversations in a single call, every extracted fact must say which session it came
from, or its `event_time` cannot be set — and without event_time the temporal layer
in P4 has nothing to order by. Hence `session_index` on every record, validated
against the batch on the way back.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

from chronomem.evaluation.datasets.longmemeval import HaystackSession
from chronomem.llm.client import GeminiClient
from chronomem.store import Memory

from .schemas import COMMON_PREDICATES, ExtractedMemory, ExtractionResult

EXTRACT_SYSTEM = """\
You transcribe what a user said about themselves into individual, self-contained \
records, for a long-term memory system.

You are a transcriber, not a summariser. Summarising is the failure mode: a record \
that reads as true but has dropped the number, the date, or the name is worse than \
no record, because it will be recalled later and answer the question wrongly.\
"""

_PROMPT = """\
Below are {n} separate chat sessions with the same user, each with its date and a \
0-based index.

Record every distinct thing the user states about themselves. One record per fact — \
if a session contains six facts, produce six records, not one summary of them.

For each record:

- `session_index`: which session it came from (0 to {max_index})
- `type`:
    - `profile`     stable identity: name, age, location, job, family
    - `preference`  likes, dislikes, habitual choices
    - `semantic`    general durable facts about the user's world
    - `episodic`    a specific dated event that happened
    - `procedural`  how the user does something, or wants things done
- `content`: a self-contained sentence, read later with no surrounding context, so \
resolve every pronoun. "The user's sister Mei lives in Osaka", never "She lives there".
- `subject` / `predicate` / `object`: the fact as a triple. `subject` is usually \
"user". Prefer these predicates when one fits: {predicates}. Invent a snake_case \
predicate only if none does.
- `entities`: named people, places, products, organizations.
- `importance` 0.0-1.0: how likely this is to be needed later.
- `replaces_previous`: true only when the user signals this supersedes something \
earlier — "I switched to X", "I no longer do Y", "I moved from A to B".

## What must survive, verbatim

These are the things later questions are actually about. Copy them into `content` \
exactly as the user gave them — do not round, generalise, or paraphrase them away.

1. **Quantities.** "25 postcards", "16GB", "four courses". Never "several", "some".
2. **Durations.** "three months", "45 minutes each way", "two years". A duration \
dropped is unrecoverable — nothing else in the record implies it.
3. **Dates and times.** "March 15", "9:15 AM", "in 2019".
4. **Relative time.** "last week", "two months ago", "next Friday", "yesterday". \
Keep the user's own wording *and* resolve it against the session date where you \
can: "The user adopted the cat last month (around February 2023)".
5. **Proper nouns.** Titles, brands, products, people: "The Glass Menagerie", \
"Mod Podge", "Sonos One". Never "a play", "a sealant", "a speaker".
6. **Negations.** "no longer", "stopped", "does not", "gave up". Record the \
negation as the fact — "The user no longer drinks coffee" — never drop it and \
never soften it to a preference.
7. **State changes.** When the user reports a change, record the new state, the old \
one if stated, and set `replaces_previous`: "The user moved from Canberra to Sydney \
in August 2023".
8. **Qualifiers.** "about", "roughly", "at least", "usually", "only on weekdays". \
They change what the fact means; keep them.

## Also record what the assistant told this user

Users ask "what sealant did you recommend?" and the answer exists only in the \
assistant's turn. Record those with `subject` set to "assistant": a named product, \
a specific figure, a particular item from a list it gave. Not its generic advice, \
not its filler, not statements about itself.

## Do not record

- generic knowledge that is not about this user
- anything the user only *asked about* — a question about Osama bin Laden's height \
states nothing about the user
- pure conversational filler ("thanks", "sounds good", "hang on a second")
- speculation, or things the user considered but did not do

## Before you finish

Re-read each session and check: does every number, duration, date, relative-time \
expression, and proper noun the user stated appear in one of your records? If one \
does not, either add a record for it or you have decided it was not about the user. \
There is no third case.

If a session contains nothing about the user, extract nothing from it.

{sessions}
"""


def render_batch(sessions: list[HaystackSession]) -> str:
    blocks = []
    for i, sess in enumerate(sessions):
        turns = "\n".join(f"{t.role}: {t.content}" for t in sess.turns)
        blocks.append(f"=== Session index {i} | date: {sess.date} ===\n{turns}")
    return "\n\n".join(blocks)


def _parse_date(raw: str) -> datetime | None:
    for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw.strip(), fmt)
        except ValueError:
            continue
    return None


def memory_id(user_id: str, session_id: str, content: str) -> str:
    """Deterministic, so re-ingesting a session replaces its memories instead of
    duplicating them. Makes the whole pipeline idempotent per session."""
    digest = hashlib.sha1(f"{user_id}|{session_id}|{content}".encode()).hexdigest()
    return f"mem_{digest[:16]}"


@dataclass(slots=True)
class ExtractionOutcome:
    memories: list[Memory]
    dropped_bad_index: int
    """Records the model attributed to a session outside the batch. Tracked rather
    than silently discarded — a nonzero rate means the batch is too large for the
    model to keep straight, which is the signal to lower sessions_per_request."""


class Extractor:
    def __init__(
        self,
        client: GeminiClient,
        model: str,
        user_id: str = "user",
        chars_per_token: float = 4.6,
    ) -> None:
        self.client = client
        self.model = model
        self.user_id = user_id
        self.chars_per_token = chars_per_token

    def extract(self, sessions: list[HaystackSession]) -> ExtractionOutcome:
        if not sessions:
            return ExtractionOutcome([], 0)

        rendered = render_batch(sessions)
        prompt = _PROMPT.format(
            n=len(sessions),
            max_index=len(sessions) - 1,
            predicates=", ".join(COMMON_PREDICATES),
            sessions=rendered,
        )
        completion = self.client.generate(
            role="extractor",
            model=self.model,
            prompt=prompt,
            system=EXTRACT_SYSTEM,
            schema=ExtractionResult,
            temperature=0.0,
            est_input_tokens=int(len(prompt) / self.chars_per_token),
        )
        result = ExtractionResult.model_validate_json(completion.text)
        return self._to_memories(result.memories, sessions)

    def _to_memories(
        self, extracted: list[ExtractedMemory], sessions: list[HaystackSession]
    ) -> ExtractionOutcome:
        memories: list[Memory] = []
        bad_index = 0
        now = datetime.now()

        for item in extracted:
            if not 0 <= item.session_index < len(sessions):
                bad_index += 1
                continue
            session = sessions[item.session_index]
            event_time = _parse_date(session.date)
            content = item.content.strip()
            if not content:
                continue

            memories.append(
                Memory(
                    id=memory_id(self.user_id, session.session_id, content),
                    user_id=self.user_id,
                    type=item.type,
                    content=content,
                    token_count=max(1, int(len(content) / self.chars_per_token)),
                    subject=item.subject,
                    predicate=item.predicate,
                    object=item.object.strip(),
                    importance=item.importance,
                    replaces_previous=item.replaces_previous,
                    event_time=event_time,
                    # valid_from starts at the event; valid_to stays open until P4
                    # finds something that supersedes it.
                    valid_from=event_time,
                    valid_to=None,
                    ingested_at=now,
                    entities=[e.strip() for e in item.entities if e.strip()],
                    source_session_id=session.session_id,
                )
            )
        return ExtractionOutcome(memories, bad_index)
