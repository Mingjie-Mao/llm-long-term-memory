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
You extract facts about a user from their chat history, for a long-term memory \
system. Be accurate: never record something the user did not say. But be thorough \
about specifics — the details the user states about themselves are exactly what \
they will ask about later.\
"""

_PROMPT = """\
Below are {n} separate chat sessions with the same user, each with its date and a \
0-based index.

Extract the facts worth remembering long-term. For each fact:

- `session_index`: which session it came from (0 to {max_index})
- `type`:
    - `profile`     stable identity: name, age, location, job, family
    - `preference`  likes, dislikes, habitual choices
    - `semantic`    general durable facts about the user's world
    - `episodic`    a specific dated event that happened
    - `procedural`  how the user does something, or wants things done
- `content`: a self-contained sentence. It will be read with no surrounding \
context, so resolve every pronoun and reference. Write "The user's sister Mei lives \
in Osaka", never "She lives there".
- `subject` / `predicate` / `object`: the fact as a triple. `subject` is usually \
"user". Prefer these predicates when one fits, so that repeated mentions of the \
same attribute line up: {predicates}. Invent a snake_case predicate only if none \
fits.
- `entities`: named people, places, products, organizations.
- `importance` 0.0-1.0: how likely this is to be needed in a later conversation. \
A job change is 0.9; the user mentioning they had toast is 0.1.
- `replaces_previous`: true only when the user signalled that this *replaces* \
something they said before — "I switched to X", "I no longer do Y", "I moved from A \
to B", "I've stopped Z". A plain new statement is false. This is what tells the \
system an old fact stopped being true, so do not set it speculatively.

**Keep the specifics.** Quantities, durations, prices, dates, counts, and proper \
names that the user states about themselves are the single most important thing to \
record — they are what gets asked about later. Write "The user watched 10 hours of \
documentaries on Netflix last month", not "The user watches documentaries". Write \
"The user attended The Glass Menagerie at the local community theater", not "The \
user went to a play". A memory that drops the number or the name has failed, even \
though it reads as true.

Prefer several precise memories over one general one. If the user mentions three \
separate purchases, that is three memories with three amounts, not one memory about \
shopping.

**Also record what the assistant told this specific user, when it is concrete \
enough to be referred back to.** Users ask "what was that sealant you \
recommended?" or "what was item 27 on that list?", and the answer only exists in \
the assistant's turn. Record these with `subject` set to "assistant": a named \
product, a specific number, a particular item from a list it gave. Do not record \
the assistant's generic advice, its filler, or anything about the assistant itself.

Do NOT extract:
- generic knowledge that is not about this user
- pure conversational filler ("thanks", "sounds good")
- speculation, or things the user asked about but did not assert

If a session contains nothing worth remembering, extract nothing from it.

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
