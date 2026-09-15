"""BEAM, read from its exported JSON and cut to the session size v2 was measured at.

`tools/beam_export.py` writes `data/beam/beam-<half>.json` from the pinned parquet files,
once, because pyarrow is not a project dependency. Nothing here reads parquet. The export
is by half, so the final-test conversations are not on disk until the final run needs them.

BEAM differs from LongMemEval in three ways that matter to a memory system, and each is
dealt with here rather than downstream:

* **Twenty questions share one conversation.** Every instance names its conversation as
  its `namespace`, so ingestion builds one store per conversation and retrieval reads it.
* **Sessions are 13-22 times longer.** A LongMemEval session averages 10,250 characters
  (D5), and the frozen extractor's batch size and measured yield belong to sessions that
  size; batch size alone moved yield about five-fold (results/failure-stages.md). So each
  BEAM session is cut into runs of whole exchanges — a user message and the replies to
  it — of about that size.
* **Only a session's first message is dated.** Every chunk of a session takes that date,
  one minute later per chunk, so temporal resolution keeps the order of facts within a day
  without the day itself changing.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from llm_long_term_memory.evaluation.datasets.longmemeval import (
    HaystackSession,
    HaystackTurn,
    Instance,
)

TARGET_CHARS = 10_250
"""Mean characters in a LongMemEval-S session: 244,648,856 over 23,867 sessions (D5)."""

HALVES = ("dev", "test")

# LongMemEval's date format, which is what ingestion parses a session date from.
DATE_FORMAT = "%Y/%m/%d (%a) %H:%M"
ANCHOR_FORMAT = "%B-%d-%Y"

# The field holding each ability's reference text. The rubric, not this, is what is graded;
# the reference is kept on the row so a reader can see what the rubric was written from.
REFERENCE_FIELD = {
    "abstention": "ideal_response",
    "contradiction_resolution": "ideal_answer",
    "event_ordering": "answer",
    "information_extraction": "answer",
    "instruction_following": "expected_compliance",
    "knowledge_update": "answer",
    "multi_session_reasoning": "answer",
    "preference_following": "expected_compliance",
    "summarization": "ideal_summary",
    "temporal_reasoning": "answer",
}
ABILITIES = tuple(sorted(REFERENCE_FIELD))


def half_of(manifest) -> str | None:
    """The BEAM half a manifest names, or None when it belongs to another dataset."""
    name = getattr(manifest, "name", "") or ""
    if not name.startswith("beam-"):
        return None
    half = name.removeprefix("beam-")
    if half not in HALVES:
        raise ValueError(f"{name} names no BEAM half; expected beam-dev or beam-test")
    return half


def exchanges(messages: list[dict]) -> list[list[dict]]:
    """A user message and the replies before the next one; never split below this."""
    groups: list[list[dict]] = []
    for message in messages:
        if message["role"] == "user" or not groups:
            groups.append([message])
        else:
            groups[-1].append(message)
    return groups


def chunk_session(messages: list[dict], target_chars: int = TARGET_CHARS) -> list[list[dict]]:
    """Consecutive whole exchanges, each chunk ending where it is nearest the target.

    An exchange joins the chunk when that leaves the chunk no further from the target than
    stopping would. Stopping before any overshoot instead left chunks about a fifth under the
    target on BEAM's development half, which is about a quarter more extraction requests
    than the operating point needs.
    """
    chunks: list[list[dict]] = []
    current: list[dict] = []
    size = 0
    for exchange in exchanges(messages):
        length = sum(len(message.get("content") or "") for message in exchange)
        if current and abs(size + length - target_chars) > abs(size - target_chars):
            chunks.append(current)
            current, size = [], 0
        current.extend(exchange)
        size += length
    if current:
        chunks.append(current)
    return chunks


def source_message_ids(value) -> set[int]:
    """Message ids in `source_chat_ids`, which BEAM gives as a list, a list of lists, or a
    mapping such as `{"first_event": ..., "second_event": ...}`."""
    if isinstance(value, dict):
        return set().union(*(source_message_ids(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(source_message_ids(item) for item in value))
    if isinstance(value, int) and not isinstance(value, bool):
        return {value}
    return set()


def conversation_instances(conversation: dict) -> list[Instance]:
    """One instance per question, all sharing the conversation's chunked sessions."""
    namespace = f"beam-{conversation['scale']}-{conversation['conversation_id']}"
    sessions: list[HaystackSession] = []
    session_of_message: dict[int, str] = {}
    for session_index, messages in enumerate(conversation["chat"]):
        anchor = messages[0].get("time_anchor") if messages else None
        if not anchor:
            raise ValueError(f"{namespace} session {session_index} has no dated first message")
        day = datetime.strptime(anchor, ANCHOR_FORMAT)
        for chunk_index, chunk in enumerate(chunk_session(messages)):
            session_id = f"s{session_index:02d}c{chunk_index:03d}"
            sessions.append(
                HaystackSession(
                    session_id=session_id,
                    date=(day + timedelta(minutes=chunk_index)).strftime(DATE_FORMAT),
                    turns=[
                        HaystackTurn(role=message["role"], content=message.get("content") or "")
                        for message in chunk
                    ],
                )
            )
            session_of_message.update((message["id"], session_id) for message in chunk)
    question_date = sessions[-1].date if sessions else ""
    probing = conversation["probing_questions"]
    instances = []
    for ability in sorted(probing):
        for index, item in enumerate(probing[ability]):
            evidence = source_message_ids(item.get("source_chat_ids"))
            instances.append(
                Instance(
                    question_id=f"{namespace}-{ability}-{index}",
                    question_type=ability,
                    question=item["question"],
                    answer=str(item.get(REFERENCE_FIELD.get(ability, "answer")) or ""),
                    question_date=question_date,
                    sessions=sessions,
                    answer_session_ids=sorted(
                        {session_of_message[i] for i in evidence if i in session_of_message}
                    ),
                    namespace=namespace,
                    rubric=tuple(str(criterion) for criterion in item.get("rubric") or ()),
                )
            )
    return instances


def load(half: str, data_dir: str | Path = "data") -> list[Instance]:
    if half not in HALVES:
        raise ValueError(f"unknown BEAM half {half!r}; expected one of {HALVES}")
    path = Path(data_dir) / "beam" / f"beam-{half}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; export it with `python3 tools/beam_export.py --half {half}`"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("half") != half:
        raise ValueError(f"{path} holds the {payload.get('half')!r} half, not {half!r}")
    return [
        instance
        for conversation in payload["conversations"]
        for instance in conversation_instances(conversation)
    ]
