"""How memories become answer context, shared by the product and the evaluation.

One copy on purpose. The product engine used to carry its own renderer and templates,
and the two drifted far enough that "the evaluated arm" and "what the API answers
with" were no longer the same thing (tests/test_product_t4_equivalence.py).
"""

from __future__ import annotations

from datetime import datetime

from llm_long_term_memory.store import Memory

ANSWER_TEMPLATE = """\
Here is what is known about the user, drawn from their chat history.

{context}

Today's date is {date}.

Question: {question}
"""

TEMPORAL_NOTE = (
    "Each fact is shown with the period it was true for. `(since <date>)` means it "
    "is still true now; `(<date> to <date>)` means it was replaced and is no longer "
    "current. Answer with what is true now unless the question asks about the past."
)


FALLBACK_TEMPLATE = """\
Your earlier attempt could not answer this from structured memory alone:
{reason}

Here are the original conversation turns those memories were extracted from. They
are verbatim, so the exact wording, links and figures are present.

{evidence}

Today's date is {date}.

Question: {question}
"""


# Scope -> heading, in the order they are shown. Ordering is semantic rather than
# alphabetical: what the user wants and is planning comes before what happened and
# what was previously suggested.
#
# Grouping exists because an undifferentiated bullet list makes every memory look
# like a candidate answer. A model told "User preferences: dislikes raw fish" knows
# to constrain a restaurant recommendation; the same line in a flat list reads as a
# fact it should either quote or ignore.
_SCOPE_HEADINGS: tuple[tuple[str, str], ...] = (
    ("profile", "About the user"),
    ("preference", "User preferences"),
    ("plan", "Current plans"),
    ("event", "Past events"),
    ("recommendation", "Previously recommended by the assistant"),
    ("commitment", "The assistant agreed to"),
    ("shared_context", "Other people and things discussed"),
)
# Pre-P10 memories carry no scope. They are not dropped or guessed at — an
# unlabelled group is honest about what is known.
_UNSCOPED_HEADING = "Other things known about the user"


def label_memories(memories: list[Memory]) -> dict[str, str]:
    """Stable citation handles in rank order: M1, M2, ... one per memory in context.

    Positional rather than derived from `memory.id`: the model has to copy these, and a
    sixteen-character hex id is a transcription error waiting to happen, while the
    position is also the reading order the prompt asks it to work through.
    """
    return {memory.id: f"M{i}" for i, memory in enumerate(memories, start=1)}


def render_grouped(
    memories: list[Memory],
    temporal: bool,
    timelines_enabled: bool = True,
    labels: dict[str, str] | None = None,
    mark_unstated: bool = False,
) -> str:
    """Group memories by scope under headings, preserving rank order within a group.

    Falls back to a plain list when nothing carries a scope, so a store built before
    P10 renders exactly as it did before.
    """
    # Chains first, so a superseded value is never also listed flat under a scope
    # heading — seeing it twice, once marked "no longer current" and once not, is
    # worse than either rendering alone.
    timelines, memories = render_timelines(memories, temporal and timelines_enabled, labels)
    tag = (lambda m: labels.get(m.id, "")) if labels else (lambda m: "")
    prefix = [timelines] if timelines else []

    if not any(memory.scope for memory in memories):
        flat = "\n".join(
            render_memory(memory, temporal, tag(memory), mark_unstated) for memory in memories
        )
        return "\n\n".join([*prefix, flat]) if flat else "\n\n".join(prefix)

    by_scope: dict[str, list[Memory]] = {}
    for memory in memories:
        by_scope.setdefault(memory.scope or "", []).append(memory)

    blocks: list[str] = []
    for scope, heading in _SCOPE_HEADINGS:
        group = by_scope.pop(scope, None)
        if group:
            lines = "\n".join(render_memory(m, temporal, tag(m), mark_unstated) for m in group)
            blocks.append(f"{heading}:\n{lines}")
    # Anything left: unscoped memories, plus any scope the model invented that the
    # validator let through.
    leftovers = [m for group in by_scope.values() for m in group]
    if leftovers:
        lines = "\n".join(render_memory(m, temporal, tag(m), mark_unstated) for m in leftovers)
        blocks.append(f"{_UNSCOPED_HEADING}:\n{lines}")
    return "\n\n".join([*prefix, *blocks])


def render_timelines(
    memories: list[Memory], temporal: bool, labels: dict[str, str] | None = None
) -> tuple[str, list[Memory]]:
    """Render supersession chains as chains, and return what was not part of one.

    The store has already decided which value of a key is current: it closed the old
    one's validity window and set its status. Flat rendering throws that away and asks
    the model to rediscover it from two adjacent bullet points that differ only in a
    parenthesised date — and on `train150` it does not: a question about where a guitar
    was serviced was answered "you *plan* to send it", with both the plan and the
    completion in context.

    So a key with more than one value is rendered as one block, in event order, with
    the current value marked. No new memory is fetched and no new field is computed;
    this is the same data the flat renderer already had.

    Only for `temporal=True`. Without dates there is no order to show, and grouping
    undated values would assert a sequence the store never established.
    """
    if not temporal:
        return "", list(memories)

    chains: dict[tuple[str, str], list[Memory]] = {}
    for memory in memories:
        if memory.subject and memory.predicate:
            chains.setdefault((memory.subject, memory.predicate), []).append(memory)

    blocks: list[str] = []
    claimed: set[str] = set()
    for (subject, predicate), group in chains.items():
        # One value is not a timeline. Rendering it as one would add a heading and a
        # "current" label to a fact nothing ever contradicted.
        if len(group) < 2:
            continue
        ordered = sorted(
            group,
            key=lambda m: (m.valid_from or m.occurred_at or datetime.max, m.id),
        )
        lines = []
        for memory in ordered:
            when = memory.valid_from or memory.occurred_at
            stamp = f"{when:%Y-%m-%d}" if when else "date unknown"
            mark = "[CURRENT]" if memory.status == "active" and not memory.valid_to else "[was]"
            # Chain members are labelled too. A count question can have a member that is
            # also part of a supersession chain, and an unlabelled one would be a member
            # the answerer has no way to cite.
            tag = f"[{labels[memory.id]}] " if labels and memory.id in labels else ""
            lines.append(f"    {stamp}  {tag}{memory.content} {mark}")
            claimed.add(memory.id)
        label = f"{subject} {predicate}".replace("_", " ").strip()
        blocks.append(f"{label} — how this changed over time:\n" + "\n".join(lines))

    remaining = [memory for memory in memories if memory.id not in claimed]
    return "\n\n".join(blocks), remaining


def render_memory(
    memory: Memory, temporal: bool, label: str = "", mark_unstated: bool = False
) -> str:
    """One memory as a context line.

    `mark_unstated` is the v2e difference and nothing else changes with it. `valid_from`
    is the date a fact was *stated*, so an undated fact has always rendered as
    "(since <the day it was mentioned>)" — which reads as the day it became true. On
    every one of the thirty temporal questions in reasoning-48 some selected evidence
    is undated, so a reader asked "how many days between X and Y" is routinely given a
    mention date wearing an event date's clothes. Saying "mentioned" instead is the
    whole mechanism.

    Off by default, because turning it on changes the context of every recorded arm.
    """
    # The label is a citation handle, not content. It goes first so that a model working
    # through the context in order reads it before the sentence it belongs to.
    tag = f"[{label}] " if label else ""
    if not temporal:
        return f"- {tag}{memory.content}"

    start = memory.valid_from or memory.occurred_at
    if start and memory.valid_to:
        window = f"({start:%Y-%m-%d} to {memory.valid_to:%Y-%m-%d}, no longer current)"
    elif start and mark_unstated and not memory.event_time_is_stated:
        window = f"(mentioned {start:%Y-%m-%d}; the fact states no date)"
    elif start:
        window = f"(since {start:%Y-%m-%d})"
    else:
        window = "(date unknown)"
    return f"- {tag}{memory.content} {window}"


def render_context(memories: list[Memory], temporal: bool) -> str:
    """The memory block of the answer prompt: grouped, with the temporal note if dated."""
    body = render_grouped(memories, temporal)
    return f"{TEMPORAL_NOTE}\n\n{body}" if temporal else body
