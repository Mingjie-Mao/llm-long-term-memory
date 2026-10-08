"""Opt-in verbatim personal assertions, kept as source navigation, not state facts.

This supplements lossy extraction without another provider call. Quotations retain
their speaker and source date; they never create a replacement chain or resolve an
unstated event time. The raw archive remains authoritative.
"""

from __future__ import annotations

import hashlib
import re

from llm_long_term_memory.store import Memory, external_session_id

from .repair import RepairOutcome

VERSION = "personal-context-quotes-v2"
SENTENCES = re.compile(r"[\s\S]+?(?:[.!?](?!\d)(?=\s|$)|$)")
PERSONAL = re.compile(
    r"\b(?:I|we)\s+(?:am|was|were|have|had|enjoy|like|love|prefer|want|need|"
    r"attended|volunteered|bought|made|learned|started|finished|use|listen|live|work|"
    r"commute|moved|remember|already|recently|usually|currently|feel|do|did|don't|didn't|"
    r"walked|got|participated|donated|spent|paid|own|replaced|fixed|noticed)\b|"
    r"\bI['\u2019](?:m|ve|d)\b|\b(?:my|our)\b[^.!?]{0,90}\b(?:is|are|has|have|takes|lasts)\b",
    re.I,
)
WORDS = re.compile(r"[a-z0-9]+", re.I)
STOP = {"the", "user", "i", "my", "a", "an", "and", "to", "of", "is", "has", "about"}


def terms_for_anchor(text):
    return {
        word[:-1] if word in {"minutes", "hours", "days", "weeks"} else word
        for word in WORDS.findall(text.lower())
    } - STOP


def quote_identity(user_id, session_id, index, start, text):
    key = f"{user_id}|{external_session_id(session_id)}|{index}|{start}|{text}"
    return "pq_" + hashlib.sha256(key.encode()).hexdigest()[:24]


def personal_spans(session):
    for index, turn in enumerate(session.turns):
        if turn.role != "user":
            continue
        for match in SENTENCES.finditer(turn.content):
            text = match.group().strip()
            # Questions without an assertion and hypothetical requests are not facts.
            if not PERSONAL.search(text) or re.search(
                r"\b(?:if I|suppose I|imagine I)\b", text, re.I
            ):
                continue
            start = match.start() + len(match.group()) - len(match.group().lstrip())
            yield index, start, start + len(text), text


def reanchor_numeric_echoes(session, memories, user_id):
    """Repair only a uniquely supported user numerical assertion behind an echo.

    Type and scope stay unchanged. Ambiguous sources are left alone. This is narrower
    than the globally rejected speaker-filter experiment.
    """
    count = 0
    for memory in memories:
        if memory.user_id != user_id:
            raise ValueError("foreign memory in personal context repair")
        if external_session_id(memory.source_session_id or "") != external_session_id(
            session.session_id
        ):
            continue
        old = memory.source_turn_index
        if type(old) is not int or not 0 <= old < len(session.turns):
            continue
        if session.turns[old].role != "assistant" or memory.source_role != "user":
            continue
        terms = terms_for_anchor(memory.content)
        numbers = {word for word in terms if word.isdigit()}
        if not numbers:
            continue
        matches = []
        for index, start, end, text in personal_spans(session):
            raw = terms_for_anchor(text)
            qualifiers = r"\b(?:not|never|used to|previously|last year|last month)\b|n['\u2019]t\b"
            if set(re.findall(qualifiers, text.lower())) != set(
                re.findall(qualifiers, memory.content.lower())
            ):
                continue
            # Every non-numeric content term must be in the actual user quote.
            if numbers <= raw and len((terms - numbers) & raw) >= 2 and terms - numbers <= raw:
                matches.append((index, start, end))
        if len(matches) == 1:
            memory.source_turn_index, memory.source_char_start, memory.source_char_end = matches[0]
            # A date from the assistant echo cannot remain event provenance.
            memory.event_time_source_expression = None
            count += 1
    return count


class PersonalContextRepair:
    version = VERSION

    @staticmethod
    def prompt_texts():
        # No LLM prompt, but regex and policy must still be fingerprinted.
        return (VERSION, PERSONAL.pattern, SENTENCES.pattern, "unique-numeric-user-anchor")

    def repair(self, session, memories, user_id):
        reanchor_numeric_echoes(session, memories, user_id)
        quotes = []
        for index, start, end, text in personal_spans(session):
            if any(
                m.predicate == "source_quote" and m.source_turn_index == index and m.content == text
                for m in memories
            ):
                continue
            quotes.append(
                Memory(
                    id=quote_identity(user_id, session.session_id, index, start, text),
                    user_id=user_id,
                    type="episodic",
                    scope="shared_context",
                    predicate="source_quote",
                    content=text,
                    token_count=max(1, len(text.split())),
                    source_role="user",
                    source_session_id=session.session_id,
                    source_turn_index=index,
                    source_char_start=start,
                    source_char_end=end,
                )
            )
        return RepairOutcome(quotes, called=False, missing=[], attempted=len(quotes))


class CombinedRepair:
    version = "combined-repair-v1"

    def __init__(self, *repairs):
        self.repairs = repairs

    def prompt_texts(self):
        return tuple(text for repair in self.repairs for text in repair.prompt_texts())

    def repair(self, session, memories, user_id):
        added, called, missing, attempted, rejected = [], False, [], 0, 0
        for repair in self.repairs:
            result = repair.repair(session, [*memories, *added], user_id)
            added.extend(result.memories)
            called |= result.called
            missing.extend(result.missing)
            attempted += result.attempted
            rejected += result.rejected
        return RepairOutcome(added, called, missing, attempted, rejected)


def configured_repair(cfg, client=None):
    """Same composition for runtime and config fingerprinting; client may be absent."""
    from .grounded import GroundedExtractor
    from .repair import SpecificityRepair

    repairs = []
    if cfg.ingest.personal_context_repair:
        repairs.append(PersonalContextRepair())
    if cfg.ingest.specificity_repair:
        repairs.append(SpecificityRepair(GroundedExtractor(client, cfg.models.extractor)))
    return None if not repairs else repairs[0] if len(repairs) == 1 else CombinedRepair(*repairs)
