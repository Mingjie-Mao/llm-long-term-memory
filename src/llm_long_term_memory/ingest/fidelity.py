"""Extraction fidelity: what fraction of the user's own specifics survive the write.

The v1 failure (D26) was that extraction discarded the details questions ask about.
The answer-coverage gate (D18) detected the symptom but only where a gold answer
appears verbatim in a memory, which is a minority of LongMemEval and is blind to
the categories that failed hardest.

This measures the same defect without gold answers at all. Take what the *user*
stated in a session — the numbers, dates, durations, money, proper nouns — and ask
how much of it is still present after extraction. Two properties follow:

  * **It cannot be overfitted to the dev questions.** There is no answer key to fit
    to; the reference is the source text itself, so it can be run on any sessions,
    including ones no evaluation will ever touch.
  * **It is per-category.** "Fidelity 0.42" is not actionable, but "durations 0.11,
    quantities 0.55" says which clause of the prompt to write.

It is a recall measure and deliberately one-sided. Extracting *more* than the user
said is a different defect (hallucination) that this does not see, so a rising
score is necessary and not sufficient. The end-to-end evaluation remains the
arbiter; this exists so that a prompt change can be judged for ~20 requests instead
of a full ingest plus two evaluation runs.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from llm_long_term_memory.evaluation.datasets.longmemeval import HaystackSession
from llm_long_term_memory.store import Memory

# Deliberately narrow patterns. A loose one inflates the denominator with tokens no
# reasonable extractor would keep, and the score stops meaning anything.
PATTERNS: dict[str, re.Pattern[str]] = {
    # 25, 3.5, 1,200, and unit-suffixed forms like 16GB or 5km. No trailing \b:
    # there is no word boundary between "16" and "GB", so requiring one silently
    # skipped every number written against its unit — exactly the specifics most
    # worth keeping.
    "quantity": re.compile(r"\b\d[\d,]*(?:\.\d+)?"),
    # three months, 45 minutes, two weeks — the category v1 lost most completely.
    "duration": re.compile(
        r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
        r"a|an|half)\s+"
        r"(?:second|minute|hour|day|week|month|year|decade)s?\b",
        re.I,
    ),
    # $45, 45 dollars, £20
    "money": re.compile(
        r"(?:[$£€]\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*\s?(?:dollars?|pounds?|euros?)\b)", re.I
    ),
    # March 15, 2023-03-15, 15 March
    "date": re.compile(
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b"
        r"|\b\d{4}-\d{2}-\d{2}\b"
        r"|\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b",
        re.I,
    ),
    # last month, two weeks ago, next Friday — anchored to the session date, and
    # unrecoverable once dropped, because nothing else in the memory dates the event.
    "relative_time": re.compile(
        r"\b(?:last|next|this|past)\s+(?:week|month|year|monday|tuesday|wednesday|"
        r"thursday|friday|saturday|sunday|night|weekend)\b"
        r"|\b(?:yesterday|today|tomorrow)\b"
        r"|\b\w+\s+ago\b",
        re.I,
    ),
    # Proper nouns: The Glass Menagerie, Mod Podge, Memrise. Multi-word runs only —
    # single capitalised words are mostly sentence starts.
    "proper_noun": re.compile(r"\b(?:[A-Z][a-z']+\s+){1,4}[A-Z][a-z']+\b"),
}

# Stop the proper-noun pattern counting conversational scaffolding as content.
_SENTENCE_START = re.compile(r"^(?:as|and|but|so|if|when|while|after|before)\s+i\b", re.I)

_PROPER_NOUN_STOP = {
    "i am",
    "i have",
    "i was",
    "i will",
    "i would",
    "i think",
    "i really",
    "thanks so",
    "thank you",
    "let me",
    "you know",
    "i just",
    "i also",
}


# A number is a fact only when it counts or measures something. `\d+.` at the head
# of a line is a list marker, and a user who pastes a numbered document is not
# stating forty quantities about themselves. Left in, these dominated the
# denominator — an inspection of the misses found the "dropped quantities" were
# `1.` `2.` `3.` from a pasted document — and made quantity recall read ~20% when
# the extractor was correctly ignoring almost all of it.
_LIST_MARKER = re.compile(r"(?:^|\n)\s*\d+[.)]\s")
_ATTACHED = re.compile(r"\d[\d,]*(?:\.\d+)?\s*[-\w$£€%]*\s*[A-Za-z]{2,}")


def _meaningful_quantities(text: str) -> set[str]:
    """Numbers attached to a unit or a noun, with list markers removed."""
    markers = {m.group(0).strip().rstrip(".)") for m in _LIST_MARKER.finditer(text)}
    out = set()
    for m in PATTERNS["quantity"].finditer(text):
        value = m.group(0).strip()
        if value in markers:
            continue
        tail = text[m.start() : m.start() + 40]
        if not _ATTACHED.match(tail):
            continue
        out.add(value.lower())
    return out


def _extract_facets(text: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for name, pattern in PATTERNS.items():
        if name == "quantity":
            out[name] = _meaningful_quantities(text)
            continue
        found = {m.group(0).strip().lower() for m in pattern.finditer(text)}
        if name == "proper_noun":
            # "As I'm", "When I" — a capitalised sentence opener followed by the
            # pronoun is grammar, not a name. Roughly a tenth of the raw matches.
            found = {
                f for f in found if f not in _PROPER_NOUN_STOP and not _SENTENCE_START.match(f)
            }
        out[name] = {f for f in found if f and f not in _FILLER}
    return out


# Fillers that match a facet pattern but assert nothing durable. Counting them
# would penalise an extractor for correctly ignoring "hang on a second".
_FILLER = {"a second", "a minute", "one second", "one minute", "a moment", "today"}

_SENTENCE = re.compile(r"[^.!?]+[.!?]?")


def user_assertions(session: HaystackSession) -> str:
    """The user's own statements, excluding what they merely asked about.

    The denominator has to be things the extractor *should* have kept, or tuning
    against it teaches the extractor to hoard. Two exclusions do most of the work:

      * **Assistant turns.** Their numbers are invented — recipe amounts, list
        positions — and were never facts about the user.
      * **Interrogative sentences.** A user asking "how tall was Osama bin Laden?"
        has stated nothing about themselves; the proper nouns and figures in that
        question are correctly dropped. Scoring them as losses made the v1 baseline
        look worse than it was and would push the prompt toward storing trivia.

    What remains is still imperfect — hypotheticals and quoted speech survive it —
    but it is the difference between a metric that rewards fidelity and one that
    rewards volume.
    """
    kept = []
    for turn in session.turns:
        if turn.role != "user":
            continue
        for sentence in _SENTENCE.findall(turn.content):
            if sentence.strip().endswith("?"):
                continue
            kept.append(sentence)
    return "\n".join(kept)


def user_text(session: HaystackSession) -> str:
    """Every user turn, questions included. Kept for callers that want the raw
    denominator; the gate uses `user_assertions`."""
    return "\n".join(t.content for t in session.turns if t.role == "user")


@dataclass(slots=True)
class FacetScore:
    facet: str
    stated: int
    retained: int

    @property
    def recall(self) -> float:
        return self.retained / self.stated if self.stated else 0.0


@dataclass
class FidelityReport:
    per_facet: dict[str, FacetScore] = field(default_factory=dict)
    sessions: int = 0
    memories: int = 0
    missed_examples: dict[str, list[str]] = field(default_factory=dict)

    @property
    def overall(self) -> float:
        stated = sum(s.stated for s in self.per_facet.values())
        retained = sum(s.retained for s in self.per_facet.values())
        return retained / stated if stated else 0.0

    @property
    def memories_per_session(self) -> float:
        return self.memories / self.sessions if self.sessions else 0.0


def score_sessions(
    pairs: list[tuple[HaystackSession, list[Memory]]], keep_examples: int = 6
) -> FidelityReport:
    """`pairs` is (session, memories extracted from it)."""
    report = FidelityReport()
    stated: Counter[str] = Counter()
    retained: Counter[str] = Counter()
    misses: dict[str, list[str]] = {k: [] for k in PATTERNS}

    for session, memories in pairs:
        report.sessions += 1
        report.memories += len(memories)

        source = _extract_facets(user_assertions(session))
        # One haystack per session: a specific counts as retained if it survived
        # into any memory drawn from that session, not necessarily the same one.
        memory_blob = " || ".join(f"{m.content} {m.object or ''}" for m in memories).lower()

        for facet, values in source.items():
            for value in values:
                stated[facet] += 1
                if value in memory_blob:
                    retained[facet] += 1
                elif len(misses[facet]) < keep_examples:
                    misses[facet].append(value)

    report.per_facet = {
        facet: FacetScore(facet, stated[facet], retained[facet]) for facet in PATTERNS
    }
    report.missed_examples = {k: v for k, v in misses.items() if v}
    return report
