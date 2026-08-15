"""The two-stage extractor, behind the single-stage extractor's interface.

Drop-in for `Extractor`: same `extract(sessions) -> ExtractionOutcome`, so the
fidelity gate, the ingestion pipeline, and the tests can swap between them and the
comparison stays single-variable. Which one runs is a config choice, not a rewrite.

    Stage A   sessions -> bare fact strings          (1 request per batch)
    Stage B   facts -> temporal_key + update_op      (1 request per batch)

Two requests per batch rather than one. The rule-based Stage B was free but scored
43% predicate accuracy and produced zero supersessions over 148 real memories; the
LLM version passes the temporal gate on all four metrics. Doubling the ingest from
~190 to ~380 requests is what that costs, and it still fits inside one day.

Rules are kept for the fields the model has no advantage on — type, entities,
importance, dates — so Stage B's single call spends its whole output budget on the
part regexes could not do.
"""

from __future__ import annotations

from datetime import datetime

from llm_long_term_memory.evaluation.datasets.longmemeval import HaystackSession
from llm_long_term_memory.llm.client import GeminiClient
from llm_long_term_memory.store import Memory

from .extract import ExtractionOutcome, _parse_date, memory_id
from .extract_facts import EXTRACTOR_VERSION, FactExtractor, FactLine
from .keying import FactKeyer, Keying, UpdateOp
from .provenance import attach_source_span
from .structure import (
    _REPLACES,
    infer_entities,
    infer_importance,
    infer_object,
    infer_predicate,
    infer_type,
)


def _rule_keying(fact: str) -> Keying:
    """Stage B as regexes — the measured control arm.

    Scored 43% predicate accuracy and produced zero supersessions over 148 real
    memories, which is why the LLM keyer exists. Kept runnable so that number can be
    reproduced rather than cited.
    """
    predicate = infer_predicate(fact)
    return Keying(
        temporal_key=predicate,
        update_op=UpdateOp.REPLACES if _REPLACES.search(fact) else UpdateOp.COEXISTS,
        object=infer_object(fact, predicate),
    )


class TwoStageExtractor:
    version = EXTRACTOR_VERSION

    def __init__(
        self,
        client: GeminiClient,
        model: str,
        user_id: str = "user",
        chars_per_token: float = 4.6,
        llm_keying: bool = True,
    ) -> None:
        self.client = client
        self.model = model
        self.user_id = user_id
        self.chars_per_token = chars_per_token
        self.llm_keying = llm_keying
        self._facts = FactExtractor(client, model, chars_per_token)
        self._keyer = FactKeyer(client, model, chars_per_token)

    def extract(self, sessions: list[HaystackSession]) -> ExtractionOutcome:
        outcome = self._facts.extract(sessions)
        if not outcome.by_session:
            return ExtractionOutcome([], outcome.dropped_bad_index, requests=1)

        # Flatten so the whole batch is keyed in one request, then map back. Keying
        # per session would multiply requests by the batch size and undo the reason
        # for batching at all.
        flat: list[tuple[str, FactLine]] = [
            (session_id, line) for session_id, lines in outcome.by_session.items() for line in lines
        ]
        # Stage A did run, but there is nothing for Stage B to classify. Reporting
        # two requests here would overstate a sparse batch's cost and make a resumed
        # ingestion's accounting disagree with the provider usage log.
        if not flat:
            return ExtractionOutcome([], outcome.dropped_bad_index, requests=1)
        # `llm_keying=False` runs the rule-based keyer rather than returning a stub,
        # so the 43%-accuracy arm stays reachable as a control. A comparison against
        # a placeholder would measure nothing.
        keyings = (
            self._keyer.key([line.content for _, line in flat])
            if self.llm_keying
            else [_rule_keying(line.content) for _, line in flat]
        )

        by_session = {session.session_id: session for session in sessions}
        now = datetime.now()
        memories: list[Memory] = []
        seen: set[str] = set()

        for (session_id, line), keying in zip(flat, keyings, strict=True):
            session = by_session[session_id]
            memory = attach_source_span(
                self._build(line, keying, session_id, session.date, now), session
            )
            # Ids are content-hashed, so a fact repeated within a batch would
            # otherwise produce two rows that collapse to one on write and make the
            # reported count disagree with the store.
            if memory.id in seen:
                continue
            seen.add(memory.id)
            memories.append(memory)

        return ExtractionOutcome(
            memories, outcome.dropped_bad_index, requests=2 if self.llm_keying else 1
        )

    def _build(
        self, line: FactLine, keying: Keying, session_id: str, session_date: str, now: datetime
    ) -> Memory:
        event_time = _parse_date(session_date)
        fact = line.content
        # `subject` and `source_role` come from Stage A, which is the only stage that
        # sees the conversation. This used to be
        #     subject = "assistant" if fact.startswith("the assistant") else "user"
        # which made the speaker and the topic the same field and left every
        # third-party fact filed under `user` — "Andy wore a blue shirt" became a
        # fact about the user, so "what was Andy wearing?" could never be answered
        # (results/assistant-gap.md).

        # `removes` closes an earlier fact just as `replaces` does; what differs is
        # that no successor value follows. The resolver acts on the boolean today,
        # so both set it, and `update_op` preserves which one it was for when the
        # removal case gets its own timeline handling.
        closes_earlier = keying.update_op in (UpdateOp.REPLACES, UpdateOp.REMOVES)

        return Memory(
            id=memory_id(self.user_id, session_id, fact),
            user_id=self.user_id,
            type=infer_type(fact),
            content=fact,
            token_count=max(1, int(len(fact) / self.chars_per_token)),
            subject=line.subject,
            predicate=keying.temporal_key,
            object=keying.object,
            source_role=line.source_role,
            scope=line.scope or None,
            importance=infer_importance(fact),
            event_time=event_time,
            valid_from=event_time,
            valid_to=None,
            ingested_at=now,
            update_op=str(keying.update_op),
            replaces_previous=closes_earlier,
            entities=infer_entities(fact),
            source_session_id=session_id,
        )
