---
name: temporal-memory-reviewer
description: Review temporal semantics in long-term-memory extraction, lifecycle, storage, retrieval, or answers. Use for event/session time, temporal keys, updates, termination, validity intervals, current/as-of state, relative dates, or out-of-order ingestion; do not use for non-temporal retrieval tuning.
---

# Temporal memory reviewer

Check these independently:

1. Conversation/session time versus fact/event time.
2. Relative-date parsing and preservation of the raw expression/provenance.
3. Explicit dates in the fact taking priority over session timestamps.
4. ADD, REPLACE, TERMINATE, and COEXIST classification.
5. Out-of-order ingestion convergence.
6. Timeline rebuild idempotence.
7. Same-value restatements versus real updates.
8. `as_of(t)` and current-versus-historical queries.
9. Ambiguous replacements.
10. Removal represented as termination, not a fake successor value.

For example, with a 2026-09-18 session, “I moved to Sydney two months ago” should retain
the raw relative expression and resolve approximately to 2026-07-18; it must not silently
use 2026-09-18 as the event time.

Use existing temporal, lifecycle, timeline, stage-B, and idempotence tests. Suggest schema
changes only when the current representation cannot preserve the required distinction.
