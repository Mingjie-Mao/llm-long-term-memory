# Two failure attributions that seemed to disagree — reconciled

Date: 2026-09-28. Zero calls. No new judgement of any answer: both sets' per-question
assignments are the hand audits already on record, placed on one taxonomy.

## The apparent conflict

- `dev50` (`results/failure-stages.md`): of 14 failures, **10 are first lost at
  extraction**, 0 at retrieval.
- `reasoning-48` (`tools/v5_offline_gate.py`, `AUDIT`): of 11 failures, **6 had the
  evidence in the selected context** and were still answered wrong.

Read together they seem to say opposite things about where to work. They do not.

## One taxonomy for both

`dev50` used first-loss stages S1–S5. The reasoning-48 audit used reach categories;
they map onto the same stages without re-judging anything:

| reasoning-48 audit | first-loss stage | why |
|---|---|---|
| `local_raw` | S1 extraction | a sentence the extractor dropped |
| `repack` | S4 retrieval/selection | in the store, not selected |
| `retrieval` | S4 | gold session never found |
| `reader`, selected memory unused (`dd2973ad`) | S4b composition | supplied, not used |
| `reader`, the other five | S5 reasoning | supplied and correct, answered wrong |

| stage | `dev50` (v2-era, batch-15 store) | `reasoning-48` (v2c, batch-8 store) |
|---|---:|---:|
| S1 extraction | **10** | 3 |
| S2 lifecycle | 1 | 0 |
| S4 retrieval / selection | 0 | 2 |
| S4b composition | 1 | 1 |
| S5 reasoning | 2 | **5** |
| failures | 14 | 11 |

## Why they differ

**Not the question mix.** Both failure sets are dominated by the same types
(`dev50`: 7 temporal-reasoning, 6 multi-session, 1 preference; `reasoning-48`: 5
multi-session, 3 temporal-reasoning, 1 each knowledge-update, user, assistant).

**The system changed between them.** `dev50` was measured on the frozen-v2 lineage
store, extracted at 15 sessions per request. `reasoning-48` ran v2c, which reads the
v2b store extracted at **8** sessions per request (`configs/v2c.yaml`), and adds
query-time repairs that hydrate source text when a question asks for finer detail
than the structured fact kept. Batch 8 raises specific recall on personal sessions to
62–66% (`results/analysis/batch8-session-kind-attribution-v1.md`), and source-local
hydration returns dropped sentences from sessions already selected. Those are exactly
the S1 losses that dominate `dev50`. What remains after them is reasoning: two
false-premise questions answered instead of declined, two counts, one arithmetic
misreading.

**And `dev50`'s own oracle already said extraction was not the whole story.**
Injecting the missing fact fixed 4 of the 9 S1 questions it was tried on; the other
five had a second problem behind the missing fact.

## What this means

Both attributions are correct for the system they measured:

- **For the frozen v2 system — the one behind 72% — extraction is the first loss.**
  The zero-call train150 index gate says the same about a batch-15 store: 38 of 39
  index misses have no memory from the gold turn
  (`results/memory-as-index-offline-decision.md`).
- **For the current batch-8 line, the leading residual is synthesis**: abstention on
  false premises, counting, and arithmetic over facts that are already present.

So the next mechanism for the current line should target synthesis, and any claim
about the frozen system should still name extraction. Neither set can say by how much:
reasoning-48 resolves a net of 4, `dev50` is burned, and both are development evidence.
