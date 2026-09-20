# v2b gate16 decision

## Decision

**STOP: do not expand batch8 to the 48-question arm.**

Batch8 improved extraction fidelity and the hybrid raw-source path produced two
repeatable rescues, but the preregistered memory-only improvement did not survive the
single allowed discordant-row repeat. The extraction change is therefore not promoted
as a whole. The source-local fallback mechanism is retained for the next candidate.

This 16-question set was deliberately enriched for prior failures and regression
controls. Its raw accuracy is a mechanism diagnostic, not a benchmark estimate.

## Evidence

| Endpoint | Result |
|---|---:|
| Ingested sessions | 780 / 780 |
| Written memories | 3,055 |
| Extraction fidelity, batch15 -> batch8 | 39.8% -> 47.5% (+7.7pp) |
| Top-20 all-source coverage | 12/16 -> 16/16 |
| Whole-store literal coverage on measurable rows | 9 -> 8 |
| Top-20 literal coverage on measurable rows | 9 -> 7 |
| First-run memory-only proxy | 8/16 -> 9/16; 4W/3L |
| First-run final | 9/16 -> 10/16; 3W/2L |
| Discordant repeat, memory-only | 3W/3L; net 0 |
| Discordant repeat, final | 4W/2L; net +2 |

The first-run paired McNemar p-value is 1.0 for both endpoints. The post-selected
discordant repeat has no valid inferential p-value.

Successful API calls for the completed gate were 338: 232 extraction, 56 answering,
and 50 judging. The ledgers contain 406 total attempts because 68 transient or invalid
requests were retried. The abandoned pre-reset full-train attempt is excluded.

## What worked

The source-local raw fallback is the only clearly repeatable new QA mechanism:

- `778164c6`: structured memory found the right session but lost the requested detail;
  the adjacent raw window fixed the answer in both final runs.
- `d905b33f`: structured memory lacked the original price needed for a calculation;
  the adjacent raw window recovered it and fixed both final runs.

This supports the hybrid architecture: structured memory should route and compress,
while raw turns remain an evidence-preserving backstop.

## What failed

1. **Exact-detail loss (`58ef2f1c`).** The source says February 14, but batch8 stores
   only “February 2023.” The correct source session is retrieved and raw top-2 covers
   it, yet fallback is not triggered. This is a routing/schema mismatch.
2. **Update conflict (`6a1eabeb`).** Both 27:12 and the newer 25:50 are active and in
   top-2. The newer row is marked `update_op=replaces`, but the older row ranks first
   and is not superseded. This is consolidation plus reading-order failure, not recall.
3. **Temporal composition (`gpt4_45189cb4`).** All three events are present in top-20,
   but semantic rank scatters them at positions 3, 6, and 18. The reader omits the first
   event instead of building a chronological evidence set. This is packing/reading
   failure, not extraction or source recall.
4. **Stochastic fallback.** Some rows change answer without any raw fallback, and one
   archive-wide fallback regressed a correct memory-only answer. The trigger is not yet
   deterministic enough to isolate mechanism gains from answerer variance.

## v2c: cheapest next candidate

No re-ingestion is needed. All changes below operate on the completed store and should
pass an offline evidence gate before any answer call.

1. Add **update-aware conflict resolution**: for the same normalized subject/predicate,
   prefer the newest `replaces` fact and either suppress or explicitly label older facts.
2. Add a **detail-completeness trigger**: date/day, amount, count, name, and duration
   questions must fall back to the located source window when the structured value has
   lower precision than the question requests.
3. Add a **temporal evidence assembler**: retrieve broadly, group relevant events,
   sort by event time, and pack a compact chronological note before answering.
4. Restrict raw fallback to **source-local first**. Archive-wide fallback must pass a
   coverage-gain check because it caused noise in this gate.
5. Make routing decisions deterministic and log reason codes so repeated answer calls
   test the reader rather than a changing retrieval path.

The next gate should reuse the existing store and focus on 6-8 hard rows plus unchanged
controls. Promote only if two runs show at least two stable memory-only fixes, no net
control regression, source-local fallback remains net-positive, and temporal ordering
reaches at least 3/4 on the existing temporal slice. Only then buy the 48-question run.

## Expected target

The frozen comparable project result remains 72% on the 100-question internal test;
this targeted gate does not replace it. A realistic next milestone is **78-82%** under
the same protocol after the three failures above are addressed. A genuinely strong
2026 hybrid-memory target is **85%+ on a fresh, protocol-matched external set**, with
**90%+** as a stretch goal. Public 2026 claims around 92-95% use different answerers,
judges, context budgets, or managed pipelines and are not direct evidence that this
project should already score there.
