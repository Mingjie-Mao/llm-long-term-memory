# v2c gate8 preregistration

Registered before any v2c answer call. The candidate reuses the completed v2b store;
there is no ingestion spend.

## Candidate

- Query-time update repair suppresses an older related value only when a newer,
  query-relevant fact explicitly carries `update_op=replaces`.
- Exact-date questions hydrate anchored source evidence before the first answer when
  relevant structured memories contain no day-level date.
- Sequence questions receive a filtered chronological evidence note built from the
  already-selected memories.
- Existing conditional raw fallback remains available for other missing details.

Two arms isolate the mechanism:

1. `two_stage_v2c_memory_only`: update repair plus temporal assembly, no raw evidence.
2. `two_stage_v2c`: the same repairs plus precision-triggered hydration and conditional
   fallback.

## Fixed rows

Five diagnosed rows: `58ef2f1c`, `6a1eabeb`, `gpt4_45189cb4`, `778164c6`, and
`d905b33f`. Three controls already answered correctly in v2b: `45dc21b6`, `6613b389`,
and `a96c20ee`.

This is a mechanism gate, not an accuracy estimate.

## Spend and decision

1. Run the zero-call plan audit first. Stop if any control receives an update
   suppression or an unrelated temporal plan, or if the three diagnosed failures do
   not receive their intended deterministic repair.
2. If offline checks pass, run each arm twice on the same eight rows. Repeats are
   required because v2b showed one-answer variance.
3. Promote to the existing 48-question development arm only if:
   - both memory-only runs fix at least two diagnosed failures relative to the frozen
     v2b outcome;
   - neither memory-only run has a net control regression;
   - both final runs are net-positive versus frozen v2b;
   - both source-local rescue rows remain correct;
   - temporal ordering is correct in both runs; and
   - no deterministic repair fires on an unrelated control.

No new baseline calls are permitted. Frozen v2b rows are the paired control.
