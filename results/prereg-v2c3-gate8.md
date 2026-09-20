# v2c.3 gate-8 preregistration

Date: 2026-09-17 (Australia/Sydney)

## Evidence before this change

- v2c.2 memory-only passed its two-run stability gate at 5/8 both times: the
  update and temporal targets were repaired, all three controls remained correct,
  and the three source-dependent targets remained wrong as expected.
- v2c.2 hybrid repetition 1 scored 7/8. The sole failure was `778164c6`.
- On that row, retrieval found the correct source session, but the first answerer
  returned `answer` rather than `need_source`, so no raw fallback ran. A zero-call
  replay of the existing store ranks an assistant turn containing “Grilled Snapper
  with Mango Salsa” first within that already-located session.

The v2c.2 hybrid artifact is exploratory evidence and is not counted as a v2c.3
formal repetition.

## Frozen change

v2c.3 adds one deterministic detail-gap condition: when a question explicitly asks
for the name/title/URL/link of a prior recommendation or suggestion, attach up to
three query-ranked raw turns from only the source sessions already identified by
the selected memories. No archive-wide search result is attached by this route.

No ingestion, store, embedding, retrieval weights, top-k, update repair, timeline
repair, answer model, judge model, or judging prompt changes.

## Zero-call gate

Before model evaluation:

1. `778164c6` must route to exact-reference detail hydration.
2. Its source-local evidence must contain “Mango Salsa”.
3. All pre-existing v2c.2 offline checks must remain green.

Stop if any check fails.

## Paid gate

Reuse the two completed v2c.2 memory-only repetitions because v2c.3 changes only
the hybrid-only source attachment path. Run `two_stage_v2c` twice on the frozen
eight-question manifest with distinct labels.

Pass only if both repetitions satisfy all of the following:

- at least 7/8 overall;
- all three controls remain correct;
- update target `6a1eabeb` and temporal target `gpt4_45189cb4` remain correct;
- source-local targets `58ef2f1c`, `778164c6`, and `d905b33f` are all correct;
- no archive-wide fallback is used by the new exact-reference route.

Do not expand to 48 questions under the existing eight-question authorization.
