# Fixed-model v12 — action/location and event eligibility

2026-10-02, before v12 output. V11 19 real answers retained UNGRADED:26requests,
261271tokens,1failure; book total8weeks and numbered successor28 now work.
Coupon still substitutes inbox origin for redemption place, consecutive-event
repair drops the requested ending event, and year count includes undated wedding.
No promotion. V11 grading not run because source-supported failures already fail
the diagnostic acceptance condition; preserve ungraded outputs rather than report a score.

Same models, same19 manifests, fixed source pool, max2 reader calls before retries.
V12 narrowly implements generic verifiable relations, not question ids or gold:
- For where-redeem-coupon, locate literal completed redemption in raw user source.
  Explicit action-bound named place has priority. A unique named shopping place in
  the same original user session supports a labeled likely inference, with both
  action/context citations. Multiple candidate places stay uncertain. Email/coupon
  origin or a dollar amount alone cannot validate the redemption-place relation.
- For elapsed time since N consecutive CATEGORY events, collect distinct named
  completed raw user events with independently anchored day expressions. Require
  one unambiguous sequence of N consecutive days; select its ending event and
  question date. Don't substitute an isolated later event or the sequence start.
  Named repeated mentions on a day deduplicate; conflicting dates/roles/plans
  cannot establish a sequence. Near-return wording may bind today in the same
  sentence to an event just returned from, with actual wording/inference preserved.
- Explicit this-year member counts require temporal eligibility of each selected
  record; quote/source attendance and year support must be reviewed independently
  from numeric calculation. Undated event mention or planning alone cannot prove
  attendance this year. Return uncertainty, not a gold-forced exact total. This
  guard is conservative and not a complete natural language year parser.

Target positive/counterexample tests for role/session/location ambiguity and
message origin vs action place; sequence end/duplicates/conflicting dates/negation;
year count undated/planned/wrong-year. Tenant boundary tests stay mandatory.
Targeted then full suite/Ruff; zero-call saved-v11 replay and context coverage gate
146/146 +median<=6000 +raw SHA unchanged; dev100 context-only. New v12 namespace,
execution/source archive/payload preview/public provenance audit. Same19 exposed
questions are a targeted development/regression diagnostic, no significance or
new overall correctness figure. Reader<=38requests before retries,~400k tokens cap
estimate; judge original gemma only after source-supported diagnostic passes.
No commits, model/framework switch, default promotion, or frozen-history edits.
