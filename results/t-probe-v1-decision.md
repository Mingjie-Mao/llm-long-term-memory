# Small t1/t2 probe — reading and diagnosis

2026-10-07. Registered in `results/prereg-t-probe-v1.md`; reading
`results/analysis/t-probe-v1.{json,md}` (`tools/t_probe_report.py`). Diagnostic, one run
per arm, exposed train150 (out of design) and dev100 (design set). Not an accuracy
figure. Five runs were interrupted by judge 500/503 errors and resumed from their saved
rows with the same protocol; every row is a single fresh answer and grade.

## Registered reading

| train150 | v1 rerun | t1 | t2 |
|---|---:|---:|---:|
| correct on 8 temporal failures | 0 | 1 | 5 |
| correct on 11 multi-session / knowledge-update failures | 2 | 5 | 4 |
| wrong on 12 controls | 0 | 1 | 1 |

t1 moves its target: **no** (+1, floor +2). t2 moves its target: **yes** (+2). No harm
signal for either (one control lost each). **Proceed to heldout100: no** — as registered,
the three-day run does not start; diagnose first. dev100 design set, reported only: t1
4 / 9 against v1's expected 2.33; t2 9 / 18 against 3.0.

## Diagnosis (zero calls, train150 per-item reading allowed)

1. **t2's temporal gain comes from the notes path, not from the date notes alone.** Six
   of the eight temporal failures are "how many days / months …" questions, which
   `asks_for_aggregate` routes to item notes with 1,024 output tokens; t2 fixed five of
   those six. The two that took the plain path were not fixed; t1 (date notes, no notes
   path) fixed one of eight. The two cannot be separated from this probe: t2 carries
   both.
2. **Most failures left are retrieval, not reading.** Of the ten failures neither arm
   fixed with complete evidence, nine have gold turns missing from v1's 4,000-token
   excerpts: multi-session counts spread over many sessions (2/3, 3/4, 1/4, 1/2, 4/6,
   3/5 gold turns present) and three temporal questions asking about "a week ago",
   "last week", "last Saturday" whose gold turn is not retrieved at all (0/1). v4's
   fused, fact-keyed retrieval moves these both ways (36b9f61e 2→3, b5700ca0 0→1;
   gpt4_a56e767c 4→1, gpt4_ab202e7f 3→2).
3. Count notes did not fix multi-session counts with incomplete evidence: listing items
   cannot recover items that never reached the context.

## What follows (to be registered separately)

- Route every temporal-reasoning-style question (dates, durations, order), not only
  "how many", to the notes path — the active ingredient in (1).
- Time-aware retrieval: resolve a relative time in the question ("last Saturday") to a
  date window and add that window's turns — LongMemEval's time-aware query expansion;
  its reach on the 0/1 questions can be measured offline first.
- Multi-session counts: retrieval reach, not answering — a larger or adaptive budget for
  count questions, or session-level retrieval; measure reach offline first.
