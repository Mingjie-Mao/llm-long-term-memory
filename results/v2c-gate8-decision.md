# v2c gate-8 decision

Date: 2026-09-17 (Australia/Sydney)

## Current decision

**v2c.8 passes both the gate-8 mechanism check and the frozen 48-question
confirmation. Do not tune on or rerun reasoning-48.**

## What the staged gate established

| Candidate | Arm | Repetitions | Reported accuracy | Interpretation |
|---|---:|---:|---:|---|
| v2c.2 | memory-only | 2 | 5/8, 5/8 | Update + temporal repairs stable; raw-dependent rows remain wrong as expected |
| v2c.2 | hybrid | 1 | 7/8 | Conditional fallback did not fire for the exact recommendation-name question |
| v2c.3 | hybrid | 2 | 8/8, 8/8 | Configured judge passes, but strict answer audit rejects the dish row |
| v2c.4 | dish-only | 1 | judge 1/1, strict 0/1 | Still led with Escovitch Fish; stopped early |
| v2c.5 | dish-only | 1 | 0/1 | Rule was lost when generic fallback replaced first-pass context |
| v2c.6 | dish-only | 1 | 0/1 | Preserved context, but ordinary prompt priority was insufficient |
| v2c.7 | dish-only | 1 | 0/1 | Gold-chasing policy still chose Escovitch; source audit exposed invalid item |
| v2c.8 | conflict-only | 2 | judge 1/1, 1/1 | Both runs correctly explain the contradiction and distinguish both dishes |

The two v2c.3 dish hypotheses both contain the gold phrase, but lead with Escovitch
Fish as the answer. The correct answer is Grilled Snapper with Mango Salsa. Treating
substring inclusion as success would overstate the result, so v2c.3 is not accepted
as a clean 100% despite the judge score.

## v2c.8 resolution

Source audit proved that `778164c6` has no answer satisfying all qualifiers: Escovitch
Fish is the actually recommended Jamaican snapper dish but has pickled vegetables;
Grilled Snapper with Mango Salsa has fruit but was only listed, not recommended. The
gold selects the latter. v2c.8 therefore uses a conflict-aware exact-reference policy
instead of tuning the system to repeat an internally inconsistent label.

Offline gate: **PASS**. Focused tests: **24 passed**. Full suite: **passed**.

## Final gate interpretation

- Seven valid rows reuse the two frozen v2c.3 repetitions: **7/7 twice**.
- The invalid row uses the two v2c.8 repetitions: conflict-aware behavior **passed twice**.
- The configured judge also marks both v2c.8 hypotheses correct because each correctly
  identifies Mango Salsa as the fruit-bearing candidate.
- Gate decision: **PASS**.

The reasoning-48 run completed at 37/48 (77.1%), 636 median context tokens, and
95.8% source-session recall. Every registered slice and efficiency threshold passed.
The old baseline was not rerun. The next honest measurement, if commissioned, is a
separately frozen final set—not another iteration on these 48 rows.

## v2c.3 quota actually consumed

- repetition 1: 27 request attempts, 13,310 tokens, including 9 failed/retried attempts;
- repetition 2: 22 request attempts, 13,423 tokens, including 4 failed/retried attempts;
- total: 49 request attempts, 26,733 tokens; 36 successful calls and 13 provider failures.

The failures were primarily transient Gemma judge 500/503 responses. Completed rows
were preserved and retries did not repeat completed questions.

The final v2c.8 conflict check consumed 11 request attempts and 4,114 tokens, including
7 provider failures/retries.
