# two_stage_v5_fixed.v5-reasoning48-fixed

> **DEVELOPMENT RESULT.** These 48 questions have been read in full. The baseline
> is the committed v2c run rather than a fresh one, so the arms were measured on
> different days. Directional only; no significance is claimed.

Decision: **STOP**

- baseline: **37/48**
- candidate: **37/48**
- paired: **2 wins / 2 losses / 44 ties**, net **+0**
- exact McNemar p = 1.0000 (shape only — one run per arm)
- median context: 634 -> 878 tokens
- the mechanism recovered turns on **48/48** questions

## Registered gates

- FAIL — `net_positive`
- PASS — `no_reader_attributed_regression`
- PASS — `mechanism_fired`
- FAIL — `registered_sentences_arrived`
- PASS — `median_context_at_most_2000`

## The three failures this arm was registered for

| question | sentence | reached the reader | v2c | v5 |
|---|---|---|---|---|
| `gpt4_7a0daae1` | received my new tennis racket today | yes | wrong | wrong |
| `a4996e51` | 50 hours per week | **no** | wrong | correct |
| `58ef2f1c` | valentine's day | yes | wrong | wrong |

Wins: `0ddfec37_abs`, `a4996e51`

Losses: `6e984301`, `982b5123`
