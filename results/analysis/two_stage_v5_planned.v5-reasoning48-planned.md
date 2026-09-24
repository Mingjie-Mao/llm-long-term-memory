# two_stage_v5_planned.v5-reasoning48-planned

> **DEVELOPMENT RESULT.** These 48 questions have been read in full. The baseline
> is the committed v2c run rather than a fresh one, so the arms were measured on
> different days. Directional only; no significance is claimed.

Decision: **STOP**

- baseline: **37/48**
- candidate: **36/48**
- paired: **3 wins / 4 losses / 41 ties**, net **-1**
- exact McNemar p = 1.0000 (shape only — one run per arm)
- median context: 634 -> 986 tokens
- the mechanism recovered turns on **48/48** questions

## Registered gates

- FAIL — `net_positive`
- PASS — `no_reader_attributed_regression`
- PASS — `mechanism_fired`
- PASS — `registered_sentences_arrived`
- PASS — `median_context_at_most_2000`

## The three failures this arm was registered for

| question | sentence | reached the reader | v2c | v5 |
|---|---|---|---|---|
| `gpt4_7a0daae1` | received my new tennis racket today | yes | wrong | wrong |
| `a4996e51` | 50 hours per week | yes | wrong | correct |
| `58ef2f1c` | valentine's day | yes | wrong | wrong |

Wins: `a4996e51`, `dd2973ad`, `e48988bc`

Losses: `57f827a0`, `6e984301`, `778164c6`, `982b5123`
