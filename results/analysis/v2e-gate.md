# v2e reasoning-48

> **DEVELOPMENT RESULT.** The 48 questions have been read in full, twice. The
> baseline is the committed v2c run rather than a fresh one, so the two arms were
> measured on different days. Directional only; no significance is claimed.

Decision: **STOP**

- baseline: **37/48**
- candidate: **37/48**
- paired: **5 wins / 5 losses / 38 ties**, net **+0**
- exact McNemar p = 1.0000 (reported for shape, not as a claim — one run per arm)
- median context: 634 -> 692 tokens (+9.0%)
- the marker reached **48/48** contexts

## Registered gates

- FAIL — `net_positive`
- FAIL — `no_temporal_regression`
- PASS — `marker_reached_every_temporal_question`
- PASS — `context_within_10_percent`

## By question type

| type | n | v2c | v2e |
|---|---:|---:|---:|
| knowledge-update | 8 | 7 | 7 |
| multi-session | 12 | 7 | 7 |
| single-session-assistant | 6 | 5 | 6 |
| single-session-preference | 4 | 4 | 4 |
| single-session-user | 6 | 5 | 5 |
| temporal-reasoning | 12 | 9 | 8 |

Wins: `0ddfec37_abs`, `a9f6b44c`, `dd2973ad`, `e48988bc`, `gpt4_70e84552_abs`

Losses: `0e4e4c46`, `3a704032`, `6e984301`, `982b5123`, `gpt4_731e37d7`
