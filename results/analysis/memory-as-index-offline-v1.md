# Memory as an index — offline evidence reach v1

> DEVELOPMENT, zero calls. Registered in `results/prereg-memory-as-index-offline-v1.md`.
> Reach, not use: coverage means the gold turn is in the context, not that it is
> answered correctly.

Store `train150`, manifest `train150.json`, retrieval `configs/v2.yaml`. 146 questions with flagged gold turns; 4 without any were excluded. Gold-turn text mismatches between corpus and store: 0.

Index hit rate — a top-20 memory is anchored to a gold turn: **73.3%**; top 50: 73.3%.

## All-gold coverage by budget

| arm | 1,000 | 2,000 | 4,000 | 8,000 |
|---|---:|---:|---:|---:|
| `mem_turn` | 41.8% (988) | 45.2% (1,990) | 45.9% (3,988) | 47.3% (7,964) |
| `mem_window` | 19.9% (962) | 38.4% (1,942) | 50.0% (3,951) | 56.8% (7,960) |
| `mem_session_bm25` | 56.2% (998) | 66.4% (1,997) | 70.5% (3,997) | 77.4% (7,997) |
| `bm25_turns` | 62.3% (998) | 71.9% (1,999) | 78.1% (3,998) | 85.6% (7,999) |
| `mem_session_whole` | 5.5% (714) | 8.9% (1,756) | 25.3% (3,732) | 63.0% (7,726) |

Cells: share of questions with every gold turn in context (median tokens used).

## Paired against `bm25_turns` at the same budget (all-gold)

| arm | 1,000 | 2,000 | 4,000 | 8,000 |
|---|---:|---:|---:|---:|
| `mem_turn` | +17 / −47 | +14 / −53 | +13 / −60 | +10 / −66 |
| `mem_window` | +4 / −66 | +6 / −55 | +9 / −50 | +11 / −53 |
| `mem_session_bm25` | +1 / −10 | +2 / −10 | +2 / −13 | +3 / −15 |
| `mem_session_whole` | +1 / −84 | +0 / −92 | +3 / −80 | +7 / −40 |

## By question type at 4,000 tokens (all-gold)

| type | n | `mem_turn` | `mem_window` | `mem_session_bm25` | `bm25_turns` | `mem_session_whole` |
|---|---:|---:|---:|---:|---:|---:|
| knowledge-update | 21 | 38% | 62% | 95% | 95% | 0% |
| multi-session | 39 | 36% | 38% | 44% | 54% | 3% |
| single-session-assistant | 17 | 53% | 65% | 94% | 94% | 94% |
| single-session-preference | 9 | 22% | 22% | 44% | 67% | 33% |
| single-session-user | 20 | 50% | 55% | 95% | 95% | 70% |
| temporal-reasoning | 40 | 60% | 52% | 68% | 80% | 8% |

## Registered gate

- best memory arm at 4,000: `mem_window`, coverage 50.0% (needs ≥ 80%): fail
- not worse than `bm25_turns` at 4,000: fail
- budget carried forward: 8,000

**Decision: STOP**
