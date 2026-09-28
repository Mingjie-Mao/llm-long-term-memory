# Raw-primary preflight on heldout100 — evidence reach, aggregates only

> DEVELOPMENT, zero calls. Descriptive preflight for `results/prereg-raw-primary-heldout100-v1.md`;
> the 4,000-token budget was fixed on train150 before this ran. Per-question rows are
> withheld under the heldout100 protocol.
> Reach, not use: coverage means the gold turn is in the context, not that it is
> answered correctly.

Store `heldout100`, manifest `heldout100.json`, retrieval `configs/fallback.yaml`. 93 questions with flagged gold turns; 7 without any were excluded. Gold-turn text mismatches between corpus and store: 0.

Index hit rate — a top-20 memory is anchored to a gold turn: **73.1%**; top 50: 74.2%.

## All-gold coverage by budget

| arm | 1,000 | 2,000 | 4,000 | 8,000 |
|---|---:|---:|---:|---:|
| `mem_turn` | 50.5% (991) | 53.8% (1,988) | 54.8% (3,987) | 55.9% (7,960) |
| `mem_window` | 15.1% (945) | 47.3% (1,951) | 54.8% (3,948) | 59.1% (7,956) |
| `mem_session_bm25` | 66.7% (997) | 72.0% (1,998) | 76.3% (3,998) | 78.5% (7,997) |
| `bm25_turns` | 74.2% (998) | 80.6% (1,999) | 86.0% (3,999) | 88.2% (7,999) |
| `mem_session_whole` | 2.2% (763) | 5.4% (1,785) | 24.7% (3,700) | 65.6% (7,700) |

Cells: share of questions with every gold turn in context (median tokens used).

## Paired against `bm25_turns` at the same budget (all-gold)

| arm | 1,000 | 2,000 | 4,000 | 8,000 |
|---|---:|---:|---:|---:|
| `mem_turn` | +8 / −30 | +8 / −33 | +4 / −33 | +5 / −35 |
| `mem_window` | +0 / −55 | +5 / −36 | +4 / −33 | +5 / −32 |
| `mem_session_bm25` | +0 / −7 | +1 / −9 | +1 / −10 | +1 / −10 |
| `mem_session_whole` | +0 / −67 | +0 / −70 | +1 / −58 | +4 / −25 |

## By question type at 4,000 tokens (all-gold)

| type | n | `mem_turn` | `mem_window` | `mem_session_bm25` | `bm25_turns` | `mem_session_whole` |
|---|---:|---:|---:|---:|---:|---:|
| knowledge-update | 14 | 50% | 57% | 79% | 93% | 7% |
| multi-session | 24 | 67% | 67% | 79% | 83% | 4% |
| single-session-assistant | 11 | 18% | 36% | 55% | 100% | 55% |
| single-session-preference | 6 | 33% | 33% | 67% | 83% | 17% |
| single-session-user | 12 | 50% | 58% | 92% | 92% | 92% |
| temporal-reasoning | 26 | 69% | 54% | 77% | 77% | 12% |
