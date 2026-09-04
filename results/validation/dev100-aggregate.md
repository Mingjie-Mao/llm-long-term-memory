# Aggregate validation report

No question ids, answers, model answers, or judge reasons are included.

| arm | repeats | accuracy (runs) | mean ± sd | majority | agreement | candidate/ranked/selected recall | median context | requests | tokens |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| `flat20` | 3 | 64.0%, 63.0%, 66.0% | 64.3% ± 1.5% | 67.0% | 86.0% | 97.0%/95.0%/95.0% | 576 | 723 | 470,441 |
| `coherent-auto` | 3 | 64.0%, 65.0%, 63.0% | 64.0% ± 1.0% | 63.0% | 91.0% | 97.0%/95.0%/93.0% | 343 | 695 | 344,829 |
| `coherent-oracle` | 3 | 66.0%, 64.0%, 66.0% | 65.3% ± 1.2% | 64.0% | 90.0% | 97.0%/95.0%/97.0% | 240 | 696 | 306,878 |
| `naive_rag` | 3 | 67.0%, 71.0%, 70.0% | 69.3% ± 2.1% | 68.0% | 90.0% | —/—/— | 13,416 | 604 | 4,143,048 |
| `memory-only` | 3 | 57.0%, 59.0%, 58.0% | 58.0% ± 1.0% | 59.0% | 93.0% | 97.0%/95.0%/95.0% | 540 | 604 | 384,821 |

## API usage (cost evidence)

| phase | requests | failed requests | input tokens | output tokens | total tokens |
|---|---:|---:|---:|---:|---:|
| shared ingestion | 1,424 | 3 | 12,268,748 | 1,523,669 | 13,792,417 |
| answer + judge arms | 3,322 | 42 | 5,523,680 | 126,337 | 5,650,017 |
| **total** | **4,746** | **45** | **17,792,428** | **1,650,006** | **19,442,434** |

> Usage artifacts prove requests and tokens. They do not contain provider billing; a dollar amount must come from the provider's billing export rather than be guessed.

## Majority-vote paired comparisons

| baseline → candidate | wins-losses | accuracy delta | exact p |
|---|---:|---:|---:|
| `flat20` → `coherent-auto` | 5-9 | -4.0% | 0.4240 |
| `flat20` → `coherent-oracle` | 7-10 | -3.0% | 0.6291 |
| `flat20` → `naive_rag` | 15-14 | +1.0% | 1.0000 |
| `flat20` → `memory-only` | 5-13 | -8.0% | 0.0963 |

## `flat20` failure stages

Raw fallback fired on 34.0% of runs; 20.0% of all run outcomes were correct after raw fallback.

| question type | majority accuracy |
|---|---:|
| `knowledge-update` | 68.8% |
| `multi-session` | 51.9% |
| `single-session-assistant` | 81.8% |
| `single-session-preference` | 83.3% |
| `single-session-user` | 92.9% |
| `temporal-reasoning` | 57.7% |

| observed stage | questions |
|---|---:|
| `wrong_after_raw_fallback` | 12 |
| `wrong_despite_gold_session_context` | 21 |

## `coherent-auto` failure stages

Raw fallback fired on 28.7% of runs; 17.0% of all run outcomes were correct after raw fallback.

| question type | majority accuracy |
|---|---:|
| `knowledge-update` | 62.5% |
| `multi-session` | 51.9% |
| `single-session-assistant` | 54.5% |
| `single-session-preference` | 83.3% |
| `single-session-user` | 92.9% |
| `temporal-reasoning` | 57.7% |

| observed stage | questions |
|---|---:|
| `gold_session_lost_in_context` | 1 |
| `wrong_after_raw_fallback` | 11 |
| `wrong_despite_gold_session_context` | 25 |

## `coherent-oracle` failure stages

Raw fallback fired on 30.0% of runs; 19.3% of all run outcomes were correct after raw fallback.

| question type | majority accuracy |
|---|---:|
| `knowledge-update` | 62.5% |
| `multi-session` | 55.6% |
| `single-session-assistant` | 72.7% |
| `single-session-preference` | 83.3% |
| `single-session-user` | 92.9% |
| `temporal-reasoning` | 50.0% |

| observed stage | questions |
|---|---:|
| `wrong_after_raw_fallback` | 11 |
| `wrong_despite_gold_session_context` | 25 |

## `naive_rag` failure stages

Raw fallback fired on 0.0% of runs; 0.0% of all run outcomes were correct after raw fallback.

| question type | majority accuracy |
|---|---:|
| `knowledge-update` | 81.2% |
| `multi-session` | 55.6% |
| `single-session-assistant` | 100.0% |
| `single-session-preference` | 100.0% |
| `single-session-user` | 71.4% |
| `temporal-reasoning` | 50.0% |

| observed stage | questions |
|---|---:|
| `stage_trace_unavailable` | 32 |

## `memory-only` failure stages

Raw fallback fired on 0.0% of runs; 0.0% of all run outcomes were correct after raw fallback.

| question type | majority accuracy |
|---|---:|
| `knowledge-update` | 62.5% |
| `multi-session` | 59.3% |
| `single-session-assistant` | 36.4% |
| `single-session-preference` | 100.0% |
| `single-session-user` | 71.4% |
| `temporal-reasoning` | 50.0% |

| observed stage | questions |
|---|---:|
| `gold_session_lost_before_top_k` | 1 |
| `no_gold_session_in_candidates` | 2 |
| `wrong_despite_gold_session_context` | 38 |

> These are source-session stage labels, not proof that the exact answer fact survived extraction. `wrong_despite_gold_session_context` therefore combines possible extraction, reasoning, and judge errors.

## Registered dev100 decision

- selected arm: `flat20`
- selected product variant: `two_stage_fallback`
- diagnostic: `coherent_shape_not_supported`
- coherent accuracy gate: False
- coherent context ratio: 0.60 (gate passed: True)
