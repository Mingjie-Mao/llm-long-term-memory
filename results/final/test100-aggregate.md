# Final one-shot test report

Each frozen arm ran exactly once. No repeat variance or agreement statistic exists.
No question ids, answers, model answers, or judge reasons are included.

| arm | accuracy | candidate/ranked/selected recall | median context | requests | tokens |
|---|---:|---|---:|---:|---:|
| `v2` | 72.0% | 96.0%/94.0%/94.0% | 574 | 236 | 159,458 |
| `full_context` | 86.0% | —/—/— | 109,059 | 204 | 10,932,294 |
| `naive_rag` | 65.0% | —/—/— | 12,763 | 207 | 1,352,847 |

## API usage (cost evidence)

| phase | requests | failed requests | input tokens | output tokens | total tokens |
|---|---:|---:|---:|---:|---:|
| shared ingestion | 1,386 | 36 | 12,247,826 | 1,509,887 | 13,757,713 |
| answer + judge arms | 647 | 12 | 12,417,781 | 26,818 | 12,444,599 |
| **total** | **2,033** | **48** | **24,665,607** | **1,536,705** | **26,202,312** |

> Requests and tokens are auditable cost evidence. Provider billing is reported separately when available; no dollar amount is guessed.

## Paired single-run comparisons

| baseline → v2 | wins-losses | accuracy delta | exact p |
|---|---:|---:|---:|
| `full_context` → `v2` | 4-18 | -14.0% | 0.0043 |
| `naive_rag` → `v2` | 23-16 | +7.0% | 0.3368 |

## `v2` failure stages

Raw fallback fired on 35.0%; 18.0% of outcomes were correct after it.

| question type | accuracy |
|---|---:|
| `knowledge-update` | 75.0% |
| `multi-session` | 61.5% |
| `single-session-assistant` | 81.8% |
| `single-session-preference` | 50.0% |
| `single-session-user` | 85.7% |
| `temporal-reasoning` | 74.1% |

| observed stage | questions |
|---|---:|
| `no_gold_session_in_candidates` | 3 |
| `wrong_after_raw_fallback` | 14 |
| `wrong_despite_gold_session_context` | 11 |

## `full_context` failure stages

Raw fallback fired on 0.0%; 0.0% of outcomes were correct after it.

| question type | accuracy |
|---|---:|
| `knowledge-update` | 81.2% |
| `multi-session` | 69.2% |
| `single-session-assistant` | 100.0% |
| `single-session-preference` | 100.0% |
| `single-session-user` | 100.0% |
| `temporal-reasoning` | 88.9% |

| observed stage | questions |
|---|---:|
| `stage_trace_unavailable` | 14 |

## `naive_rag` failure stages

Raw fallback fired on 0.0%; 0.0% of outcomes were correct after it.

| question type | accuracy |
|---|---:|
| `knowledge-update` | 56.2% |
| `multi-session` | 57.7% |
| `single-session-assistant` | 100.0% |
| `single-session-preference` | 50.0% |
| `single-session-user` | 85.7% |
| `temporal-reasoning` | 55.6% |

| observed stage | questions |
|---|---:|
| `stage_trace_unavailable` | 35 |

> Failure stages track whether the source session survived. They do not prove that the exact answer fact survived extraction.
