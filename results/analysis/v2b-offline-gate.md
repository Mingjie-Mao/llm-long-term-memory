# v2b zero-call gate

**No model/API calls were made.** The archived 60-session paired cohort still matches the corpus: `True`; prompts match: `True`.

## Extraction change

Batch 15 retained **37.3%** of annotated specifics with 92 memories. Batch 8 retained **64.9%** with 223 memories: **+27.6 points** and +131 emitted memories.

| facet | gained | lost | batch15 | batch8 |
|---|---:|---:|---:|---:|
| date | 0 | 0 | n/a | n/a |
| duration | 6 | 1 | 35.7% | 71.4% |
| money | 0 | 0 | 0.0% | 0.0% |
| proper_noun | 19 | 1 | 34.0% | 67.9% |
| quantity | 14 | 2 | 44.4% | 66.7% |
| relative_time | 2 | 0 | 25.0% | 41.7% |

## Low-support screen

- `batch15`: median source-word support 87.5%; 0/92 below 25%; 0 with zero overlap.
- `batch8`: median source-word support 87.5%; 0/223 below 25%; 0 with zero overlap.

These are **not hallucination rates**: paraphrases can have low overlap and false claims can reuse source words. They are only a cheap regression floor.

## Can the archived cohort predict QA?

Only **1** benchmark question has all gold sessions in this extraction cohort, so the archived fidelity run cannot honestly estimate QA lift.
- `00ca467f`: literal gold present batch15=`True`, batch8=`True`.

## Offline retrieval decisions

Raw BM25 natural-query recall is 45.8% @1, 64.6% @2, and 68.8% @3 over 48 questions in the current `stores/two-stage-hydrated.db` snapshot (50 memory namespaces). Raw top-2 remains worth testing only behind the per-question coverage gate.
- `smaller_top_k`: Rejected as a general mechanism: archived context-arms reruns found flatN worse than flat20 on one target and no rescue on the other stable failures.
- `dense_raw_retrieval`: Not justified by the archived diagnostic: natural-query BM25 misses were mostly aggregation, temporal, or preference tasks rather than lexical misses.
- `raw_top2_parallel`: Still eligible for a gate16 evidence comparison because it is cheap, but must increase per-question source coverage before any answer calls.

## Decision

Batch 8 passes the **extraction-mechanism** gate, but archived data cannot prove end-to-end QA transmission. Proceed only to the registered 16-question candidate ingest; do not run 48 questions yet. Ordinary-fact fidelity and true unsupported rate remain unresolved and must be checked on the gate16 output before promotion.
