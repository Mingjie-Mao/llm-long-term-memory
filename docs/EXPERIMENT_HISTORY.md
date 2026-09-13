# Experiment history

This consolidates the two overlapping timelines previously in the README. Each row is
an experiment on its own sample. Compare candidate and control **within a row**, not
accuracy values across question sets. `dev60` deliberately contains many temporal and
multi-session questions.

| # | Phase | What changed | Set | Result | Median context |
|---|---|---|---|---|---:|
| 1 | v1 | First end-to-end system: two-stage extraction, bi-temporal facts, hybrid retrieval | `heldout100` | **70%** · repeats 71, 73 | 1,468 |
| 2 | v2 dev | Chose the arm: flat ranking against session-coherent context, plus baselines | `dev100` | flat **67%** · coherent 63% · naive RAG 68% | 576 |
| 3 | **v2 final** | Nothing — the frozen v2 measured against two baselines on unseen data | `test100` | **72%** · full transcript **86%** · naive RAG 65% | **574** |
| 4 | v3.0 | Answer policy: name the reasoning operation, then answer | `pilot48` | **72.9%** vs control 68.8% · `p=0.6875` | 605 |
| 5 | v3.1 | Same idea, tuned | `tune42` | **66.7% vs 66.7%** — tied, both target slices fell 10 points · **negative result** | 558 |
| 6 | v3.2 | Hydrate verbatim source for temporal/aggregation questions | `tune42` | 73.8% vs 66.7%, but 2.52x context against a registered 2x ceiling · **STOP** | 1,477 |
| 7 | v3.3 | Same evidence, compacted: exact sentences, session-fair, 425-token cap | `tune42` | **73.8%** held at 1.92x · gate passed | 1,124 |
| 8 | **v3.3 validation** | Nothing — the frozen candidate against the v2 control | `dev60` | **55.0%** vs **46.7%** · +8.3 points · `p=0.1797` · see confidence audit below | 1,115 |
| 9 | Diagnosis | Failure taxonomy over 222 failures, and an offline probe · **zero API calls** | 3 readable pools | retrieval misses **0–6%** · abstention-with-source **44–50%** | — |
| 10 | Instrument | 229 synthesis probes, ground truth by SQL | derived from `train150` | `current_state` 86.7% · `comparison` 58.7% · `duration` 42.5% · `count` 4.0% | 1,120 |
| 11 | v4.0 flat | Operation-first verdict and Python arithmetic; timeline rendering disabled in this arm | 142 development probes | **61.3%** vs 54.9% · 22 wins / 13 losses · `p=0.1755` | — |

Before these stages, the first memory-only version scored 26.0% on `dev50`, at 465
median context tokens, against 56.0% for full context and 54.0% for naive RAG. After
the extraction rewrite, memory-only reached 56.0%; the conditional raw-fallback arm
reached 72.0% at 1,455 median context tokens. These are development results, not the
later `test100` measurement.

The earlier `heldout100` repeats were 70, 71 and 73 (mean 71.3%; range 3 points).
The v2 final `test100` arms each ran once; no repeat variance is available for that run.
The v3 pilot and dev60 comparisons used three repeats per arm, with majority accuracy
reported. The two tune-stage negative results remain part of the history: v3.1 tied,
and v3.2 exceeded its registered context ceiling despite a higher accuracy.

## Qualifications that belong with the history

- The archived v3.3 dev60 run passed the checks as implemented. A later schema audit
  established that its confidence field was fixed at a default, so the confidence gate
  did not measure the intended condition. Eight other gates remain interpretable;
  the reported accuracy and paired comparison are unaffected. This is a qualification
  of the archive, not a rewritten frozen result.
- Selected source recall measures whether a selected memory comes from **any** gold
  session. It does not establish full evidence coverage or correct fact extraction.
  The 98.3% recall and 55.0% accuracy are useful diagnostic context, not proof that the
  entire gap is answer synthesis.
- The 229 synthesis probes were generated from a readable store with SQL-derived
  targets. They are diagnostic instruments, not an external benchmark or an unseen
  final test. Later development runs use a 142-probe subset; those percentages cannot
  be directly compared with the whole original pool.
- v4.0 flat has already been measured; describing v4 as entirely unmeasured is stale.
  Its +6.3-point development result remains statistically inconclusive. Relation
  routing/scanning is a separate research direction, not part of the default service.

## Evidence

| Stage | Record |
|---|---|
| Initial development | [Frozen v1 table](../results/frozen/chronomem-v1/table.md) |
| Earlier held-out repeats | [Held-out variance](../results/heldout-variance.md) |
| v2 selection | [dev100 aggregate](../results/validation/dev100-aggregate.md) |
| v2 final | [test100 aggregate](../results/final/test100-aggregate.md) |
| v3 stages | [Pilot](../results/validation/v3-answer-pilot-aggregate.md), [tune1](../results/validation/v3-phase3-tune1.md), [tune2](../results/validation/v3-phase4-tune2.md), [tune3](../results/validation/v3-phase5-tune3.md) |
| v3.3 validation | [dev60 aggregate](../results/validation/v3-dev60.md), [schema audit](../results/audit/v3-verdict-schema-never-sent-20260906.json) |
| Diagnosis | [Failure taxonomy](../results/failure-taxonomy.md), [synthesis probes](../results/analysis/synthesis-probes-v3.3.md) |
| v4 development | [First attempt](../results/archive/v4.0-attempt1/README.md), [flat arm](../results/archive/v4.0-flat/README.md) |

See [current status](CURRENT_STATUS.md) for the evolving research log and the
[architecture atlas](ARCHITECTURE.md) for the actual current wiring.
