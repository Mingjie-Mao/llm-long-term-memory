# Evaluation Protocol and Interpretation

## Scope of the published v1 result

The four published rows use a stratified 50-question LongMemEval-S development
subset. They are diagnostic results, not a final benchmark claim. Category-level
percentages can be based on small denominators and are reported for failure
localization, not as evidence of a general improvement.

The result is deliberately negative: v1 LLTM scores 26.0% against naive RAG's
54.0%. The exact paired comparison for temporal filtering is 3 wins and 3 losses
(`p = 1.000`), so the only supported statement is **no detectable difference at
n=50**. It does not establish equivalence; that would require a pre-specified
equivalence or non-inferiority design on a larger held-out set.

## What each diagnostic means

| Metric | Meaning | It does not establish |
|---|---|---|
| Source-session recall | A selected structured memory cites an answer-session ID. | That the selected memory contains the answer-supporting fact. |
| Structured literal coverage | The gold appears in extracted memory fields. | That a derived answer is unsupported when it does not appear verbatim. |
| Source literal coverage | The gold appears in the immutable source turns. | That an answer is derivable when its final wording is absent. |
| End-to-end accuracy | The answerer was judged correct. | Which pipeline stage caused a wrong answer. |

For each wrong answer, `lltm eval failure-audit <variant>` writes a worksheet
that assigns exactly one primary cause: extraction loss (E1), retrieval miss (E2),
temporal resolution error (E3), context assembly loss (E4), answer reasoning failure
(E5), or judge error (E6). E1 is further labelled as number, date, duration, entity,
event, relation, or negation. `lltm eval failure-report <worksheet>` is the
roadmap input; it must precede additional ranking or temporal tuning.

## Fidelity-preserving representation experiment

New ingestion persists every raw turn and records a deterministic sentence anchor on
each extracted memory. `two_stage_hydrated` is a distinct ablation: structured
retrieval and temporal filtering select memories first, then a bounded local span of
their raw source text is added to the prompt. It must be compared with
`two_stage`, `two_stage_no_temporal`, and `two_stage_hydrated_no_temporal` on a
freshly ingested store. Older stores do not contain raw turns and cannot test this
hypothesis.

The source anchor is a retrieval aid, not an assertion that the extractor produced
an exact quotation. The prompt labels hydrated text as verbatim source evidence and
records missing anchors, skipped evidence, and hydrated tokens in each JSONL result.

## Repeats, pairing, and latency

Use `lltm eval repeat <variant> --runs 3` (up to five) before interpreting a
headline difference. `lltm eval variability <variant>` reports individual
accuracies, mean, sample standard deviation, observed spread, and a deterministic
bootstrap interval for the mean. Compare matched variants with `lltm eval
compare`; report wins/losses and exact McNemar p-values, not only percentage deltas.

The results table's p95 is observed answer-provider API latency, excluding quota
waiting. Memory-run JSONL diagnostics additionally record retrieval latency, context
assembly latency, and provider API latency. Small-sample p95 values are operational
observations, not claims that a temporal policy is faster.

## Judge audit

The v1 judge audit has 94% agreement on 50 independently labelled cases, with one
judge-lenient and two judge-strict disagreements. This supports the wording **no
directional bias was observed in the three disagreements**; it does not prove that
the judge is unbiased or that an LLM label is ground truth.

## Required next run

1. Run the temporal gate and ingest a fresh `two-stage` store, which now includes
   raw turns and source anchors.
2. Run the four structural ablations and classify every wrong answer before tuning
   retrieval weights.
3. Repeat the selected configurations three to five times on the development split.
4. Freeze the configuration, then run a disjoint held-out split once for the final
   project claim.
