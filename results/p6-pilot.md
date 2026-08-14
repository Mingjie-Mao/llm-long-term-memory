# P6 Utility and Packing Pilot

**Status:** diagnostic pilot; not a release-selection or benchmark result.

## Scope

- Dataset: the existing stratified LongMemEval-S dev store, 10 questions.
- Retrieval: frozen v1 `chronomem` store, top-20 memories per question.
- Utility labels: one full-context answer and one leave-one-out answer for every
  retrieved memory; each answer was independently judged. Leave-one-in probes
  were intentionally omitted to keep this pilot within one free-tier day.
- Generalization check: five-fold cross-validation grouped by question. No rows
  from a held-out question enter the corresponding training fold.

## Utility-signal result

| Metric | Result |
|---|---:|
| Questions / labels | 10 / 200 |
| Critical labels | 15 |
| Inert labels | 185 |
| Held-out RMSE | 0.310 |
| Mean-baseline RMSE | 0.263 |
| Predictor beats baseline | No |
| Spearman correlation with retrieval score | 0.37 |

The learned utility predictor does **not** generalize better than predicting the
mean on this pilot. It must not be used to select a P6 packing policy. A utility
budget sweep was therefore deliberately not run: applying a failed predictor to
the same calibration questions would produce an optimistic but uninformative
comparison.

## Relevance-packing baseline

| Token budget | Questions | Accuracy | Median context tokens | p95 answer API latency |
|---:|---:|---:|---:|---:|
| 250 | 10 | 20.0% | 288 | 1.1s |
| 500 | 10 | 20.0% | 477 | 0.8s |
| 1,000 | 10 | 20.0% | 478 | 0.9s |

The flat curve means this pilot supplies no evidence that allocating more than
250 tokens to the frozen v1 relevance packer improves accuracy. It does not show
that 250 tokens is generally sufficient: the sample is small, development-only,
and uses the lossy v1 representation.

## Reproduction artifacts

- Labels: `results/raw/influence-chronomem.jsonl`
- Feature metadata: `results/raw/influence-chronomem.features.json`
- Relevance sweep: `results/pack-sweep-chronomem-relevance.json`
- Pareto chart: `results/pack-sweep-chronomem-relevance.svg`

## Decision

Keep relevance packing as the P6 baseline. Do not enable or publish a utility
packer. Revisit only after a larger, pre-specified measurement with both
leave-one-out and leave-one-in probes, followed by evaluation on questions not
used to fit the predictor.
