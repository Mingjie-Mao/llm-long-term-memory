# Cross-encoder reranking and the accuracy–context frontier

**Measured 2026-08-14 on `results/manifests/dev31-pilot.json`.** Same store
(`two-stage-hydrated`), same answerer, same judge, six arms differing only in
`retrieval.top_k` and `retrieval.rerank.enabled`.

## Result

| top_k | rerank | Accuracy | Median ctx tokens | Source-session recall | p95 answer latency |
|---:|---|---:|---:|---:|---:|
| 20 | off | 48.4% | 439 | **93.5%** | 1.4s |
| 20 | cross-encoder | 48.4% | 420 | 90.3% | 1.3s |
| 10 | off | **51.6%** | **242** | 87.1% | 1.2s |
| 10 | cross-encoder | 51.6% | 240 | 83.9% | 1.0s |
| 5 | off | 48.4% | 146 | 87.1% | 1.2s |
| 5 | cross-encoder | 48.4% | 143 | 83.9% | 1.3s |

Paired exact McNemar:

```
k=20 off   vs k=10 off      1W-0L of  1 disagreement    p = 1.000
k=10 off   vs k=10 rerank   0 disagreements — identical on every question
k=10 off   vs k=5  off      0W-1L of  1 disagreement    p = 1.000
naive_rag  vs k=10 off      6W-6L of 12 disagreements   p = 1.000
```

## The reranker does not earn its place

**At k=10 the two arms answered every one of the 31 questions identically.** That is
stronger than "no detectable difference": there is nothing to detect. Reranking
changed the ordering, and the answer never moved.

What it did change is recall. At every k, the reranked arm has **lower**
source-session recall than the plain one — 93.5% → 90.3% at k=20, 87.1% → 83.9% at
both k=10 and k=5. The staged recall instrumentation was added to answer exactly one
question: *is the reranker rescuing evidence or discarding it?* The answer here is
that it discards.

That is not a claim that cross-encoders are useless for retrieval in general. It is
a measured statement about this store: the hybrid ranking already places the
answer-bearing memory inside the top-k, so a reranker has no buried evidence to
rescue and can only reorder — occasionally pushing a correct source out of the cut.

**Decision: `retrieval.rerank.enabled` stays `false`.** It costs a model download, a
torch dependency in the serving path, and 3.2 points of recall, in exchange for
zero measured answer changes. The implementation and `configs/rerank.yaml` stay in
the repo so the result is reproducible rather than merely asserted.

This repeats the P6 pattern: a plausible mechanism was built, measured, and not
shipped. The utility predictor lost to predicting the mean; the reranker loses to
not reranking.

## Context can be cut hard for free

Accuracy across k = 20, 10, 5 is 48.4% / 51.6% / 48.4%, and every pairwise
comparison turns on a **single question**. At n=31 the k setting is not
distinguishable — but the context cost is, and it falls by **67%** from k=20 to k=5
(439 → 146 median tokens).

Against the retrieval baseline on the same questions:

| | Accuracy | Median ctx tokens | Reduction |
|---|---:|---:|---:|
| `naive_rag` | 51.6% | 13,057 | — |
| `two_stage` k=10 | 51.6% | 242 | **54x** |
| `two_stage` k=5 | 48.4% | 146 | **89x** |

`naive_rag` vs k=10 is 6W-6L, p = 1.000: no detectable accuracy difference at a
fiftieth of the context. That is the compression claim, and it is the strongest
result the project has.

## What this does not establish

- **k=10 is not shown to be optimal.** The 3.2-point spread across k values rests on
  one question each way. Choosing 10 over 5 or 20 needs the full 50, and probably
  repeated runs given the measured ±8-point noise floor.
- **"No detectable difference" is not "equivalent."** With 31 questions and 1–12
  disagreements, these tests have little power. They rule out large effects, not
  small ones.
- **Recall differences are not accuracy differences.** The reranked arms lose source
  recall without losing accuracy here, which means the lost sources were not
  answer-bearing *on these questions*. On a larger set they might be.
- Not a table row. The dev31 set is an ingest-order prefix, not a stratified sample.

## Reproduction

```bash
for k in 20 10 5; do
  for ce in --rerank --no-rerank; do
    lltm eval run two_stage --questions results/manifests/dev31-pilot.json \
      --store-name two-stage-hydrated --top-k $k $ce --label "k${k}_..."
  done
done
```

Artifacts: `results/raw/two_stage.k{20,10,5}_{ce,plain}.jsonl`. Labelled files are
excluded from the default results table on purpose — a sweep is a diagnostic, not a
published row.
