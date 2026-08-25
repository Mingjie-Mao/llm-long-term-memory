# Four of the five retrieval signals were never on, and turning them on makes it worse

**Run 2026-08-25 on `train150`, offline, no API calls.** `scripts/session_recall.py`
against `configs/v2.yaml` with only `retrieval.weights` varied; 150 questions, the
production `top_k` of 20, seven configurations. Raw artifacts:
`results/analysis/weight-sweep-*.json`.

## Why this was worth running

The README described five weighted signals — semantic, BM25, recency, importance,
entity overlap — and then noted in Limitations that the shipped config weights only
`semantic`. That left "the other four are unmeasured, not rejected" standing for weeks.
It is the only lever in this project that could be tested for **zero quota**, and it sat
behind an experiment costing six quota days. So it was run first.

## Result

| weights | Top-3 (mean) | assembled | median ctx tokens | vs baseline |
|---|---:|---:|---:|---:|
| **semantic 1.0 only (shipped)** | **95.3%** | **95.3%** | 140.5 | — |
| + recency 0.3 | 95.3% | 95.3% | 140.5 | **bit-identical** |
| + importance 0.3 | 90.0% | 90.0% | 134.0 | -5.3pp |
| + bm25 0.5 | 89.3% | 89.3% | 138.5 | -6.0pp |
| + bm25 1.0 | 86.7% | 86.7% | 136.0 | -8.6pp |
| all five (1.0 / 0.5 / 0.3 / 0.3 / 0.5) | 87.3% | 87.3% | 151.0 | -8.0pp |
| + entity 0.5 | 78.0% | 78.0% | 153.0 | **-17.3pp** |

Every added signal degrades ranking. **Keep `semantic` alone.**

## The recency row is not a tie, it is a dead signal

`+ recency 0.3` reproduces the baseline bit for bit at every k. That is not robustness;
the signal is identically zero across the corpus:

```
memory age    min 932 days    median 1,189    max 1,727
S_recency     min 4.7e-18     median 1.2e-12  max 4.5e-10
memories with S_recency > 0.01 : 0 of 18,519
```

`recency_halflife_days` is **30**, and the freshest memory in LongMemEval-S is **932
days** old. With a 30-day half-life that is `2^-31`. Every memory collapses to zero, so
any weight on this signal is a no-op.

**This is a configuration defect, not a signal defect**, and it means `recency` has
never participated in any retrieval this project has ever run. No published result is
wrong because of it — the weight was 0 anyway — but anyone who later sets that weight
non-zero and expects an effect would get a silent no-op. A half-life in the 300-600 day
range is what this corpus would need. Retesting it is free and is listed as future work.

## Why the others hurt

Two mechanisms, neither exotic:

* **The normalization is relative.** BM25 is min-max scaled across *this query's*
  candidates, so the best of a uniformly poor lexical set still receives 1.0. Adding
  that at weight 0.5 lets a weak signal reshuffle a ranking that was already correct.
* **`entity` is deliberately coarse.** It is `max` over the memory's entities of the
  query-token overlap ratio, so a single fully-matched entity scores 1.0 regardless of
  whether it is the entity the question is about. At weight 0.5 that is close to noise,
  and it costs 17 points.

## What this does and does not establish

It **closes** the open question "would weighting the other signals help?" at these
magnitudes, on this proxy, on this store. It does **not** establish that a tuned hybrid
is impossible: no grid over weight magnitudes was run, no interaction search, and the
endpoint is source-session recall rather than accuracy. What it does establish is that
the four unused signals are not free upside sitting on the table, which is what the
Limitations section implied while they stayed untested.

The README line changes from "unmeasured, not rejected" to measured and negative.
