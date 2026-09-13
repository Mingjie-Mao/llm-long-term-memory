# Failure taxonomy — 222 failures, 63 distinct questions

**Zero provider calls. No sealed set opened.** Every row here was already on disk:
`heldout100` is spent, so reading it costs nothing that was not already spent, and
`pilot48`/`tune42` are `train150` slices the data protocol allows to be read without
limit. `dev60` and `test100` were not touched.

This exists because the v3.3 diagnosis that was shaping the roadmap rested on **11
failures from one run**, split across four buckets — two to four samples each. That
is not enough to choose what to build next, and the project's own data protocol says
so: *"No single-run significance on small slices."*

Classifier: [`tools/failure_taxonomy.py`](../tools/failure_taxonomy.py). Rule-based,
not another model call, and validated against the eleven v3.3 failures that were read
by hand first — `--validate` reports **10/10** agreement. Two of those hand labels
were corrected during validation; the reason is recorded in the source, because
"the reference moved towards the rule" is how a validation gate gets quietly defeated.

## Results

Intervals are computed on **distinct questions, not rows**. Repeats and arms make the
same question fail several times and those failures are not independent draws;
treating them as independent would shrink every interval by roughly the repeat count.
Row counts stay visible because they say how *reproducible* each failure is.

| bucket | heldout100 (v1) n=30 | pilot48 (v3) n=17 | tune42 (v3) n=16 |
|---|---|---|---|
| `abstained_with_source` | **50.0%** [33.2, 66.8] | **47.1%** [26.2, 69.0] | **43.8%** [23.1, 66.8] |
| `count_mismatch` | 36.7% [21.9, 54.5] | 29.4% [13.3, 53.1] | 25.0% [10.2, 49.5] |
| `wrong_specific_value` | 10.0% [3.5, 25.6] | 17.6% [6.2, 41.0] | 31.2% [14.2, 55.6] |
| `retrieval_miss` | 3.3% [0.6, 16.7] | 5.9% [1.0, 27.0] | 0% [0, 19.4] |

v1 and v3 are never merged: `heldout100` came from a different extractor generation,
and folding its extraction defects into v3's totals would attribute the wrong cause.
That they agree anyway is the point — the shape is stable across two extractor
generations and three independent question sets.

## What this establishes

**Retrieval is not the bottleneck, and this no longer rests on one run.** Between 0%
and 6% of failures never reached the labelled conversation, in every pool including
v1. Any proposal whose mechanism is "find more candidates" is aimed at under a
sixteenth of the problem.

**Abstention with the source in hand is the largest bucket in all three pools**, at
44–50%, with heavily overlapping intervals. On roughly half of all failures the
system holds the right conversation and answers "I do not know".

**Counting is second**, at 25–37%, also stable. Together the two account for 69–87%
of failures.

## The abstention bucket is not what it looked like

The obvious reading — the model never saw raw text — is wrong. Every abstention
failure in both v3 pools had **already been through the fallback**:

| | pilot48 | tune42 |
|---|---:|---:|
| abstention failures | 37 | 18 |
| of which reached `archive_wide` or `source_local` | **37** | **18** |
| saw neither hydration nor fallback | **0** | **0** |

So raw turns were retrieved and shown, and the answer was still "I do not know".

The cap is the thing worth looking at. `fallback.max_turns` is **3**, and it binds on
every single call — successes and failures alike show exactly three turns:

| | abstention failures | correct answers that used fallback |
|---|---:|---:|
| pilot48 | 3 turns × 37 | 3 turns × 58 |
| tune42 | 3 turns × 18 | 3 turns × 37 |

Three turns is enough when BM25 ranks the needed turn in its top three, and nothing
at all when it does not. The namespace holds hundreds of turns.

**This yields a hypothesis that costs nothing to test:** for the abstention failures,
does the needed turn appear at rank 4–10? That probe is pure local FTS5 over a store
already on disk — no provider call, no sealed data — and it separates two very
different fixes: *raise the cap* (cheap, config only) against *the answerer is too
conservative* (prompt work). It should be run before either is attempted.

## What this does **not** establish

- 63 distinct questions is better than 11, not large. Every interval above is 20–40
  points wide, so these buckets are ordered with confidence but not sized precisely.
- The classifier assigns one label per row by a fixed precedence: evidence
  completeness is diagnosed before answer behaviour. A row that both lacked half its
  sources and then abstained is filed under `partial_evidence`, on the grounds that a
  system holding half the evidence has already failed regardless of what it says next.
- `partial_evidence` is only detectable in runs that record
  `all_source_sessions_recalled`, which is v3.2 onwards. It can only be under-counted.

---

# A2 · Fallback depth probe — the hypothesis was wrong

The section above proposed that the abstention bucket was a truncation artefact:
`fallback.max_turns` is 3, it binds on every call, and the needed turn might be
sitting at rank 4-15 inside a pool the code already ranks but never shows.

**It is not.** Probe: [`tools/fallback_depth_probe.py`](../tools/fallback_depth_probe.py),
zero provider calls, BM25 replayed over the `train150` store.

| where the gold session's turn ranked | pilot48 (9 q) | tune42 (9 q) |
|---|---:|---:|
| already **shown** (rank 1-3) | 9 — 100% | 8 — 89% |
| in the pool but **cut** (rank 4-15) | 0 | 1 — 11% |
| beyond the pool, or absent | 0 | 0 |

Seventeen of eighteen abstentions were already looking at a turn from the right
conversation. Raising `max_turns` would reach **one question**. That fix is dead, and
it was cheap to kill: no provider call, no sealed data, one afternoon of a wrong idea
caught before it became a config change and an eval run.

## What is actually happening

Session-level presence is not the same as the answer being visible, so the probe also
checks whether the gold answer's own distinctive words appear in the text the model
received:

| | pilot48 | tune42 | total |
|---|---:|---:|---:|
| gold words **all present** in the shown turns | 1 | 2 | **3** |
| gold words partly present | 2 | 4 | 6 |
| gold words absent | 1 | 1 | 2 |
| gold has no matchable words — a derived answer | 5 | 1 | 6 |
| cut by `max_turns` | 0 | 1 | 1 |

Two things fall out.

**At least three abstentions had the answer's own words in front of them.** The
system held the text, the text contained the answer, and it said it did not know.
Nothing about retrieval can fix that.

**A third of abstentions are on answers that must be derived, not looked up.** The
six "no matchable words" rows have gold answers like `5`, `3`, `18 days` — a count or
a duration that appears nowhere verbatim because it has to be computed from what is
shown. Abstention there is the model declining to do arithmetic over evidence it has.

That is the same failure as the `count_mismatch` bucket seen from the other side:
one refuses to compute, the other computes wrongly. Together they are 69-87% of all
failures, and they share a cause — **synthesis over evidence already in context**,
not the finding of evidence.

## Limits of this probe

- **Eighteen questions.** Nine per pool. Every percentage here is one or two
  questions wide, so it is evidence about direction, not size.
- **The query is an approximation.** The real call passes
  `verdict.source_query or instance.question`, and `source_query` is not recorded in
  any row. The probe uses the question text. That biases *against* the cut
  hypothesis being falsely rejected, not for it: if the model's keywords rank better
  than the raw question, the true ranks are at least this good.
- **Word overlap is a crude proxy** for "the answer was visible". It cannot see
  paraphrase, and it says nothing at all for derived answers — which is why those are
  reported as their own row rather than folded into a hit rate.
