# v4 data protocol — pre-registration

**Status: registered, not yet executed.** Written before any v4 code exists, so the
boundaries are chosen without knowing which of them a result will want moved.

## The constraint that shapes everything below

LongMemEval-S is **exhausted**. Its 500 questions are partitioned across five disjoint
sets with **zero remaining**:

| set | n | exposure so far |
|---|---:|---|
| `dev50` | 50 | burned — read in full, prompts fitted on it |
| `heldout100` | 100 | spent as the v1 final; rows readable, never tuned on since |
| `train150` | 150 | v2/v3 development, readable without limit — except `dev60` inside it |
| `dev100` | 100 | v2 validation; **aggregate reads only**, one arm decision taken |
| `test100` | 100 | v2 final; one shot, spent |

`longmemeval_oracle.json` carries the **same 500 question ids** — verified, not
assumed — so it is a different haystack over identical questions, not new data.

There is therefore no way to cut a `v4-hidden` set from what is on disk. Anything
described as a fresh final test has to come from outside this dataset, and saying so
now is cheaper than discovering it after v4 is built.

## Three tiers, and what each may decide

### Tier 1 — Synthesis probes · a new instrument, not a benchmark

The failure the whole of v4 targets is arithmetic and enumeration over evidence
already in context. Measuring that on LongMemEval questions gives **two to four
samples per operation**, which is where the current diagnosis ran out of resolution.

So Tier 1 generates probes whose ground truth is a **SQL query over the stored
facts**, never a label and never a model's opinion. Feasibility is measured, not
hoped for — on the `train150` store today:

| probe kind | available | ground truth |
|---|---:|---|
| count / enumerate | 354 groups of 3-13 members | `COUNT` over one `set`-arity relation in a date window |
| duration | 150 namespaces | subtraction of two `event_time` values |
| comparison / ordering | 150 namespaces | comparison of two `event_time` values |
| current state | 71 real supersession chains | the `active` end of the chain |

**What Tier 1 may decide:** whether a change to the answerer improves the operation it
targets, and by how much, at a sample size that can tell.

**What Tier 1 may never do:** stand in for a benchmark score. It measures synthesis
*conditional on the store being right*. Where extraction was wrong, the probe's
"ground truth" inherits that error, so a Tier 1 number is not accuracy and must never
be reported beside a LongMemEval figure or in the README results table.

Probes are regenerated from a seed and a store fingerprint rather than stored as a
fixed file, so a probe set cannot be quietly curated after seeing a result.

### Tier 2 — `v4-dev` · the decision set

Nothing unused exists, so this is a choice between contaminated options and the
honest thing is to name the contamination.

**Registered choice: `dev100`.** Its only prior exposure is aggregate-level, and the
one decision taken on it — selecting a retrieval arm — is orthogonal to v4.0, which
changes the answerer and leaves retrieval untouched. `train150` is rejected for this
role precisely because it is readable: ninety of its questions have been read
question-by-question during v3 development, which is the widest possible channel.

Two limits are registered with it, because a validation set spends validity per
decision and this one has already spent some:

- **At most three decisions may be taken on `dev100` for the whole of v4.** Each is
  registered in advance with its gate, as `dev60` was.
- **Aggregate metrics and pre-declared slices only.** No per-question reads, ever.
  This is the rule `dev60` now lives under and the reason `dev100` still has value.

If v4 needs a fourth decision, that is the signal to acquire new data, not to relax
this cap.

### Tier 3 — `v4-hidden` · must be acquired

**No `v4-hidden` set exists and none can be created from this dataset.** This tier is
registered as a dependency, not a plan: a genuinely unseen final test requires data
from another source, and which source is a decision with real cost that has not been
taken. Candidates worth evaluating — none verified here, and each needs its format,
licence and ingestion cost checked before being adopted:

- a conversational-memory benchmark from outside LongMemEval
- a fresh LongMemEval-style corpus built from new conversations
- a held-back slice of real user data, if the product ever has any

Until one exists, **v4 has no final test**, and every v4 number is a development
number. Reporting one as a final result would be the single worst thing this protocol
can prevent, so it is stated here rather than in a footnote.

## What the existing sets may still be used for

| set | v4 role |
|---|---|
| `train150` minus `dev60` (90 questions) | error analysis and iteration, readable without limit |
| `heldout100` | error analysis only — spent, so reading costs nothing; never a score |
| `dev60` | **nothing.** Aggregate already reported; not read again, not tuned on |
| `dev100` | `v4-dev`, at most three registered decisions, aggregate-only |
| `test100` | **nothing.** Spent |

## The single-variable rule

v4.0 changes the **answerer only**. Retrieval, extraction, storage, the predicate
vocabulary and the fallback stay exactly as `v3.3-compact` left them, and the
comparison is `v3.3 retrieval + v3.3 answerer` against `v3.3 retrieval + v4.0
answerer`.

This is registered because the project has already paid for breaking it: v3.2 moved
accuracy and context together and could only be reported as `STOP`, because nothing
in the result said which change did what.

The controlled predicate vocabulary — already proposed offline in
[`results/analysis/predicate-vocabulary.md`](analysis/predicate-vocabulary.md) — is
therefore **not** part of v4.0. It is registered as v4.1, to be measured against a
v4.0 baseline, so that the schema's contribution is separable from the answerer's.

## Registered order

| phase | changes | measured on | may decide |
|---|---|---|---|
| v4.0 | answerer synthesis only | Tier 1 probes, then `train150`-minus-`dev60` | whether synthesis is the bottleneck |
| v4.0 gate | none | `dev100` — decision 1 of 3 | whether v4.0 replaces v3.3 |
| v4.1 | controlled relation vocabulary + set-valued enumeration | Tier 1 probes, then `train150`-minus-`dev60` | whether the schema adds on top of v4.0 |
| v4.1 gate | none | `dev100` — decision 2 of 3 | whether v4.1 replaces v4.0 |
| multi-hop probe | none | 20-30 hand-written probes, train data only | whether a graph is needed at all |
| final | frozen | `v4-hidden`, **once it exists** | the reported result |
