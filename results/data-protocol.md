# Data protocol — what each set is allowed to tell you

**Frozen 2026-08-20.** Five sets, disjoint, covering all 500 LongMemEval-S
questions. Verified: every pairwise overlap is zero and the union is 500.

| set | n | role | may individual failures be read? |
|---|---:|---|---|
| `dev50` | 50 | v1 development — **burned** | historical only; not evidence for anything |
| `heldout100` | 100 | v1 final test — **spent**, 70.0% | spent, so reading costs nothing more |
| `train150` | 150 | **v2 development** | **yes, without limit** |
| `dev100` | 100 | **v2 validation** | **no** — aggregate metrics only |
| `test100` | 100 | **v2 final test** | **no** — not before the single run |

`dev100` keeps its filename. It was frozen on 2026-08-18 and the
[batch-size pre-registration](prereg-batch-size.md) names it, and renaming a
frozen artifact is the thing freezing exists to prevent. Its name says
development and its role is validation; this table is the authority, not the name.

## The rule that matters more than the names

    train150     read anything — question ids, raw turns, retrieved memories,
                 the model's answer, the gold answer. Diagnose. Change the system.

    dev100       run the whole experiment. Read aggregate metrics and
    (validation) pre-declared slices, nothing else. Decide which arm survives.
                 Never change the system because one question failed.

    test100      run after v2 is frozen. One reported result.
                 Repeats are allowed for exactly one purpose — estimating
                 run-to-run variability — and may never change the system.

The asymmetry is the point. A validation set that gets read question by question
becomes a second training set within days, which is not a hypothetical here.

## Why `dev50` lasted two weeks

There was no training set, and that is not a saving. In ordinary supervised
learning the training set absorbs the fitting and validation is touched by a
handful of model-selection decisions — a narrow channel. Here **the fitting is
prompt engineering**, and it was done by reading `dev50`'s failures in full: the
extraction prompt was rewritten from them, the fallback design was chosen from
them, gates and thresholds were tuned on them, and
[six modules were cancelled](frozen/p10-final/README.md) on them. Fifty questions
were doing training and validation duty simultaneously, through the widest
possible channel.

`train150` is the largest set for that reason. Error analysis is what consumes
data fastest in a retrieval or memory system, and `train150` is what gets
consumed.

## Repeats are part of the protocol now

`heldout100` scored 70/100 and 71/100 on identical configurations with three
questions flipping ([heldout-variance.md](heldout-variance.md)). So:

**Report accuracy as a mean over repeated runs with its spread, not as a single
number.** For a full set of 100+, one run is an acceptable headline if the spread
has been characterised once; for anything smaller, it is not.

**No single-run significance on small slices.** The fallback's `p = 0.022` came
from 9W-1L on 50 questions. One flipped question makes it 8W-2L, `p = 0.109`. At
the measured flip rate — 3 in 100 — a 50-question paired comparison expects about
1.5 flips, which is enough to move that result across the threshold on its own.
Any paired claim on 10-50 questions needs three runs per arm and an agreement
rate reported beside it.

This applies retroactively. Every paired result in this repository predates the
protocol and is a single run.

## What has already been spent, and what follows from it

The five `knowledge-update` failures analysed on 2026-08-20 were read on
`heldout100`, which was already spent — so reading them cost nothing that was not
already gone. **But any fix derived from them must be validated on `train150` and
`dev100`, never by re-running `heldout100`.** `heldout100`'s 70.0% is a v1
historical result and does not get restated with a v2 system.

`test100` is the last unseen data this project will ever have. When it is spent
there is no more.

## That has now happened

`test100` was spent on 2026-09-02 and `dev60` — the last sealed slice inside
`train150` — on 2026-09-05. The five sets are disjoint and their union is all 500
questions, so **zero unused questions remain**. `longmemeval_oracle.json` carries the
same 500 question ids and is a different haystack over identical questions, not a
reserve.

What each set may still be used for, and the fact that a fresh final test now requires
data from outside this dataset, is registered in
[`prereg-v4-data-protocol.md`](prereg-v4-data-protocol.md). Until such a set exists,
**every v4 number is a development number.**
