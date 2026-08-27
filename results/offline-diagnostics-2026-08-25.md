# Three cheap fixes, measured, and all three are negative

**Run 2026-08-25, offline, zero API calls.** Three open questions were closed in one
morning using only the existing stores and the already-collected randomised pilot.
Each had been proposed — in this project's own notes — as a plausible cheap alternative
to the expensive fix. None of them survives.

Recording negative results is the point. Two of the three had already been written into
the report as a mechanism or a promising direction, and both statements were wrong.

## 1. Predicate normalization does not reunite orphaned replacement signals

**The claim being tested.** Supersession matches the exact `(user_id, subject,
predicate)` triple. 88% of `replaces_previous` signals sit alone on their key, and 96% of
those belong to a `(user, subject)` that holds other predicates. The report inferred a
mechanism from that: extraction invents a fresh predicate string per fact, so the signal
and the fact it should replace land on different keys, and normalizing predicate names
would reunite them.

**The test.** Canonicalize every predicate (snake_case, then strip a trailing plural
`s` where it is not `ss`), re-group, and count how many orphaned signals gain a target.

| | train150 | heldout100 |
|---|---:|---:|
| keys before → after canonicalization | 9,934 → 9,917 | 6,614 → 6,602 |
| keys merged | 17 | 12 |
| orphaned signals before | 777 | 546 |
| orphaned signals after | 775 | 546 |
| **reunited with a target** | **2 (0.3%)** | **0 (0.0%)** |

**The mechanism was wrong.** The singular/plural collisions are real, but they barely
touch the keys that carry replacement signals.

**What the orphans actually look like.** Sampling them shows well-formed, specific
predicates whose siblings are simply the user's other attributes:

```
signal predicate : age
content          : "The user turned 32 years old on July 15th, 2023."
siblings (48)    : art_studio, asylum_status, audiobook_app, books_read, …

signal predicate : home_city
content          : "The user moved to Tokyo 9 months ago."
siblings (48)    : age, art_studio, asylum_status, …
```

`age`, `home_city`, `rent_budget`, `job_tenure` are not misspellings of each other or of
anything else. No normalization scheme merges them, because they are different attributes.

**The corrected reading.** `replaces_previous` is being set on facts that have **no
predecessor in the store**. Two explanations remain, and this test does not separate them:

1. The earlier value was never extracted — in which case this is not an independent
   problem but a second symptom of extraction loss;
2. The extractor sets the flag on update-shaped wording ("turned", "renewed", "moved")
   regardless of whether anything earlier exists.

Distinguishing them requires reading the raw archive for an earlier statement of the same
attribute, which needs semantic judgement rather than SQL. Until that runs, the honest
statement is that **622 signals produce 42 supersessions and the reason is not predicate
naming**.

## 2. A second pass over each batch's tail recovers 22% of the loss, not most of it

**The claim being tested.** `batch-position-pilot.md` asks whether "a second pass over
the tail of each batch recovers the yield without paying 4,800 requests". Since the loss
is position-dependent, re-running only the crowded positions should be far cheaper than
batch size 1.

**The test.** Re-analysis of the already-collected reversal pilot: every session appears
at some position in one arrangement and at `14-p` in its reverse, so front and tail yields
are paired within session. 53 sessions have a batch-15 front record, a batch-15 tail
record and a batch-1 record.

| | memories / session |
|---|---:|
| batch 15, tail (positions 4–14) | 2.64 |
| batch 15, front (positions 0–3) | 5.00 |
| **batch 1** | **13.15** |

A tail pass is genuinely cheap and genuinely works — on those sessions it takes tail yield
from 140 to 273 memories (**+95%**) and recovers 3 of 3 tail zero-yields, at **1.73x** the
requests (11 of every 15 sessions re-sent) against **12x** for batch size 1.

**But the decomposition is what matters:**

```
tail → batch-1 gap            10.51 memories/session
  explained by POSITION        2.36   = 22%
  the rest is BATCH CROWDING   8.15   = 78%
```

Even at the very front of a batch of fifteen, a session yields 5.00 against 13.15 when it
is extracted alone. **Position is the small half.** A tail pass is cheap first aid, not a
substitute for a smaller batch, and describing it as an alternative to batch size 1
overstates it by roughly four times.

## 3. Fixing the recency half-life does not make recency useful

**The claim being tested.** [`retrieval-weights.md`](retrieval-weights.md) found `recency`
to be a no-op: a 30-day half-life against a corpus whose freshest memory is 932 days old
sends every score to ~1e-12. That was recorded as a *configuration defect, not a signal
defect*, with a 300–600 day half-life listed as free future work.

**The test.** Re-sweep with a corrected half-life and a non-zero weight.

| Configuration | @1 | @2 | @3 | @5 | assembled |
|---|---:|---:|---:|---:|---:|
| **baseline** (h=30, weight 0) | 84.7% | 93.3% | **95.3%** | 96.0% | **95.3%** |
| h=400, weight 0.3 | 84.7% | 92.7% | 95.3% | 96.0% | 95.3% |
| h=400, weight 1.0 | 85.3% | 92.0% | 95.3% | 95.3% | 95.3% |
| h=600, weight 0.3 | 84.7% | 92.7% | 95.3% | 96.0% | 95.3% |
| h=1200, weight 0.3 | 84.7% | 92.7% | 95.3% | 96.0% | 95.3% |

At a corrected half-life the signal is no longer degenerate — and Top-3 and assembled
recall do not move at all, while @2 drops 0.6–1.3 points across every setting.

**So the earlier note was half right.** It *was* a configuration defect. Repairing it does
not help, because the signal carries no information for this task: LongMemEval asks which
month a museum was visited and what the assistant recommended, and **the gold session is
not preferentially recent**. This closes the question on stronger grounds than before —
not "misconfigured so untested" but "configured correctly and still useless".

## What the three together settle

Each of these was a candidate route around the expensive fix. All three are closed:

- the loss is **not** mainly positional (22%);
- it is **not** a retrieval-ranking problem (all five signals measured, §6.5);
- it is **not** predicate naming (0.3% recovery).

**What is left is the extractor's batch size itself, and there is no cheap way around it.**
That is a less convenient conclusion than any of the three hypotheses, and it is better
evidence for prioritising a rebuild at a smaller batch than anything measured before it,
because it eliminates the alternatives rather than only supporting the favourite.

One by-product: because normalization does not reunite the orphaned replacement signals,
the 622-signals-to-42-supersessions gap is most plausibly **another face of extraction
loss** rather than an independent defect in the temporal layer. That is a hypothesis, not
a result; the experiment that would decide it is in §1 above.

**Reproduction.** All three are read-only over `stores/train150.db`,
`stores/heldout100.db` and `results/raw/batch-position-pilot.json`, with temporary configs
outside the repository so that no frozen artifact is touched.
