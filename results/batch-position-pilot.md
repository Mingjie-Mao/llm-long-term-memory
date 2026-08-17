# Batched extraction loses memories, and the loss is not mostly zero-yield

**Status: complete.** Both experiments finished; 420 records, no duplicate keys,
all three batch-size arms over one shared cohort of 60 sessions.

The follow-up that decides what to do about any of this is
[the counterfactual](counterfactual-qa.md): recovering the lost facts fixed two
answers out of six, and two of the four non-conversions were caused by material
the smaller batch added. Read that before drawing a remedy from the numbers here.

Run by `scripts/batch_position_pilot.py`, read by `scripts/batch_position_analyse.py`.
Raw records in `results/raw/batch-position-pilot.json`.

## Why this ran

The clean P10 store left 14.5% of substantive sessions with no memory at all. The
standing explanation, recorded in `zero_yield_sessions`, was extractor variance:
zero-yield sessions had the same median turn count and the same assistant share as
normal ones, so nothing about the conversation predicted it.

Nothing about the conversation did. Something about the *request* did — and the
audit that established "indistinguishable" had not looked at where a session sat
in its extraction batch.

Observationally, over 2,135 substantive extraction attempts:

| batch position | zero-yield |
|---|---:|
| 0-3 | 6.4% |
| 4-14 | 18.3% |

Risk ratio 2.86x. A within-batch permutation of the position labels — holding
batch composition, each batch's total yield, namespace and request-level randomness
completely fixed — puts it at **p = 0.00005** (N = 20,000), with a batch-clustered
bootstrap 95% CI of **[2.15, 4.35]** on the ratio.

That is observational: every session sat at exactly one position, so it is still a
comparison between different sessions. This pilot randomises what was never
randomised.

> **A note on the denominator.** `zero_yield_sessions()` counts distinct sessions
> (2,096); this counts extraction *attempts* (2,135). The 39-attempt difference is
> the sessions that appear in more than one namespace and are deliberately
> extracted once per namespace. Position belongs to an attempt, so the attempt is
> the unit here. Keying yield on `source_session_id` alone — as a first pass here
> did — conflates the two and under-counts zero-yield by 11.

## Design

60 sessions: 30 that yielded nothing in production and 30 that yielded normally,
matched on turn count (±2) and assistant share (±0.08). Both groups are carried
because the question is not only whether a failure can be rescued but whether a
success can be broken.

Model, prompts, schema and decoding are identical throughout. The extractor is
driven directly rather than through `IngestionPipeline`, so batch composition and
position are set by the script and **no store is written**.

* **Experiment 1 — position.** Batch size fixed at 15, four arrangements: two
  random orderings each paired with its own reverse, so a session at position `p`
  in one sits at `14-p` in the other. The comparison is within session.
* **Experiment 2 — batch size.** The same 60 sessions at 15, 5 and 1 per request.

## 1. Position causes the loss

| position band | n | zero-yield |
|---|---:|---:|
| front 0-3 | 64 | 12.5% |
| mid 4-10 | 112 | 27.7% |
| back 11-14 | 64 | 25.0% |

Paired within session, 51 sessions seen both front and back:

| | |
|---|---:|
| yielded in front, **zero** in back | **8** |
| zero in front, yielded in back | **0** |
| exact McNemar | **p = 0.0078** |

**The coverage result is the smaller half of the finding.** The same pairing on
memory *count*:

| | |
|---|---:|
| memories per session, front | **4.5** |
| memories per session, back | **1.6** |
| more in front / more in back | 35 / 3 |
| exact McNemar | **p < 0.0001** |

Same conversation, same batch size, same prompt, same neighbours. Moved to the back
of the request, it yields a third as much. Zero-yield is the tail of a continuous
attenuation, not a separate failure mode.

## 2. Batch size causes the loss

| batch size | zero-yield | memories / session | corpus extraction requests |
|---|---:|---:|---:|
| 15 (production) | 11.7% | 2.7 | 400 |
| 5 | 8.3% | 4.8 | 1,000 |
| 1 | **0.0%** | **12.7** | 4,800 |

Paired across sizes on all 60 sessions:

| | more | fewer | exact McNemar |
|---|---:|---:|---|
| batch 5 vs batch 15 | 39 | 10 | p = 3.8e-05 |
| batch 1 vs batch 15 | **60** | **0** | p = 1.7e-18 |
| batch 1 vs batch 5 | **60** | **0** | p = 1.7e-18 |

Not one session in sixty yielded more under a larger batch, twice over.

Split by production history:

| | was zero in production | was fine in production |
|---|---:|---:|
| batch 15 | 23.3% zero, 2.1 mem/session | 0.0% zero, 3.3 mem/session |
| batch 5 | 10.0% zero, 3.6 | 6.7% zero, 5.9 |
| batch 1 | 0.0% zero, 11.2 | 0.0% zero, 14.2 |

This is the cleanest control in the pilot. At batch 15 the historical labels
reproduce — the sessions that failed in production fail again at 23.3% while the
ones that worked fail at 0% — so those two groups really are different, and
zero-yield is not purely positional. But the gap closes at batch 5 and disappears
at batch 1. Whatever makes those sessions harder to extract only becomes fatal in a
crowded request.

## 3. The extra memories are not obviously junk

A free quality floor: content-word overlap with the session's own turns, and
near-duplicate pairs within a session measured with the retrieval encoder.

| batch size | memories | overlap with source | near-duplicate pairs |
|---|---:|---:|---:|
| 15 | 163 | 86.7% | 0.0% |
| 5 | 285 | 87.0% | 0.2% |
| 1 | 762 | 86.1% | 0.1% |

Groundedness is flat and duplication is nil at every size, so the extra memories
at batch 1 are not fabrications and not restatements of each other. That is all it
says. The counterfactual later found that 69.5% of them begin "The assistant
recommended" — grounded, distinct, and mostly not asked about — which is exactly
the failure mode these two numbers cannot see.

**This is a floor, not a verdict.** Overlap catches a memory that invents a proper
noun and misses one that recombines real words into a false claim, and neither
number says the extra memories are worth *keeping*. An LLM-judged fidelity pass is
the honest next step, and it costs quota.

## What this does to the planned remedy

The experiment was going to be a retry pilot: re-run the 304 zero-yield sessions
and see how many recover. That design would have produced a confident wrong answer.
Pulling those sessions out and re-batching them moves most of them to early
positions, so they would have recovered at a high rate, and the conclusion would
have been "stochastic dropout, add a retry" when the mechanism was "we moved them".

Worse, retry-on-zero-yield treats the wrong thing. The loss is continuous: a
session that should yield ten memories and yields two never triggers a retry and is
never revisited. On these numbers that is most of the loss.

## Mechanism: hypothesis only

Larger batched extraction shows **position-dependent omission**. Why is not
established. Output-budget exhaustion, enumeration drift across a long list,
input-position effects, and schema-length pressure are all consistent with what is
measured here, and this pilot separates none of them.

## Open

* LLM-judged fidelity and unsupported-fact rate on the extra memories. The free
  floor cannot see the problem the counterfactual found: 69.5% of what batch 1
  adds begins "The assistant recommended", which is grounded, non-duplicated, and
  mostly not asked about.
* Latency and token cost per strategy, alongside the request counts above.
* Whether a smarter batching scheme — small batches, or a second pass over the tail
  of each batch — recovers the yield without paying 4,800 requests.

**Nothing in the product has been changed on the strength of this.**
`sessions_per_request` is still 15. The decision waits on the completed table.
