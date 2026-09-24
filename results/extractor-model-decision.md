# Extractor model decision

> **DEVELOPMENT RESULT** on the extraction ruler. Not a QA result and not a benchmark
> number.

## Decision

**STOP: the extractor model stays `gemini-3.1-flash-lite`.**

`gemini-3.5-flash-lite` retains more of the user's specifics, and it does so by writing
forty per cent more memories for a ten per cent relative gain. Three of the five
registered gates fail.

## Evidence

| arm | model | overall | memories | per session | specifics kept | kept per memory |
|---|---|---:|---:|---:|---:|---:|
| B — baseline | `gemini-3.1-flash-lite` | 64.9% | 223 | 3.72 | 87 | **0.390** |
| R — replicate | `gemini-3.1-flash-lite` | 64.9% | 225 | 3.75 | 87 | 0.387 |
| C1 — candidate | `gemini-3.5-flash-lite` | **71.6%** | 313 | 5.22 | 96 | **0.307** |

Paired over the cohort's 134 specifics, C1 against B: **17 gained, 8 lost, net +9**,
exact McNemar p = 0.108.

| Registered gate | Result |
|---|---|
| 1. gain ≥ twice arm R's deviation | **degenerate — see below** |
| 2. at least +10 percentage points | **FAIL** — +6.7 |
| 3. at least 10 net specifics, gains exceeding losses | **FAIL** — net +9, and 8 lost |
| 4. the gain is not volume | **FAIL** |
| 5. frozen, sealed and archived artifacts unmodified | PASS |

## Gate 4 is the one that decides it

    memories        223 -> 313      +40%
    fidelity        64.9% -> 71.6%  +10% relative
    kept per memory 0.390 -> 0.307  -21%

C1 writes forty per cent more and returns ten per cent more retention. The information
density of a memory falls by a fifth.

This is the curve `results/batch-size-result.md` already recorded rather than a
departure from it: the citation-grounded arm wrote 4.33 memories per session for 64.2%,
the ungrounded one 13.10 for 78.4%. Retention rises with volume almost mechanically.
C1 sits on that curve.

The recall ruler could not say whether the ninety extra memories were true. A precision
ruler was built afterwards to ask, and the answer is below — it does not change the
decision, and it does change the reason.

## Gate 1 degenerated, and was not quietly rewritten

Arm R measured the instrument's own noise by re-running the baseline configuration
from an empty cache — 168 calls, all 60 sessions freshly extracted:

    134 specifics.  Gained 0.  Lost 0.  Flips 0.
    overall 0.6493 against 0.6493.  223 memories against 225.

The deviation is zero, so "at least twice the deviation" becomes "greater than zero",
which is far weaker than intended. The gate is recorded as degenerate and the decision
rests on gates 2 and 3. It is **not** restated as a pass because the arithmetic happens
to allow it.

One replicate bounds the noise; it does not establish that it is exactly zero.

## What arm R is worth on its own

It is the first repeat measurement of the extraction ruler in this repository, and it
changes how the project should choose what to work on:

| instrument | disagreement at fixed configuration |
|---|---|
| **extraction fidelity** | **0 of 134 specifics** |
| QA answerer | 6 of 100 questions; 11.4% on `temporal-reasoning` |

The extraction side is an order of magnitude the better instrument. That is why the
specificity repair resolved +14.5pp at 21 gains and 0 losses, while v2e and v5.0 moved
inside ±2 on 48 questions and could not be read. Those mechanisms were not worse; they
were measured with a blunter ruler.

Two extra memories appeared in arm R without changing a single specific's status, so
the extractor is not bit-identical — only its measured output is.

## Provenance

- Amendment 1 withdrew arm C2 (`gemini-3.5-flash`) before any arm was scored, because
  the provider returned `503 UNAVAILABLE` on most requests all day and the original
  design no longer fit the daily cap. No claim is made about that model.
- `gemini-3.1-flash-lite`: 173 of 500 daily calls, of which roughly 160 were spent on
  failed retries. `gemini-3.5-flash-lite`: 122 of 500.
- The archived baseline extraction (`results/raw/extraction-batch-size.extractions.json`)
  is intact and was the source of arm B. Arm R necessarily overwrote the CLI cache slot
  it shares with B, because the fidelity cache is keyed on prompt, model and batch — a
  replicate collides with what it replicates by construction. Run a replicate under a
  separate `LLTM_STORE_DIR` rather than deleting the cache file.

## Addendum — the extra memories are supported

`ingest/precision.py` was written after this decision was taken, to answer the question
the recall ruler is blind to: of what the extractor wrote, how much is in the
conversation it came from. Same facet detector, same normalisation, opposite direction.

| arm | recall | memories | checkable | with an unsupported specific | unsupported rate | specific support |
|---|---:|---:|---:|---:|---:|---:|
| B | 64.9% | 223 | 153 | 12 | 7.8% | 96.2% |
| R | 64.9% | 225 | 152 | 10 | 6.6% | 96.7% |
| C1 | 71.6% | 313 | 217 | 15 | 6.9% | 96.4% |

**C1's ninety extra memories did not come with extra invention.** Its unsupported rate
is 6.9% against the baseline's 7.8%, and its specific support 96.4% against 96.2% —
both marginally *better*, and both inside the spread the replicate shows at a fixed
configuration (7.8% against 6.6%). There is no detectable precision difference.

So the reason for STOP narrows and sharpens. It is **not** that the candidate invents
things: gates 2 and 3 fail on their own arithmetic — +6.7pp against a registered 10,
and a net of +9 specifics that also lost 8. What C1 does is write more memories, each
carrying fewer of the measured specifics, all supported at the same rate. That is finer
granularity or more redundancy, not hallucination, and it is not worth changing the
extractor for on this evidence.

**The precision ruler's own limits, from its first real use.** Four of its five example
flags are artifacts rather than invention: `'2023,'` and `'104,'` fail only on a
trailing comma the shared facet detector keeps, and `'page turners facebook'` and
`'author stuart turton'` are multi-word proper nouns the source states differently. So
roughly 7% unsupported is an upper bound dominated by matching artifacts, and the useful
reading is the *difference between arms*, not the level.

The trailing-comma defect is **not** fixed, deliberately: `_extract_facets` is shared
with the recall ruler, and changing it would silently move 37.3%, 64.9%, 44.8% and
59.3% — every fidelity figure this project has published — without any of them being
re-measured.
