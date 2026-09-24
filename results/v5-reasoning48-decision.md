# v5.0 reasoning-48 decision

> **DEVELOPMENT RESULT.** These 48 questions have been read in full, three times now.
> Nothing here is a benchmark estimate and it does not belong on a version curve with
> the frozen 72% on `test100`.

## Decision

**STOP both arms. Keep the mechanism, default off, and record this as a negative
result — with one finding that is worth more than the result.**

## Evidence

| | baseline `v2c` | B `v5_fixed` (3 windows) | C `v5_planned` (2–6 by kind) |
|---|---|---|---|
| Correct | 37/48 | 37/48 | 36/48 |
| Paired | — | 2W / 2L, net **+0** | 3W / 4L, net **−1** |
| Median context | 634 | 878 | 986 |
| Mechanism fired | — | 48/48 | 48/48 |

| Registered gate | B | C |
|---|---|---|
| `net_positive` | **FAIL** | **FAIL** |
| `no_reader_attributed_regression` | PASS | PASS |
| `mechanism_fired` | PASS | PASS |
| `registered_sentences_arrived` | **FAIL** | PASS |
| `median_context_at_most_2000` | PASS | PASS |

The pre-registration's rule between the arms — "prefer B unless C wins strictly more
questions; a larger budget that buys nothing is a cost, not a tie" — selects B. C spends
108 more context tokens at the median and wins one fewer.

## The finding: delivering the evidence is not enough

This arm existed for three failures that the offline gate attributed to `local_raw` — a
sentence the extractor dropped, sitting in the raw turns of a session retrieval had
already selected. Arm C put **all three** in front of the reader.

| question | sentence | arrived | v2c | v5 C |
|---|---|---|---|---|
| `gpt4_7a0daae1` | "I just received my new tennis racket today" | yes | wrong | **wrong** |
| `a4996e51` | "up to 50 hours per week" | yes | wrong | correct |
| `58ef2f1c` | "back on Valentine's Day" | yes | wrong | **wrong** |

**Three delivered, one converted.** And in arm B, where the 50-hours sentence did *not*
arrive because three windows cannot reach it, that question was answered correctly
anyway — so even the one conversion is not attributable to the sentence.

This falsifies the premise the arm was built on. `results/analysis/v5-offline-gate.md`
asked *where the answer is* and found it reachable for 3 of 11 failures. It could not
ask whether putting it there helps, and the answer is mostly no. A reachability gate is
not a usefulness gate, and treating the first as evidence for the second is the mistake
this run cost a day of quota to find.

## The second finding: ±2 on this set is noise

Three arms, with mechanisms that share nothing — v2e re-labels dates, B attaches a fixed
raw budget, C varies the budget by question kind:

| question | lost in |
|---|---|
| `6e984301` | **3 of 3 arms** |
| `982b5123` | **3 of 3 arms** |

Two questions lost by every mechanism tried, including one that only changes how a date
is punctuated, are unstable questions rather than questions any of these mechanisms
broke. The same holds for the wins: `e48988bc` is won by two arms and is the one failure
the v2d retrieval gate established as a genuine retrieval miss — no arm here touches
retrieval, so those wins are variance.

**A net of ±2 on 48 questions is below this set's resolution.** Both of the last two
experiments were sized to detect something they could not have detected.

## Provenance

- Provider instability throughout. Arm B stopped twice on `503 UNAVAILABLE` (at
  questions 9 and 42 across attempts) and was resumed from its own rows; arm C needed
  two resumes. No question was answered twice — the harness resumes from what it wrote.
- Arm B: 130 requests, 24 failures (answerer 2/60, judge 22/70). Arm C: 162 requests,
  56 failures (answerer 2/60, judge 54/102). Every question ended with exactly one
  successful judgement, but both arms were measured while the judge was failing about
  a third to a half of its calls.
- Store: `stores/v5-reasoning48.db`, a copy of the v2c store, migrated but **not**
  backfilled — so its memories are identical to v2c's and the only difference between
  the arms and the baseline is when the raw turns arrive. The committed
  `stores/v2c-reasoning48.db` still has no `observed_at` column, which is the proof it
  was never opened for writing.

## What must not happen next

No re-tuning of the window budget, the ranking, or the rendering on these 48 questions.
They are exhausted three times over. Any repaired candidate needs a new label and a
different question set.

More importantly: **do not register another mechanism against this set expecting to
detect a net of one or two questions.** Two runs have now spent quota on comparisons
whose resolution was smaller than their noise. The next mechanism experiment needs
either a set enriched for the failure it targets, repeated runs per arm, or both — and
the cost of that should be counted before the mechanism is built, not after.

## What is kept

- `MemoryRunner.parallel_raw_windows` / `parallel_raw_planned`, off by default.
- `configs/v5.yaml`, the two variants, and `tools/v5_gate.py`.
- The offline gate, with its limitation now written down: it locates evidence, it does
  not predict whether supplying it changes an answer.
