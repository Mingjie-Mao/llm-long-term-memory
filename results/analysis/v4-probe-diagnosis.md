# v4 probe diagnosis — what the instrument can resolve, and where counts fail

> Audit correction (2026-09-12): categorical disagreement is 38, binary correctness disagreement is 27; the repeat model is not guaranteed conservative and historical source identity is unknown. See [the correction](../audit/v4-gate-correction-20260912.md) before using the historical interpretation below.

Produced by `tools/diagnose_v4_probes.py`. **Zero provider calls**: every number below is
arithmetic over rows that were already paid for. Machine-readable record:
[`v4-probe-diagnosis.json`](v4-probe-diagnosis.json).

Probe ground truth is derived by SQL over stored facts and inherits extraction loss.
These are instrument readings and must never be quoted beside a LongMemEval figure.

## The finding that reorders everything else

`v4.0-flat` and `v4.0-flat2` are **the same variant on the same store**, and their
retrieval is byte-identical on all 142 probes — which is precisely what Gate 0 tests when
it certifies that nothing differs but the thing under test. Here nothing differs at all.
Every verdict that disagrees between them is the provider's own sampling.

| | overall | net vs control | `current_state` |
|---|---:|---:|---:|
| control `v3.3-dev` | 78/142 · 54.9% | — | 43/49 · 87.8% |
| `v4.0-flat` (registered, archived) | 87/142 · 61.3% | **+9** (22W-13L) | 42/49 · 85.7% |
| `v4.0-flat2` (same configuration, rerun) | 80/142 · 56.3% | **+2** (23W-21L) | 36/49 · 73.5% |
| `v4.1-scan` | 81/142 · 57.0% | **+3** (23W-20L) | 37/49 · 75.5% |
| **flat vs flat2 — pure sampling** | **−5.0 points** | **net −7** (10W-17L) | **−12.2 points** |

**38 of 142 verdicts changed between two runs of one configuration.** Per stratum the
swing was `count` +10.0, `duration` −3.3, `comparison` −9.1, `current_state` −12.2 points.

Three consequences, none of which require another run to establish:

1. **The archived `v4.0-flat` conclusion is a single draw.** Its headline, 54.9% → 61.3%
   at 22W-13L, is a net +9. The replication of the same configuration gives net +2, and
   the two runs differ from each other by net −7. The candidate's measured effect is
   smaller than the disagreement between two runs of that candidate. The conclusion is
   hash-bound and is not rewritten; this file is the qualification, in the same way
   `results/audit/v3-verdict-schema-never-sent-20260906.json` qualifies dev60.
2. **The "at most one probe regression" rule is unreadable on this instrument.** Judged
   by it, `v4.0-flat2` — byte-identical in configuration to the registered control it is
   compared against — regresses 1 probe on `duration` and 6 on `current_state` and fails.
   A gate finer than the noise floor rejects and promotes the same configuration
   depending on the draw. `v4.1-scan` must not be rejected *by that rule*; it should be
   stopped for the reason in the next section.
3. **The timeline-rendering attribution survives, but not at its stated size.** Attempt 1
   scored `current_state` 55.1% with rendering against flat's 85.7% without — a claimed
   −30.6. Same-configuration noise on that stratum is 12.2 points. The direction holds at
   roughly 2.5x the noise; the magnitude does not, and "caused the −32.7 regression"
   overstates what one pair of runs can say.

## Where count answers actually fail

| run | correct | reading failure | retrieval gap | abstained | number ≠ own item list | under/over | gold facts in context, never named |
|---|---:|---:|---:|---:|---:|---:|---:|
| control `v3.3` | 5/30 | 12 | 11 | 2 | **27** | 18/5 | 130 |
| `v4.0-flat` | 5/30 | 11 | 10 | 4 | 1 | 15/6 | 45 |
| `v4.0-flat2` | 8/30 | 10 | 9 | 3 | 0 | 15/4 | 45 |
| `v4.1-scan` | 8/30 | 13 | 7 | 2 | 0 | 16/4 | 44 |

**Deterministic arithmetic worked, and it is the one v4 change the evidence supports.**
The control's stated number disagreed with its own item list on **27 of 30** count probes.
Under v4 that is 0. Handing the arithmetic to Python did exactly what it was built to do.

**And it bought almost nothing, because arithmetic was never the binding constraint.**
Count went 5/30 → 8/30, which is inside the +10.0-point `count` noise measured above. The
number is now reliably the length of the list the model named; the list is the wrong list.

**The failure is item selection, and it is overwhelmingly under-enumeration.** Of the
wrong numbers in every run, ~80% are too low (16 under against 4 over in `v4.1-scan`).
In the best run **44 gold facts sat in the retrieved context and were never named**. The
model stops early on evidence it already has.

**Scanning closed the retrieval gap and the reading failure absorbed the gain.** From
flat2 to scan, count probes with incomplete evidence fell 9 → 7 and gold facts absent
from context fell 17 → 10 — the mechanism did what `routed-scan-gain.json` predicted.
Reading failures rose 10 → 13 over the same probes, and the count score did not move
(8/30 both ways). This is the `count enumeration ceiling` measurement reappearing as a
run-level result: more complete evidence cannot help a reader that under-enumerates.

## Both unconcluded runs, closed

- **`v4.0-flat2`** — **replication, not a candidate.** Its value is the noise floor above.
  It does not reproduce `v4.0-flat`'s headline and it supersedes nothing: `v4.0-flat`
  stays the registered, archived record of what was run.
- **`v4.1-scan`** — **stop, on mechanism rather than on the regression rule.** Routed
  scanning is confirmed to improve evidence completeness and confirmed not to improve
  answers. Separately, **`scan_route` is null on all 142 rows of every run, including the
  scan run itself**, so v4.1's Gate 0 — retrieval identical on *abstained* questions —
  cannot be evaluated on this evidence at all. The rows postdate the fix that records
  routing. Any future claim about routed retrieval needs a run whose rows carry it.

Neither run has a source-bound identity record: `run_identity()` is computed by
`tools/run_synthesis_probes.py` but was not persisted for these runs. The claim that flat
and flat2 share a configuration rests on identical variant, identical store fingerprint
and identical retrieval on 142/142 probes — strong, but reconstructed, and labelled as
reconstructed rather than written up as recorded evidence.

## What this says about the next candidate

Do not spend a run on retrieval, on scanning, or on arithmetic. All three are either
measured as working or measured as not binding. The open problem is that the answerer
names 6 of 9 facts that are in front of it and then confidently reports 6.

Two things have to change before the next paid comparison is worth running:

1. **A candidate that targets enumeration completeness directly**, verifiable on a
   deterministic fixture before any provider call — under-count, over-count, subject
   mixing, missing dates — in the shape the existing `tests/test_synthesis_answer.py`
   fixtures already use.
2. **A gate the instrument can actually resolve.** Either repeats per arm with the
   decision on the mean (the v2/v3 protocol, which is why those results replicate), or a
   stratum large enough that the effect being claimed exceeds the noise floor measured
   here. A single-run, single-probe-margin gate is not a gate.
