# v4.2 cited enumeration — pre-registration

**Registered 2026-09-12, before any provider call on this candidate.** The mechanism is
built and rehearsed offline; no paid run has happened. Nothing below may be edited after
the run starts.

Probe ground truth is derived by SQL over stored facts and inherits extraction loss.
Every number here is an instrument reading and must never be quoted beside a LongMemEval
accuracy figure.

## The finding this candidate answers

From [`analysis/v4-probe-diagnosis.md`](analysis/v4-probe-diagnosis.md), on rows already
paid for:

- Deterministic arithmetic **worked**. The control's stated number disagreed with its own
  item list on **27 of 30** count probes; under every v4 arm that is **0**.
- It bought **almost nothing**. Count went 5/30 → 8/30, inside the measured noise.
- Because the list, not the number, is wrong. In the best run **44 gold facts sat in the
  retrieved context and were never named**, and **16 of 20** wrong counts were too low.
- Scanning closed the retrieval gap (incomplete evidence 9 → 7) and the count score did
  not move. More complete evidence cannot help a reader that under-enumerates.

## The one change

Every memory in the context carries a label (`[M1]`, `[M2]`, …). For `count`, the model
cites the labels of the members it finds instead of writing free-text phrases, and the
code resolves those citations against the labels the context actually carried.

Labelling the context and citing labels are **one mechanism**, not two bundled changes:
labels nobody cites are decoration, and citations need something to cite. This is stated
because bundling independent changes inside the answerer is the v4.0 design fault, and
the claim that this is a single variable is the thing a reader should check first.

Nothing else moves. Retrieval, hydration, packing, the fallback and every non-`count`
operation are the control's.

| arm | variant | differs by |
|---|---|---|
| control | `two_stage_synthesis_flat` | — |
| candidate | `two_stage_synthesis_enumerate` | count members cited by label |

## Resolvability

Required by [`gate-protocol.md`](gate-protocol.md). Run before this document was written:

```
$ python tools/check_gate_is_resolvable.py --effect 6 --stratum 30 --repeats 3
measured noise floor — v4.0-flat vs v4.0-flat2, the same configuration
  probes compared          142
  retrieval identical on   142
  verdicts that changed    38  (26.8% per probe)
  observed net difference  -7

judging a gate that claims to detect +6 net probes
  on 3 run(s) per arm
  stratum   30: needs   3.9  OK

OK: every stratum can resolve the claimed effect.
```

At one run per arm the bar on 30 probes is 5.7 and a +6 claim would sit on top of it.
Three repeats is what makes this decidable, and is the reason for the cost below.

## Gate

**Primary, and the only promotion decision.** `count`, 30 probes, 3 repeats per arm,
majority vote per probe.

| | |
|---|---|
| promote | candidate ≥ control **+6** net probes |
| stop | candidate ≤ control **−6** net probes |
| inconclusive | anything between, which is the honest reading of a 30-probe stratum |

**Gate 0, exact and not subject to the noise floor.** Retrieval must be byte-identical on
all 142 probes: `retrieved_ids`, `evidence_found`, `evidence_needed`, `context_complete`.
One differing probe voids the run. `tools/check_arm_invariant.py`, already **PASS** on the
rehearsal.

**Safety scan, direction only.** The other 112 probes, one run per arm. A single run
cannot resolve a small regression, so this is registered as a direction and not a
threshold: the candidate must not lose on a majority of the three non-`count` strata. It
exists to catch labels harming operations they were not meant to touch, which is the risk
of changing the context for every question.

## Predictions, written before the run

1. **`labels_uncited` will be greater than zero on most count probes.** The instrument is
   new; this says the mechanism records what it was built to record, not that it helps.
2. **`counted_from` will be `cited_labels` on at least 80% of count rows.** Below that the
   model is ignoring the field and the comparison measures a prompt change, not citation.
3. **Count accuracy will improve, and by less than the 44 uncited facts suggest.** The
   ceiling measurement stands: on 17 count probes with complete evidence the model was
   already wrong on 10 with everything in front of it. Citation addresses attention, not
   reading. A +6 gate is set at the edge of what is resolvable, not at what is hoped for.
4. **Non-count strata will not move in a consistent direction.** If they do, the labels
   are doing something to the context beyond making members citable, and prediction 4
   failing is more informative than the primary gate passing.

## Cost

| | |
|---|---|
| rows | (30 count × 3 repeats + 112 others) × 2 arms = **404** |
| requests at the measured 1.34/row | **≈ 541** |
| quota days at 500 answerer requests/day | **≈ 1.1** |
| held-out probes touched | **0** |

The held-out half stays sealed. The global one-shot ledger and the output lock are
unchanged, and `--half all` remains rehearsal-only.

## Rehearsal, completed

- Candidate, 12 development probes, `--dry-run`, zero provider calls: rows carry
  `counted_from=cited_labels`, `labels_in_context=20`, `labels_uncited=17`,
  `members_cited=3`. The citation path is exercised, not the free-text fallback.
- Control, same 12 probes, `--dry-run`.
- `check_arm_invariant.py` on the pair: **PASS**, retrieval identical on all 12.

The dry-run's canned reply was fixed as part of this: it matched `answer_policy` exactly
against `"synthesis_v4"`, so the new arm would have rehearsed the base verdict shape and
passed while the paid run exercised nothing. That is the same defect the block's own
comment was written to prevent, one policy later.

## What this cannot conclude

- Not a benchmark result, and not comparable to any LongMemEval number.
- Not a held-out result. Promotion here means the candidate earns a place in the next
  frozen comparison, nothing more.
- A single-run safety scan cannot clear the candidate of small regressions on the other
  operations. It can only catch a large one.
