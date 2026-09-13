# Gate protocol — what a registered threshold has to clear

> Audit correction (2026-09-12): categorical disagreement is 38, binary correctness disagreement is 27; the repeat model is not guaranteed conservative and historical source identity is unknown. See [the correction](audit/v4-gate-correction-20260912.md) before using the historical interpretation below.

Applies to every comparison registered from 2026-09-12 onward. It does not reopen any
archived result: v2, v3 and the v4.0-flat conclusion stay exactly as captured, and the
qualifications on them live in their own audit records.

## The rule

**A gate may not be registered until its claimed effect has been checked against the
instrument's measured noise floor.** The check is
`tools/check_gate_is_resolvable.py --effect <net probes> --repeats <runs per arm>`, run
*before* the pre-registration is written, with its output pasted into the registration.

A gate that cannot resolve its own effect is not a conservative gate. It is a coin flip
wearing a threshold, and it fails and passes the same configuration depending on the draw.

## The measured noise floor

`v4.0-flat` and `v4.0-flat2` are the same variant on the same store, with byte-identical
retrieval on all 142 probes — which is exactly what Gate 0 certifies when it says nothing
differs but the thing under test. Here nothing differs at all, so their disagreement is
provider sampling and nothing else.

| | |
|---|---:|
| probes compared | 142 |
| retrieval identical on | 142 |
| **verdicts that changed** | **38 (26.8%)** |
| observed net difference | −7 |

Minimum detectable effect at 2σ, in net probes:

| stratum | 1 run | 3 runs | 5 runs |
|---:|---:|---:|---:|
| 30 (`count`, `duration`) | 5.7 | 3.9 | 2.7 |
| 33 (`comparison`) | 5.9 | 4.1 | 2.8 |
| 49 (`current_state`) | 7.2 | 5.0 | 3.4 |
| 142 (all development probes) | 12.3 | 8.5 | 5.9 |

Measured: the 26.8% disagreement rate. Modelled: the spread of a net difference under the
null, as `sd = sqrt(discordant)` — the paired-sign null, which assumes only that two runs
of one configuration are exchangeable. Modelled more loosely: the repeat columns, from a
homogeneous-`p` approximation. Treat the 1-run column as solid and the repeat columns as
indicative; heterogeneity makes majority voting help more than the model predicts, not less.

## What this retires

| gate | stratum | needed | verdict |
|---|---:|---:|---|
| "at most one probe may regress" (v4.1) | 30 | 5.7 | **not resolvable** |
| v4.0-flat selected on net +9 | 142 | 12.3 | **not resolvable** |

Both are refused by the tool today. `v4.1-scan` is therefore **not** rejected by the
one-probe rule — that rule decided nothing. It is stopped on mechanism, in
[`analysis/v4-probe-diagnosis.md`](analysis/v4-probe-diagnosis.md): routed scanning
demonstrably improves evidence completeness and demonstrably does not improve answers,
because the binding constraint is a reader that under-enumerates facts it already has.

## What a gate may look like instead

Any one of these, chosen and written down **before** the run:

1. **Repeats, decided on the mean.** Three runs per arm is what v2 and v3 did, and it is
   why those results replicate. The cheapest way to buy a readable gate.
2. **A pooled stratum.** `count` alone cannot carry a 3-probe claim; `count + duration`
   pooled to 60 can carry 8. Pooling must be declared in advance, because choosing the
   favourable denominator after seeing the split is the failure v4.1's own registration
   warned about.
3. **A direction, not a threshold — but only with a resolvable magnitude.**
   **Corrected 2026-09-12 by the v4.2 run**, which this recommendation got wrong. v4.2 was
   recorded `not_promoted` because two of three non-count strata had a negative net: −1 and
   −1, both far inside the 2σ bars of 4.8 and 6.1 probes for a single run. The *sign* of a
   net is dominated by noise exactly when the net is near zero, so a bare direction rule is
   a coin flip there. The verdict happened to be right for another reason, and the trigger
   was noise. A direction is only a gate when paired with a magnitude the instrument can
   resolve, or with repeats.
4. **A deterministic check.** The strongest option, and free: an invariant verified on a
   fixture rather than a score compared across runs. `number equals the length of its own
   item list` is worth more than any single-run accuracy delta, and costs no quota.

## Two things that are not noise

Not everything needs this treatment, and treating deterministic facts as though they were
noisy would be its own error.

- **Gate 0 stays exact.** Retrieval identity is a byte comparison, not a score. One
  differing probe still voids a run.
- **Per-fact counts are evidence, not scores.** "27 of 30 count answers disagreed with
  their own item list under the control, 0 under v4" is a property of the rows, and it
  replicates across flat, flat2 and scan. That is why deterministic arithmetic is the one
  v4 change the evidence actually supports.

## Recording it

Every pre-registration from now on carries a **Resolvability** section holding the
command, its output, and the stratum the decision is read on. A registration without one
is not registered.
