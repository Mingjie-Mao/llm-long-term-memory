# v4.0 synthesis answerer — pre-registration

**Registered 2026-09-06, before any provider call for this comparison.** Nothing below
may be changed once the run starts.

## The claim being tested

Accuracy is lost after retrieval, not during it. `dev60` selected a source-containing
context for **98.3%** of questions and answered **55.0%** correctly, and the failure
taxonomy over 222 failures puts retrieval misses at **0–6%** while abstention-with-source
is **44–50%**. v4.0 changes what the answerer does with evidence it already has, and
nothing else.

If synthesis is the bottleneck, handing the arithmetic to code should move the operations
that need arithmetic and leave the one that does not roughly where it is.

## Arms

| arm | variant | differs by |
|---|---|---|
| control | `two_stage_reasoned_evidence` | — |
| candidate | `two_stage_synthesis` | `answer_policy` only |

Same store (`train150`), same config (`configs/v3-phase5-compact.yaml`), same retrieval,
same hydration, same fallback. Verified by construction: both arms build with
`adaptive_reasoning_hydration=True`.

## Data

`results/manifests/v4-probe-split.json`, **development half only — 142 probes**. The 87
held-out probes are not answered in this phase and the runner refuses to resume into
them.

Both arms are run now rather than reusing the earlier v3.3 probe reading: that reading
was taken on the archived v1 probe set, whose questions and ground truth differ. Reusing
it would compare two arms across two instruments.

## Cost, declared in advance

142 probes x 1.25 requests (measured fallback rate 24.5%) x 2 arms ≈ **355 answerer
requests**. No judge calls — grading is derived. One quota day at 500 requests.

## Gate 0 — the run is not read unless this passes

`tools/check_arm_invariant.py baseline candidate` must report `PASS`.

v4.0 changes the answerer only, so `retrieved_ids`, `evidence_found`, `evidence_needed`
and `context_complete` must be identical per probe. If they are not, the difference has
two causes and the run answers nothing. This is checked **before** any accuracy number is
looked at.

`context_tokens` is deliberately excluded: v4 renders supersession chains differently, so
the same memories may be spelled with a different number of tokens. What must not move is
which memories.

## Registered readings

Reported per operation and per stratum. **Never pooled into one number** — pooling a
retrieval ceiling with a synthesis result produces a figure that is neither.

| operation | what v4.0 changes for it |
|---|---|
| `duration` | the subtraction moves from the model to Python |
| `comparison` | the ordering moves from the model to Python |
| `count` | enumeration is made explicit, then deduplicated and counted in Python |
| `current_state` | supersession chains are rendered as chains with the current value marked |

`count` is additionally split on `context_complete`. On the incomplete stratum the
evidence a correct count needs was never retrieved, and v4.0 cannot reach it; that is
data-layer work registered as v4.1.

**There is no untouched operation to serve as a control.** All four are affected. Gate 0
is the control, and it is a stronger one: it constrains the mechanism rather than a
comparison group.

## Registered predictions

Written down so they can be wrong.

1. `duration` and `comparison` rise materially.
2. `count` rises on the `context_complete` stratum and moves little on
   `context_incomplete`.
3. `current_state` rises slightly at most — it was already the strongest operation,
   which is the observation that motivated the whole hypothesis.
4. The abstention rate falls.

**If 1 does not happen, the hypothesis is wrong** and the failure is in identifying the
operands, not in computing with them. That is a real result and it redirects v4 toward
rendering and retrieval, not toward more answerer work.

## Registered stop conditions

- **Gate 0 fails** → the run is void. No number from it is reported or used.
- **Confident-wrong rises by more than accuracy does.** v4.0's risk is converting an
  abstention into a confidently wrong computed answer: the model names two wrong dates
  and Python subtracts them exactly, producing a number that arrives with a derivation
  attached.

  Counted as: rows where `verdict == "wrong"` **and** `synthesis_computation.computed`
  is true — the code asserted a value and the value was wrong. Compared against the same
  count in the control, where every wrong answer is the model's own.

  A net accuracy gain that arrives with a larger rise in that count is **STOP**, not a
  win.

  **What guards it, and what does not.** `missing_field` now blocks computation: a
  verdict that names an absent operand and supplies operands anyway is contradicting
  itself, and the code keeps the model's prose instead of subtracting the half it filled
  in. That guard is tested. It does **not** cover the dangerous path — a model that
  supplies two *wrong* dates and leaves the field empty is indistinguishable from one
  that supplies the right two, and only grading against ground truth separates them.
  That is why this is a measured stop condition and not an asserted property.

  When this pre-registration was first written it claimed `missing_field` prevented this
  failure. It did not: the field was defined in the schema, named in the prompt, and read
  by nothing. It was wired on the same day, before any provider call for this comparison.
- **No movement in any arithmetic operation** → the hypothesis is falsified; do not
  proceed to the `dev100` gate.

## What this phase may and may not decide

**May:** whether the synthesis mechanism does what it was built to do, on development
data.

**May not:** replace v3.3 as the product. That requires the `dev100` gate, which is
decision 1 of the 3 that data protocol allows, and which must be registered separately
using `assert_gate_input_observed` so it cannot pass on a field that was never populated
— the defect that made the `dev60` `no_new_confident_errors` check vacuous.

**Is not:** accuracy. These probes derive their ground truth by SQL over stored facts and
inherit any extraction loss. No figure here may be quoted beside a LongMemEval number.
