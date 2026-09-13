# v4.2 execution and statistical addendum — 2026-09-12

Prospective addendum written before any v4.2 provider evaluation. It supersedes the
resolution interpretation and fills execution details in the original
[preregistration](prereg-v4.2-cited-enumeration.md); that original document is preserved.
The user authorised implementing, rehearsing, freezing and running this development
comparison on 2026-09-12. This authorisation does not spend the held-out probe set.

## Estimand and sampling unit

The primary estimand is candidate minus control correct-probe proportion on the fixed
30 development count probes, when each arm is aggregated by majority correctness over
three complete answers. A probe is correct iff at least two saved deterministic grader
verdicts are `correct`. Wrong, abstained and unparseable all count as incorrect; none
is dropped. This fixes what “majority” means, independently of answer wording.

There are 29 namespaces for these 30 probes. Repeat answers are correlated measurements
of an existing probe, not 90 independent questions. All probes in the same namespace
are kept together in inference. Results describe this store-derived development
instrument; they are neither a new benchmark result nor a claim about real user tasks.

## Decision and uncertainty

Retain the original minimum practical gain of **+6 net count probes (20 percentage
points)**. Promotion additionally requires a **two-sided exact namespace sign-flip
p <= 0.05**, the original non-count direction check, and observed use of the citation
mechanism on at least 72/90 candidate count answers. This last condition operationalises
the originally stated 80% prediction: if the mechanism is not observed, no positive
mechanistic attribution is made even if scores improve.

For each namespace, sum candidate-minus-control majority-correctness differences across
its probes. Enumerate the exact null distribution by independently swapping arm labels
for entire namespaces; dynamic programming counts every sign configuration without
Monte Carlo error. The test assumes independent namespaces and exchangeable arm
outcomes under the null. Small sample size and the non-random selection of development
probes limit generalisation. No claim of guaranteed 80% power is made.

Report all of: the 30-probe two-by-two paired correctness table, net change, cluster
sign-flip p, descriptive exact probe-level McNemar/binomial p, and a descriptive 95%
percentile interval from 10,000 namespace bootstrap resamples (seed 420912). The
bootstrap resamples whole namespaces and recomputes the probe-weighted difference;
it is not a second promotion test. The exact p=0.5 binomial calculation is the paired
sign special case; see the [SciPy binomial-test definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html)
and [paired permutation-test description](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).

The old 38/142 rate counted verdict categories; only 27/142 changed correctness. Neither
that pair nor its homogeneous-repeat approximation establishes this experiment's power.
No threshold is changed after provider evaluation starts. No interim performance looks,
optional sample extension or repeated testing until success is allowed.

## Enumeration completeness — a second registered reading

**Added 2026-09-12, before any provider call. No existing threshold is moved.**

The binary gate asks whether a count answer flipped from wrong to right. Simulated
against the measured pair churn, it promotes a genuine six-probe improvement about **31%**
of the time and an eight-probe one about **56%** — under the pessimistic heterogeneous
churn, 15% and 27%. So "inconclusive" is the *expected* outcome of this run even if the
candidate works, and that has to be stated before the run rather than discovered after it.
Thirty binary outcomes is a coarse ruler.

So a second reading is registered on the quantity the candidate is actually built to
change: **gold facts that were in the retrieved context and were never named.** On the
runs already paid for, that reads control 130, `v4.0-flat` 45, `v4.0-flat2` 45,
`v4.1-scan` 44.

The two same-configuration runs agreeing exactly at the total is partly luck — 11 of 30
probes move and the movements cancel. The honest figure is the paired per-probe spread:
sd 0.83, so the standard error of the 30-probe total is **4.6 facts** and the two-sigma
band is **±9**. The control-to-v4 step of 85 facts is about **19 sigma** on it.

| | |
|---|---|
| measure | in-context gold facts never named, summed over 30 count probes |
| aggregation | mean over the three repeats per probe and arm, then the paired difference |
| registered threshold | candidate at least **10 facts** below control (two sigma, rounded up) |
| direction | negative difference means the candidate named more of what it was shown |

**What it decides: nothing on its own.** It is evidence about the mechanism, not a
promotion criterion. The binary gate above remains the only promotion decision, unchanged.
A candidate that clears this reading and fails the gate has shown that it does what it
claims and not that it is worth promoting; that is a useful and reportable result, and it
is what this run is most likely to produce.

**What it inherits.** Membership is decided by token overlap against the probe's
SQL-derived gold memories, so the absolute level carries the matcher's error. The matcher
is byte-identical across arms and the reading is paired, so a *difference* survives a
biased absolute level — which is why only the difference is registered and the absolute
totals are reported as context.

Final outcomes:

- **Promote to the next development comparison:** net >= +6, exact cluster p <= .05,
  citation mechanism observed >=80%, and safety direction passes.
- **Not promoted:** net <= -6, or at least two of the three non-count strata have a
  negative candidate-minus-control correct count. This is an operational decision;
  significance is only claimed when supported by the reported test.
- **Mechanism not demonstrated:** no previous stop applies, but <72 candidate count
  answers counted from cited labels.
- **Inconclusive:** all remaining completed outcomes. A positive net below +6, or an
  unconvincing p, earns no promotion.
- **Invalid/incomplete:** missing rows, changed identities, Gate 0 failure, unobserved
  required fields, technical failure or exhausted budget. No complete efficacy verdict.

## Schedule and grading

Control: `two_stage_synthesis_flat`; candidate: `two_stage_synthesis_enumerate`.
Both use `configs/v3-phase5-compact.yaml`, the same frozen store, and the same provider
model ID/settings. Temperature is the existing 0; three calls do not imply three
independent seeds. Provider model aliases may change remotely; the snapshot cannot
freeze the provider's unpublished weights.

Round 1 includes all 142 development probes. Rounds 2 and 3 include only 30 count
probes each. Within each round shuffle deterministically with seed 420912; alternate
the first arm within adjacent probe pairs. Persist all 404 scheduled cell IDs,
including probe, type, repeat and arm, in the freeze. Never include held-out IDs.

Gate 0 requires equal ordered `retrieved_ids`, `evidence_found`, `evidence_needed`, and
`context_complete` for every observation of a probe, across arms and repeats. Scope
and question identity are also checked. Labels may change context text and token count;
retrieval cannot change. Malformed structured final answers invalidate the run. Valid
abstentions and parse failures remain scored as incorrect. Grading is deterministic;
there is no LLM judge and no extraction in this run.

## Budget and stops

- Exactly **404 answer rows**; historical planning estimate **about 541 API attempts**.
- Hard cumulative cap **600 SDK generate-content attempts**, counting retries, failures
  and fallback calls across process restarts. Persist each reservation before sending.
- Stop before the next attempt once observed input + output + thinking tokens reaches
  **2,000,000**. This is an observed-token threshold, not a hard billing cap: one in-flight
  response may cross it, and failed requests may not report tokens. Request cap is hard.
- Stop on **20 failed SDK attempts** cumulatively. At most 3 API-error attempts and
  3 transport retry-budget counts per generation via the existing client; all count
  toward the global cap. A daily 429 pauses immediately. No automatic model substitution.
- Respect the configured 10 RPM, 500 RPD, 250,000 TPM and any lower learned/provider
  limit; quota state is shared with existing project runs rather than reset by the
  frozen-store directory. Other processes may share the provider's account quota.
- On temporary network/quota interruption, retain exact completed rows and durable
  completion records. Resume only the same freeze and schedule. An unfinished row can
  replay completed calls without paying again. A crash leaving an unresolved attempt
  requires reconciliation; do not automatically resend an uncertain request.
- Stop on any source/configuration/database/index/model hash mismatch, runtime version
  change, duplicate/out-of-scope cell, identity mismatch or retrieval invariant failure.
- No efficacy stopping before 404 rows. A terminal budget failure is incomplete, not a
  negative model-quality result; increasing its budget needs a new recorded decision.

## Freeze and rehearsal acceptance

Hash the exact source files (including SQL schema and orchestration), config, original
preregistration and this addendum, probes/split, dependency lock and runtime versions.
Preserve the source tarball. Use a SQLite backup snapshot, copy the matching index and
local encoder snapshot, verify DB/index IDs, and hash every copied data/model file.
Run inference with read-only SQLite and pinned CPU encoder files. Credentials are
loaded normally at runtime and never included in the snapshot.

Before any provider request, require a complete **404-row fake-provider rehearsal**
against the same freeze, with Gate 0 passing and cited enumeration observed on all
candidate count cells. Deliberately stop after 17 rows, resume the exact prefix and
confirm no duplicate/replayed provider calls. Verify all hashes before and after.
Unit fixtures additionally exercise budget persistence, in-row completion replay,
wrong arm/repeat/freeze rejection, retrieved-order changes, and clustered inference.
The live runner requires the matching rehearsal certificate.
