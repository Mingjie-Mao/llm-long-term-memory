# v4.1 relation routing and exhaustive scan — pre-registration (draft)

**Status: structure registered 2026-09-06. The mechanism is now built and runnable as
variant `two_stage_synthesis_scan`; the thresholds are still blank.**

Wired and verified offline on the development count probes, with no provider call:
**evidence completeness 57% → 73%**, 16 routed and 14 abstained — matching the earlier
oracle-armed measurement. That is retrieval completeness, not accuracy.

Three fields below are marked `[FILL FROM v4.0]`. They are the numbers v4.1 must be
compared against, and they do not exist yet. Registering the *structure* now is the
point: a gate designed after its result is known is not a gate, and v4.1's headline
mechanism has already been measured offline, which is exactly when the temptation to
shape the threshold around it is strongest.

## What has already been measured, for free

Nothing below needs a provider call to establish, and all of it is on record before the
gate is written.

| | measured | source |
|---|---|---|
| Routing upper bound, after merging `owns` into `acquired` | **100%** (30/30, development half) | [`relation-routing.md`](analysis/relation-routing.md) |
| Routed share of real aggregation questions at margin 0.15 | **25%** (18/72) | same |
| Scan completeness on the routed slice | **100%** against top-k's 69% | [`routed-scan-gain.json`](analysis/routed-scan-gain.json) |
| Completeness on the abstained slice | **29%**, unchanged by construction | same |
| Overall evidence completeness | 50.0% → **66.7%** | same |

## The finding that shapes the gate

**The router abstains precisely where top-k is worst.** It claims questions already 69%
complete and declines the ones at 29%; after the scan, all remaining incompleteness sits
in the abstained slice — ten of fourteen probes.

So v4.1 must be gated on the **routed slice**, and the gate must say so in advance.
Reporting a pooled number would dilute a ceiling with questions the mechanism never
claimed; reporting only the routed slice *without* having registered that in advance
would be choosing the favourable denominator after seeing it. Both are registered here.

## Arms

| arm | variant | differs by |
|---|---|---|
| control | whichever of `two_stage_synthesis` / `two_stage_reasoned_evidence` v4.0 selected | — |
| candidate | control + relation routing + exhaustive scan on routed questions | retrieval for routed questions only |

**v4.1 changes retrieval, so Gate 0 from v4.0 inverts.** Retrieval must be *identical on
abstained questions* and may differ only where `Route.routed` is true. A divergence on an
abstained question means the router leaked into questions it declined, and voids the run.
`tools/check_arm_invariant.py` is extended for this before v4.1 runs, not after.

## Registered readings

Per operation, and on `count` split three ways rather than two:

| stratum | meaning |
|---|---|
| routed · context complete under top-k | the scan had nothing to add |
| **routed · context incomplete under top-k** | the slice the mechanism exists for |
| abstained | unchanged by construction; reported so the pooled figure can be reconstructed by anyone who wants it |

## Registered gate

All must hold:

**Baselines locked 2026-09-06 from `results/archive/v4.0-flat/`, before any v4.1
provider call.** They are the measured `two_stage_synthesis_flat` figures, not targets
chosen to be reachable.

| quantity | v4.0-flat | n |
|---|---:|---:|
| `count` · context incomplete | **7.7%** | 13 |
| `count` · context complete | **23.5%** | 17 |
| `duration` | **70.0%** | 30 |
| `comparison` | **57.6%** | 33 |
| `current_state` | **85.7%** | 49 |
| median context tokens | **516** | 142 |

1. On the **routed** slice, `count` accuracy must exceed **7.7%** — the v4.0-flat figure
   on `context_incomplete`, which is the gap the scan claims to close. A scan that
   completes the evidence and does not move this has not helped.
2. On the **abstained** slice, `count` accuracy must not fall below **7.7%**. The router
   is supposed to leave these alone; movement here means it did not.
3. `duration` ≥ **70.0%**, `comparison` ≥ **57.6%**, `current_state` ≥ **85.7%**, each
   within one probe of those figures. v4.1 touches retrieval, and retrieval feeds every
   operation — a rise in `count` bought by a fall elsewhere is not a gain.
4. Median context does not exceed **1,032 tokens**, twice v4.0-flat's 516. The offline
   scan was *cheaper* per relation (68.5 tokens against 439), so a rise beyond this means
   the union of `("acquired", "owns")` is pulling in more than the measurement suggested.
5. Confident-wrong does not rise faster than accuracy, counted as in the v4.0
   registration. A complete set of the wrong relation is the specific way this
   mechanism fails, and completeness is what makes such an answer look trustworthy.
6. Every gate input passes `assert_gate_input_observed`. A check whose input is pinned
   at its default cannot fail, which is how `dev60`'s `no_new_confident_errors` passed
   without testing anything.


## A ceiling measured before the run, which lowers what this arm may claim

Registered 2026-09-06 from `results/archive/v4.0-flat/`, before any v4.1 provider call.

The scan closes one gap: evidence that was never retrieved. It cannot touch a second one,
and that second one is larger than assumed when this file was first written.

Of the 30 development count probes, **17 already had complete evidence** under top-k —
the scan has nothing to add to them. On those 17, v4.0-flat scored:

| | n |
|---|---:|
| correct | 4 |
| **under-enumerated** — listed fewer members than were present | **7** |
| over-enumerated | 3 |
| no number stated | 3 |

    count_0016   gold 6   the model listed 2
    count_0030   gold 3   the model listed 1
    count_0039   gold 5   the model listed 3

**With every fact in front of it, the model still failed to enumerate on 10 of 17.** That
is a reading failure, not a retrieval one, and no amount of scanning reaches it.

### What this changes

**Gate 1 is unchanged.** It is written on the routed slice, whose members are exactly the
probes whose evidence was incomplete, and the scan's claim there stands.

**The registered predictions are revised downward, before the run:**

- Prediction 1 stands: `count` rises on **routed · incomplete**.
- Prediction 2 is replaced. The earlier version expected the pooled `count` figure to
  rise by roughly a quarter of the routed gain, on the assumption that complete evidence
  would be counted correctly. It is not: on complete evidence the model enumerates
  correctly 4 times in 17. **The pooled `count` figure may therefore barely move even if
  Gate 1 passes comfortably**, and that is not a failure of the scan.
- A new prediction: on the 17 already-complete probes, v4.1 changes nothing, because
  nothing about their retrieval changes. If it does, retrieval leaked into questions the
  router declined and Gate 0 should have caught it.

### What it means for the work after v4.1

Even a perfect scan leaves enumeration as the binding constraint on counting. Whether the
next lever is a prompt that forces exhaustive listing, an enumeration step outside the
model, or something else is out of scope here — but this file should not be read later as
having promised that scanning would fix counting.

## Registered predictions

1. `count` rises materially on **routed · incomplete**, and barely moves elsewhere.
2. The pooled `count` figure rises by roughly a quarter of that, because only ~53% of
   development count probes route.
3. No other operation moves.
4. Median context on routed questions **falls**.

If 3 fails, retrieval changed for questions the router declined, and the mechanism is not
what the arm is testing.

## Stop conditions

- Gate 0 (inverted) fails → void.
- `count` does not move on the routed slice → the offline ceiling does not survive
  contact with a real answerer, and the remaining work is coverage, not scanning.
- Coverage, not routing, is confirmed as the binding constraint → v4.1 is capped at the
  routed share regardless of how well it performs, and that cap is reported next to the
  result rather than discovered later.

## Explicitly out of scope

**Growing the relation vocabulary past 15 set relations.** The routing probe showed real
questions asking about things the vocabulary does not name — "how many babies were born",
"how many shirts did I pack" — and routed them to the nearest thing that exists. That is a
data-layer decision with its own cost, and folding it into v4.1 would make two changes at
once. It is registered as a separate decision, not as part of this one.

## Preconditions before this may run

- [x] v4.0 measured, and the baselines written in (2026-09-06)
- [ ] v4.0's `dev100` gate taken, or explicitly deferred
- [ ] `check_arm_invariant.py` extended for the inverted Gate 0
- [ ] Dry-run rehearsal with a fake provider, on the development half only
- [x] Mechanism built and verified offline: completeness 57% → 73%, 16 routed / 14 abstained
