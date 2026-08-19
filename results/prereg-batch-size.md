# Pre-registration — extraction batch size on `dev100`

**Written 2026-08-19, before the held-out run.**
`results/raw/two_stage_hydrated.heldout100.jsonl` does not exist. The held-out
ingest stands at 3,517/4,544 and is quota-blocked, so no number from
`heldout100` could have informed anything below. That is checkable rather than
asserted: the file's absence is the evidence, and it is the reason this is being
written today instead of next week.

The thing being protected is not the honesty of the analysis. It is the design.
A bad held-out result invites adding arms until something moves; a good one
invites skipping the experiment as unnecessary. Both are decisions about *what to
measure* made by looking at an unrelated number, and both are avoided by fixing
the design first.

---

## The question

Extraction currently processes **15 sessions per request**, and that is not a
choice anyone defended on quality grounds — it was set by the free tier's 500
requests per day. Three results say it costs something:

| finding | where |
|---|---|
| same session, moved to the back of a batch, yields **1.6 vs 4.5** memories | [batch-position-pilot.md](batch-position-pilot.md), McNemar p < 0.0001 |
| batch 1 vs batch 15: **12.7 vs 2.7** memories/session, zero-yield **0% vs 11.7%**, 60/60 sessions | same, p = 1.7e-18 |
| **10 of 14** remaining dev50 failures are lost at extraction; **0** at retrieval | [failure-stages.md](failure-stages.md) |

And two say the fix is not free:

| finding | where |
|---|---|
| recovering 6 lost facts fixed **2 answers of 6** | [counterfactual-qa.md](counterfactual-qa.md) |
| batch 1 recovered a missing quantity and **destroyed a present one** in the same memory — "has scored 3 goals" became "has scored several goals and two assists" | same, `4adc0475` |

So the question is not "does batch 15 lose memories" — that is settled. It is:

> **Does recovering them answer more questions than it breaks, and at which batch
> size does the trade stop paying?**

---

## What varies, and what does not

Exactly one parameter moves. Everything else is pinned to
[`results/frozen/p10-final/freeze.json`](frozen/p10-final/freeze.json), and
`scripts/freeze.py` must pass on every arm except for `sessions_per_request`.

| | |
|---|---|
| **varies** | `ingest.sessions_per_request` ∈ {15, 5, 1} |
| fixed | extractor model, both stage prompts, schema, dedup threshold 0.9200 |
| fixed | retrieval: `top_k` 20, `candidate_limit` 50, semantic-only weights, rerank off |
| fixed | fallback: on, `max_turns` 3, `max_chars` 2400 |
| fixed | answerer `memory-aware-v2`, judge `lme-type-aware-v2` |
| fixed | question set: `results/manifests/dev100.json`, frozen 2026-08-18 |

**The batch-15 arm is re-run, not reused.** The held-out result cannot serve as
the baseline: different questions, so nothing pairs. A baseline that is not
paired is not a baseline.

### Arms and cost

Derived from the measured held-out rate of 0.266 requests per session at batch
15, over `dev100`'s 4,796 sessions.

| arm | requests | free-tier days | paid (3.1-flash-lite) |
|---|---:|---:|---:|
| `batch15` — baseline | ~1,280 | 3 | ~$5 |
| `batch5` | ~3,800 | 8 | ~$9 |
| `batch1` | ~9,600 | 20 | ~$19 |
| **total** | **~14,700** | **31** | **~$33** |

Three arms rather than two because two cannot find a knee. If `batch5` captures
most of what `batch1` captures, that is the configuration worth shipping and the
saving recurs on every future rebuild. With only {15, 1} measured, a positive
result says "smaller is better" and leaves the operating point unknown.

---

## Endpoints, in the order they will be read

### The power problem, stated before the data

At n = 100 this design **cannot** reach significance on accuracy under its own
prior. Exact McNemar needs 6W-0L, 8W-1L, or 10W-2L to clear p < 0.05. Projecting
the known conversion rates onto dev100's expected ~28 failures, of which ~20 are
S1:

| assumed conversion | fixed | broken | result |
|---|---:|---:|---|
| oracle A's 4/9 = 44% | 9 | 2 | p = 0.065 |
| oracle A's 4/9 = 44% | 9 | 3 | p = 0.146 |
| counterfactual's 2/6 = 33% | 7 | 2 | p = 0.180 |
| counterfactual's 2/6 = 33% | 7 | 3 | p = 0.344 |

**Not one cell clears 0.05.** So a rule of the form "adopt only if accuracy
improves significantly" would commit this project, in advance, to rejecting the
change whether or not it works. Discovering that after spending 31 quota days
would be the expensive way to learn it.

The consequence is registered here so it cannot be forgotten later: **a null
result on accuracy is this design's expected outcome and must not be reported as
evidence that batch size does not matter.** It would be evidence of n = 100.

### Primary — extraction coverage

Directly caused by the intervention, measured over 4,796 sessions rather than 100
questions, and therefore the only endpoint this design can actually resolve.
Paired within session across arms.

| metric | why |
|---|---|
| **user facts per session** | memories not beginning "The assistant" — the pilot's 12.7 headline is inflated by recommendation lists (52–62% of all memories at every batch size), so the raw count is the wrong unit |
| zero-yield rate | the tail of the same attenuation, not a separate mode |
| memories per session | reported for continuity with the pilot; not decisive |

### Secondary — fidelity

The axis that already caught `batch1` once. Coverage and fidelity move
independently, and an arm that recovers ten facts while corrupting three is not
an improvement.

| metric | how |
|---|---|
| **quantity preservation** | for `(subject, predicate)` keys present in two arms, whether a numeric object in one becomes a hedge ("several", "some", "a few", "many") in the other. Counted in both directions |
| groundedness floor | content-word overlap with source turns, as in the pilot |
| near-duplicate rate | retrieval encoder, within session |

`4adc0475`'s failure is exactly a quantity becoming a hedge, so this metric is
named by a case that already happened rather than invented for the occasion.

### Secondary — dilution

Two of the four non-conversions in the counterfactual were caused by material the
smaller batch *added*, not by anything it missed — one gold fact came back buried
in 68 memories of which 52 began "The assistant recommended".

| metric | how |
|---|---|
| median context tokens | per question, per arm |
| gold-memory rank | where the answering memory lands in the ranking, per arm |

### Confirmatory — accuracy

Paired McNemar vs `batch15` on the same 100 questions, same judge. Reported with
its exact counts, never as an accuracy delta alone, and read against the power
table above.

---

## The decision rule, committed now

Applied in this order. "Adopt" means changing `ingest.sessions_per_request` in
the shipped config and opening a new freeze record.

1. **Adopt the cheapest arm** that improves user facts per session over `batch15`
   at p < 0.05 (paired, per session) **and** does not degrade accuracy, where
   degradation means paired losses exceed paired wins.
2. If an arm improves coverage and **loses** more questions than it wins, do not
   adopt it, and publish the dilution numbers as the reason. That outcome is a
   result about ranking and context, not a failed experiment.
3. If an arm degrades quantity preservation in net, do not adopt it regardless of
   coverage, and open fidelity as separate work.
4. If no arm improves coverage — which contradicts the pilot and would mean
   something is wrong with the run — publish that and stop.
5. **The schedule is the only open variable.** Free tier or paid changes when
   this finishes, not what it measures. If the budget is not approved, the arms
   run in the order `batch15` → `batch1` → `batch5`, so that a run truncated by
   patience still yields the extreme comparison.

## Registered prediction

Scored later against what happens, whichever way it goes.

- `batch1` improves user facts per session by **2.5–4x**, p well under 0.001.
- `batch1` nets **+3 to +6 questions** over `batch15`, at **p between 0.06 and
  0.35** — a real effect this design cannot certify.
- `batch5` captures **roughly half** of `batch1`'s coverage gain at a fifth of
  its cost, and is the arm that gets adopted.
- Quantity preservation is **worse** at `batch1` than at `batch15` in at least
  one direction, on the strength of `4adc0475` being a mechanism rather than an
  accident.

## What would make this uninterpretable

- Any arm whose store fails `scripts/freeze.py` on a field other than
  `sessions_per_request`.
- Content refusals landing unevenly across arms — the held-out ingest hit 15
  refused sessions, and if one arm loses sessions the others keep, coverage is
  confounded. Refused session ids are recorded per arm and excluded from all
  paired comparisons.
- A partial ingest in any arm. The held-out gate exists because a partial store
  is indistinguishable from an extraction failure; the same gate applies here.

## What is deliberately not asked

Why larger batches attenuate. Output-budget exhaustion, enumeration drift,
input-position effects and schema-length pressure are all consistent with the
pilot and this separates none of them. The mechanism does not change which batch
size to ship, and a hypothesis about it would not survive being tested on the same
data that suggested it.

---

## Amendment 1 — memory-only accuracy is the QA endpoint, not final accuracy

**Added 2026-08-19, while the held-out evaluation was running and before any of
its rows had been read.** The result file held 1 of 100 questions at the time and
its contents were not opened. The amendment comes from a mechanism argument, not
from data.

The original draft made final accuracy the confirmatory endpoint. That is the
wrong quantity, and the reason is in this system's own design: **the archive
fallback exists to compensate for exactly the failure batch size causes.** When
extraction drops a fact, the answerer reports it cannot answer, the archive is
searched, and the question is often recovered. On dev50 that path is 18.0 of the
72.0 points.

So an extraction fix does not primarily add correct answers. It **moves them from
the expensive path to the cheap one**. Memory-only accuracy rises, the fallback
fires less, and final accuracy sees only whatever is left over. An arm could
improve extraction substantially and show a flat final accuracy, and under the
original rule that would have been read as "batch size does not matter" — the
second way this design was pre-committed to the wrong conclusion.

It also worsens the power problem the draft already identified: the effect on
final accuracy is the *residual* after the fallback absorbs part of it, so it is
strictly smaller than the effect on memory-only accuracy. Measuring the residual
and calling it the endpoint maximises the chance of a null.

### Added endpoints — no extra cost

All three are already recorded per question by the same run. Nothing new is spent.

| endpoint | why |
|---|---|
| **memory-only accuracy** | what structured memory answers without the archive. The quantity extraction actually acts on, and the one the product wants to grow |
| **fallback trigger rate** | the mechanism check. A real extraction improvement must show this falling. If coverage rises and the trigger rate does not fall, the recovered facts are not the ones being asked about |
| fallback success rate | guards the opposite failure: an arm that fires less but converts worse |

### Revised endpoint order

1. **Coverage** — user facts per session, zero-yield rate. Paired per session over
   4,796 sessions. Primary, and the only endpoint this n can resolve decisively.
2. **Memory-only accuracy** — paired McNemar vs `batch15`. The QA endpoint.
3. **Fallback trigger rate** — must fall if 1 and 2 both moved. A confirmation
   that the three numbers describe one mechanism rather than three coincidences.
4. **Final accuracy** — non-degradation guard only. Not the headline.
5. Fidelity and dilution, unchanged.

### Revised decision rule

Replaces rule 1 of the original. Rules 2–5 stand.

> Adopt the **cheapest** arm that improves user facts per session at p < 0.05 and
> does not degrade final accuracy. Among arms that additionally improve
> memory-only accuracy, prefer the one at the quality–cost knee rather than the
> maximum: an arm costing twice as much for one point of memory-only accuracy is
> not adopted. **If two arms are statistically indistinguishable on memory-only
> accuracy, the cheaper one wins** — stated now so the operating point is not
> chosen from noise after the fact.

### Revised prediction

Supersedes the accuracy line of the original prediction; the coverage and
fidelity predictions stand.

- Memory-only accuracy rises more than final accuracy in every arm, and the gap
  between the two shrinks monotonically as batch size falls.
- Fallback trigger rate falls at `batch1` relative to `batch15`.
- Final accuracy moves by **less than the +8pp oracle ceiling** — oracle A
  converted 4 of 9 dev50 questions with *perfect* injected facts, and `batch1` is
  not perfect extraction, so its final-accuracy gain is bounded well below that.
- `batch5` is still the adopted arm.
