# v2b-gate16 with the specificity repair — decision

Date: 2026-09-23 (Australia/Sydney). Registered in `results/prereg-v2b-gate16-repair.md`
before the first provider request. **DEVELOPMENT extraction evidence**; no question was
answered and nothing here bears on 72% or the 86% / 72% gap.

## Verdict: PASS — all eight gates

"Batch 8 + specificity repair" (`configs/v2b-batch8-repair.yaml`) is the ingestion
configuration for the next store build. Existing configs, stores and results are
unchanged; this restates no v2 or v2b figure.

| gate | registered | measured | |
|---|---|---|---|
| 1 repair contribution | ≥ +10 pp | 47.5% → 60.5%, **+13.0 pp** | pass |
| 2 end to end | ≥ +10 pp, ≤ 22 lost | **+13.0 pp, +304 gained / 0 lost** | pass |
| 3 extraction drift | within ±3 pp | **0.0 pp, 0 flips** | pass |
| 4 support | ≥ 91.7%, unsupported ≤ 16.2% | 93.0%, 13.4% | pass |
| 5 kept per memory | ≥ 0.363 | 0.429 | pass |
| 6 cost | ≤ 50% sessions, ≤ 1.0 memory/session | 44.6% (348 / 780), 0.35 recovered | pass |
| 7 tenant isolation | see prereg | 244 repaired memories, 0 mis-scoped, 0 ids in two namespaces | pass |
| 8 nothing else touched | see prereg | baseline store + index hashes unchanged; frozen / sealed clean | pass |

## Measurements

`tools/store_fidelity.py`, manifest `results/manifests/v2b-gate16.json`, 780 namespaced
sessions, 2,338 user-stated specifics. No model calls.

| reading | memories | recall | kept / memory | support | unsupported memories |
|---|---:|---:|---:|---:|---:|
| baseline `stores/v2b-gate16.db` | 3,055 | 47.5% (1,110) | 0.363 | 92.7% | 284 / 2,003 (14.2%) |
| candidate, repair excluded | 3,055 | 47.5% (1,110) | 0.363 | 92.7% | 284 / 2,003 (14.2%) |
| candidate, whole store | 3,299 | **60.5% (1,414)** | 0.429 | 93.0% | 297 / 2,224 (13.4%) |

Artifacts: `results/analysis/store-fidelity.v2b-gate16{,-repair,-repair.extracted}.{json,md}`.

**Drift is zero because the extraction was identical, not because it was replayed.** All
621 extractor calls in this run were real (`results/raw/v2b-gate16-repair.ingest.usage.json`:
3.30M input tokens), and the 3,055 extractor memories match the baseline store id for id.
Ids are a digest of tenant, session and content, so the second extraction returned
byte-identical content for every memory. This is the arm-R finding (0 of 134) at 780
sessions: at batch 8 this extractor is deterministic on identical input. It also means
gate 2's zero losses are a consequence of zero drift, not an independent check.

**The repaired memories are better supported than the extractor's.** The 221 checkable
repaired memories carry an unsupported specific in 13 cases (5.9%), against 14.2% for the
extractor's own memories — the span-anchoring requirement doing what it was built for.

### Per facet (descriptive; no test applied)

| facet | baseline | candidate | gained / stated |
|---|---:|---:|---:|
| date | 17.6% | 64.7% | 8 / 17 |
| relative_time | 24.0% | 59.6% | 65 / 183 |
| duration | 46.1% | 69.7% | 42 / 178 |
| quantity | 48.8% | 62.8% | 119 / 852 |
| proper_noun | 50.0% | 56.9% | 70 / 1,011 |
| money | 61.9% | 61.9% | **0 / 97** |

The date row rests on 17 specifics. Money gaining nothing is unexplained and is not
investigated here; it is the obvious next question for this mechanism.

### Against the pilot

The pilot measured +14.5 pp (44.8% → 59.3%, 21 gained, 0 lost) on 60 holdout sessions
in its own loop. Inside the pipeline on 780 sessions: +13.0 pp from 47.5%, 304 gained,
0 lost. The trigger rate (45% against the pilot's 42%) and recovered memories per session
(0.35 against 0.33) also reproduced.

## Cost

| | this run | baseline store |
|---|---:|---:|
| extraction requests | 214 | 212 |
| repair requests | 348 | — |
| adjudication requests | 42 | 20 |
| total calls (usage file) | 621, 12 failed and retried | — |

Two quota days (2026-09-22 Pacific day from 12:22 to 12:58 Sydney, then the 09-23 day).
No `503 UNAVAILABLE` in the run. The repair roughly tripled ingestion calls on this set
(212 → 604 counted); the memo's 1.0-quota-day estimate held.

Recovered 272 repaired memories, wrote 244: deduplication removed the rest. Adjudication
doubled because the extra memories bring more near-duplicate pairs to the adjudicator.

## Found outside the gates: one false supersession, caused by the event-time split

The candidate store has 7 superseded memories against the baseline's 6. The other six are
identical in both. The extra one:

- superseded: *"The user has been using Adidas Ultraboost 22 shoes for daily runs since
  February 10, 2023."* (`running_shoes = Adidas Ultraboost 22`, no stated event time)
- by: *"The user replaced their Nike Air Zoom Pegasus 38 shoes around February 10, 2023."*
  (`running_shoes = Nike Air Zoom Pegasus 38`, stated event time 2023-02-10)

It is **wrong** — Adidas is the current state — and it is **not the repair**: both are
extractor memories, and they are byte-identical in the baseline store. What differs is
the event-time split (`observed_at` / stated `event_time`), which the baseline predates:

- baseline: both carried the session date, 2023-01-14, as `event_time`, so neither was
  later and nothing was superseded;
- candidate: the Nike fact states a date and gets 2023-02-10; the Adidas fact's
  "since February 10" is dropped by the anchor guard, so it falls back to the session
  date. The Nike fact now looks later, and the resolver supersedes Adidas with it.

First failing layer: **temporal keying**. The extractor keyed a replacement event with
the *old* value as its object. That error was already in the baseline store and did
nothing; the event-time split made it fire. On this cohort the split changed exactly one
lifecycle decision, and that one change is a regression.

Not fixed here: it is outside this registration and a fix needs its own evidence (see
`research/EXPERIMENT_INDEX.md`).

## Reproduction

```bash
lltm ingest run --config configs/v2b-batch8-repair.yaml \
  --questions results/manifests/v2b-gate16.json --store-name v2b-gate16-repair
python tools/store_fidelity.py stores/v2b-gate16-repair.db results/manifests/v2b-gate16.json \
  --pair-with results/analysis/store-fidelity.v2b-gate16.json
python tools/store_fidelity.py stores/v2b-gate16-repair.db results/manifests/v2b-gate16.json \
  --exclude-prefix g_ --pair-with results/analysis/store-fidelity.v2b-gate16.json
```

Run under a resume loop (retry every 10 minutes after exit 2, stop on 0 or any other
code, deadline 2026-09-30): 23 attempts, 22 of them quota stops. Git SHA `4da5868` plus
the uncommitted working tree described in the registration; extractor
`gemini-3.1-flash-lite`, `two-stage-p10-v2`, prompts `017e6620b1b9`,
`specificity-repair-v1:5022faf04eb0`.

Gate 7's collision case — one session repaired under two tenants — did not arise in this
cohort (no shared session was repaired in more than one namespace), so the id fix made
before the run is covered by its tests, not by this store.
