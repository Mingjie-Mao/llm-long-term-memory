# Capturing preferences shown in passing — extraction probe — pre-registration

Date: 2026-10-01 (Australia/Sydney). Written before any call of this probe.

## Classification

**DEVELOPMENT, extraction only.** No question is answered or judged. The cohort is every
single-session-preference question of LongMemEval-S outside `test100` — 24 of the
benchmark's 30 (dev50 3, train150 9, dev100 6, heldout100 6) — all already exposed.
`test100`'s 6 are not touched.

## Why

A pinned preference block (Letta's core memory, Mastra's always-present observations)
was measured first, at zero cost, and could not help: across these 24 plus the three
sets' other material, **no preference or profile memory is anchored to any gold turn of
any preference question** (`results/prereg-raw-primary-dev100-v4.md`, "Not included").
The evidence those questions need — "I'm usually in bed by 9:30" for "suggest something
to do in the evening" — was never written down as a preference. Mastra's Observer
records "a preference expressed in passing" as its own observation. This probe asks
whether one instruction to the extractor does the same.

Answering is not measured: 24 questions of the least stable type (37.5% of preference
questions flipped between unchanged heldout100 runs) cannot resolve an answer effect.
Whether the evidence gets written down can be counted exactly.

## Arms (the 24 gold sessions, 8 per request, `gemini-3.1-flash-lite`)

| arm | extractor |
|---|---|
| **plain** | `TwoStageExtractor`, the production prompt |
| **observer** | the same with `preference_observer=True`: one appended paragraph (`extract_facts.PREFERENCE_OBSERVER`) asking for preferences revealed by habits, routines, constraints and offhand remarks, as `preference` lines in the user's own words |

Anchors are `provenance-v2`, applied inside the extractor as for any new ingest.

## Measures (per question)

- **preference-covered**: a memory of type `preference` or `profile`, or Stage A scope
  `preference`, anchored to one of the question's gold turns;
- **any-covered**: any memory anchored to a gold turn;
- **volume**: memories per session.

## Decision rule

The observer instruction is worth an ingest-scale registration if **both**:

1. preference-covered rises by at least **6 of 24** over plain;
2. volume rises by at most **30%**.

Otherwise it is recorded and not pursued. The observer's preference lines on gold turns
are listed in the report for reading: a line that invents a preference the conversation
does not support is a failure however the counts come out.

## Cost

3 requests per stage per arm: **about 12 extractor calls** from the 500-a-day pool.
The raw extractions are archived (`results/raw/preference-extraction-probe-v1.json`) so
the analysis reruns without calls.

## Reproduction

```bash
python tools/preference_extraction_probe.py          # plan and cost, no calls
python tools/preference_extraction_probe.py --run    # extract, archive, analyse
python tools/preference_extraction_probe.py --analyse-only
```
