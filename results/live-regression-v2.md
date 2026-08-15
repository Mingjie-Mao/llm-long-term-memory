# Live-model regression: memory-aware answering, type-aware judging, raw fallback

**Run 2026-08-14 (PT). 7 questions × 3 runs. Not a benchmark row.**

Three capabilities had code and unit tests but had never run against a real model:

1. **Memory-aware answering** (`ANSWER_PROMPT_VERSION = memory-aware-v2`) — using
   retrieved memories as context to personalize from, rather than as a lookup table
   to quote or decline on.
2. **Type-aware judging** (`JUDGE_PROMPT_VERSION = lme-type-aware-v2`) — grading a
   `single-session-preference` gold as the rubric it is, not as a reference answer.
3. **Conditional raw-conversation fallback** — recovering original turns only when
   the answerer reports structured memory is insufficient.

This run closes that gap. It is **deliberately not comparable to anything**: the 7
questions were selected *because they failed*, so the accuracy is a regression
signal, not a measurement of the system.

- Manifest: `results/manifests/live-regression-v2.json`
- Config: `configs/fallback.yaml` (`fallback.enabled: true`, `top_k=10`)
- Artifacts: `results/raw/two_stage_fallback.live_v2_run{1,2,3}.jsonl`
- Prompts unchanged during the run. Existing dev31 and frozen v1 results untouched.

## Result

**6 / 7 correct, identical across all three runs.** These same 7 questions scored
**0 / 7** for every variant in the dev31 pilot.

| Case | Type | Verdict | Fallback | Correct | 3-run |
|---|---|---|---|:---:|:---:|
| Battery | preference | `answer` | none | ✅ | 3/3 same |
| Cookies | preference | `no_evidence` | archive-wide | ✅ | 3/3 same |
| Miami hotel | preference | `no_evidence` | archive-wide | ❌ | 3/3 same |
| Mindful.org | assistant | `no_evidence` | archive-wide | ✅ | 3/3 same |
| **Mayo URL** | assistant | `no_evidence` | archive-wide | ✅ | 3/3 same |
| Andy's shirt | assistant | `no_evidence` | archive-wide | ✅ | 3/3 same |
| Mod Podge | assistant | `no_evidence` | archive-wide | ✅ | 3/3 same |

Zero run-to-run variation is worth noting on its own: the measured repeat-run noise
floor on the stratified 50 is ±8 points (`results/table.md`), so a 3/3 agreement on
every question is a stronger stability signal than the headline number.

## The golden case, end to end

**Mayo URL (`41275add`)** — the case the whole fallback design exists for:

```
structured memory     : nothing about Mayo Clinic (verified in results/assistant-gap.md)
answerer verdict      : no_evidence
fallback              : archive_wide
turns recovered       : answer_sharegpt_81riySf_0:1  ← the gold source turn
answer                : "…titled 'How to Sit Properly at a Desk to Avoid Back Pain',
                         and you can watch it here: https://www.youtube.com/watch?v=UfOvNlX9Hh0"
judge                 : PASS — "provides the exact title and URL"
```

Compression dropped the URL; retrieval could not recover it at any ranking; the
archive returned the original assistant turn and the answer carries the exact
string. This is the architecture working as designed, on a real failure, with a
real model.

## The two preference fixes, separately confirmed

**Cookies (`38146c39`) — the judge fix.** Previously failed with *"the reference
answer describes the user's preferences, whereas the candidate answer provides
actual suggestions"* — the judge grading a rubric as a reference answer. Now:

> judge: PASS — "specifically references and builds upon the user's use of turbinado sugar"

**Battery (`09d032c9`) — the answerer fix.** Previously retrieved the right memories
and then declined: *"I do not know what phone you use"*. Now:

> "Since you commute 30-40 minutes daily and **have a new portable power bank and
> wireless charging pad**, you could keep your portable power bank charged in your
> bag for your commute…"

Verdict `answer`, no fallback: structured memory was sufficient once the answerer
was willing to apply it. That is the intended behaviour — the fix was meant to stop
unnecessary abstention, not to route more questions to the archive.

## The failure, unchanged

**Miami hotel (`0edc2aef`)** still declines: *"I do not know of any hotel preferences
for Miami, as your recent travel planning has been focused on Seattle."*

Consistent with the earlier audit, where `full_context` — seeing the entire history —
also scored 0/3 on this question. The reply suggests the namespace contains travel
planning for a different city, so this may be a question whose premise is not
supported by its own haystack. **Not investigated further here**, and deliberately
not fixed by prompt tuning during a regression run.

## What this run did *not* verify

**`source_local` fallback was never exercised.** Across 21 runs the verdicts were
`answer` ×3 and `no_evidence` ×18 — **`need_source` never occurred**.

That is the correct verdict for these questions: on all four assistant cases
structured memory was genuinely unrelated, not merely missing a detail. But it means
level 1 of the cascade — recovering the source turn of a memory that *was* found —
has unit tests and still no live confirmation. A case for it needs a memory that
retrieves correctly while lacking the requested specific, which is a narrower
situation than the four here.

Recorded in `results/raw-retrieval-regressions.json` as the next gap to fill.

## Cost

21 answerer calls plus 18 fallback second-passes, and 21 judge calls: the
`gemini-3.5-flash-lite` counter moved 236 → 278 and `gemma-4-31b-it` 229 → 250.
Under 10% of the daily allowance for three full repetitions, because the extractor —
the expensive pool — is not touched by an evaluation.
