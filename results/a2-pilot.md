# A2 Pilot — Causal Ablation on 31 Fully-Ingested Questions

**Status:** interim pilot on a partial ingest. Not the A2 result, not a table row.
Full A2 runs on the frozen `dev50` manifest once the ingest reaches all 50
namespaces.

## Scope

The two-stage ingest paused on daily quota at 1,528 / 2,400 sessions. Thirty-one of
the fifty dev namespaces are **fully** ingested; a thirty-second (`c9f37c46`) has 30
of its 50 sessions and is excluded, because a partial haystack produces a retrieval
miss that is an artifact of the pause rather than a property of the system.

Retrieval is namespace-isolated (`memory.user_id == namespace`, and the namespace is
the question id), so these 31 results stay valid when the ingest completes; the run
resumes over the remaining 19.

Question set: [`results/manifests/dev31-pilot.json`](manifests/dev31-pilot.json), a
subset of the frozen [`dev50`](manifests/dev50.json). **Not stratified** — it is an
ingest-order prefix, so category splits below are descriptive only.

## Result

All variants on the same 31 questions, same answerer, same judge:

| Variant | Accuracy | Median ctx tokens | Source-session recall |
|---|---:|---:|---:|
| `full_context` | **64.5%** | 109,605 | — |
| `naive_rag` | **51.6%** | 13,057 | — |
| `two_stage_hydrated` | **51.6%** | 1,318 | 93.5% |
| `two_stage` | **48.4%** | **439** | 93.5% |
| `chronomem` (frozen v1) | **19.4%** | 465 | 80.0% |

Paired exact McNemar, disagreements only:

```
chronomem  vs two_stage             11W- 2L of 13   p = 0.0225   <- the extraction pipeline
two_stage  vs two_stage_hydrated     2W- 1L of  3   p = 1.000    <- hydration
chronomem  vs two_stage_hydrated    11W- 1L of 12   p = 0.0063   <- both together
naive_rag  vs two_stage              6W- 7L of 13   p = 1.000
naive_rag  vs two_stage_hydrated     6W- 6L of 12   p = 1.000
full_context vs two_stage            2W- 7L of  9   p = 0.180
full_context vs two_stage_hydrated   2W- 6L of  8   p = 0.289
```

## Attribution: the gain is the extraction pipeline, not hydration

Decomposing the 19.4% → 51.6% jump:

| Step | What changed | Delta | Paired result |
|---|---|---:|---|
| `chronomem` → `two_stage` | two-stage extraction | **+29.0pp** | 11W-2L, **p = 0.022** |
| `two_stage` → `two_stage_hydrated` | evidence hydration | +3.2pp | 2W-1L, p = 1.000 |

**Essentially all of the measured gain comes from the rewritten extraction
pipeline.** Hydration adds 3.2 points across 3 disagreements — indistinguishable
from noise at this sample size — while tripling median context (439 → 1,318 tokens).

Source-session recall is **93.5% for both**, identical to three significant figures.
That is the confirming detail: hydration operates after retrieval and cannot change
which sessions are recalled, so the 80.0% → 93.5% improvement belongs to extraction
too. An earlier draft of this file credited the gain to hydration. That attribution
was wrong, and the `two_stage` row is what corrects it.

## What this supports

**Structured memory is a viable compression of conversational history.**
`two_stage` reaches accuracy indistinguishable from `naive_rag` (6W-7L, p = 1.000)
on **439 median context tokens against 13,057** — a ~30x reduction. That is the
strongest defensible claim in the pilot, and it belongs to the cheapest variant.

**The v1 diagnosis was right about the disease, wrong about the cure.** v1 lost
answer-bearing detail during extraction. Fixing extraction fixed it. Rehydrating raw
evidence — the other candidate fix — is not yet earning its context.

## What this does not support

- **Hydration is not shown to work.** 2W-1L on 3 disagreements shows nothing in
  either direction. It is not shown to be useless either; it is untested at this n.
  Its 3x context cost, however, is measured.
- **No win over `naive_rag`.** Both memory variants draw (p = 1.000). The claim is
  equal accuracy at a fraction of the context, not better accuracy.
- **No loss to `full_context` either.** −16.1pp (`two_stage`) and −12.9pp
  (`two_stage_hydrated`) both fail to reach significance (p = 0.180, p = 0.289).
  With 8–9 disagreements there is not enough statistical power to judge the gap in
  either direction. "Not significant" here means *undetermined*, not *equal*.
- **Not the A2 decision.** 31 unstratified questions, one run each. The ±8-point
  repeat-run noise floor measured on the stratified 50 applies at least as strongly.

## Two categories score zero for every memory variant

| Category | full_context | naive_rag | v1 | two_stage | hydrated |
|---|---:|---:|---:|---:|---:|
| `single-session-assistant` (4) | **4/4** | **4/4** | 0/4 | 0/4 | 0/4 |
| `single-session-preference` (3) | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |

These are **different problems** and must not be pooled:

- **`single-session-assistant` is a LLTM defect.** Both baselines answer all
  four; every memory variant answers none. The plausible mechanism is that
  extraction records user facts and discards what the *assistant* said, which would
  make these questions unanswerable from the store regardless of retrieval quality.
  Checkable directly against the store, no quota required.
- **`single-session-preference` is not a LLTM defect.** `full_context` sees the
  entire history and still scores 0/3, so the failure is upstream of memory — the
  answerer prompt, the judge, or the reference answers. Diagnosing this by changing
  the memory system would be chasing the wrong layer.

## Process note

The first attempt at this pilot was invalid; its rows were discarded and re-run.
`--limit 31` does not evaluate the first 31 questions of the 50-question set.
Stratified samples *do* nest (`stratify(pool, 31)` is a subset of
`stratify(pool, 50)`), but they are drawn per category and re-sorted by id, so a
sample is scattered across all 50 positions rather than forming a prefix. Ingestion
processes namespaces in dataset order, so a partial ingest leaves data for a prefix.
Assuming those two orderings agreed cost 11 questions, which retrieved nothing and
were scored wrong, producing a spurious 35.5%.

`eval run` now takes `--questions <manifest>`, and `eval freeze` writes manifests
that refuse to be silently regenerated. A4's pre-registered held-out run needs
exactly this.

## Reproduction

```bash
lltm eval run two_stage --questions results/manifests/dev31-pilot.json --store-name two-stage-hydrated
lltm eval run two_stage_hydrated --questions results/manifests/dev31-pilot.json --store-name two-stage-hydrated
lltm eval compare two_stage two_stage_hydrated
```

Artifacts: `results/raw/two_stage.jsonl`, `results/raw/two_stage_hydrated.jsonl`
(31 rows each) and their `.usage.json` companions.
