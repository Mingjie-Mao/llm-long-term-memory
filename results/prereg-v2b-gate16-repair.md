# v2b-gate16 with the specificity repair — pre-registration

Date: 2026-09-23 (Australia/Sydney). Written before the first provider request of this
experiment. Git SHA at registration: `4da5868` plus the uncommitted working tree listed
under *Code state*.

## Classification

**DEVELOPMENT extraction experiment.** Option C of `results/measurement-ceiling-decision.md`:
spend on the extraction side and claim only extraction. LongMemEval-S is exhausted and
`v2b-gate16` has been inspected and used (it is the v2b and v2d gate). Nothing here is
an unseen result, and nothing here is a QA result: no question is answered, and the
outcome says nothing about 72% or the 86% / 72% gap.

## Question

When the whole ingestion pipeline runs at batch 8 with the specificity repair — scoped
ids, deduplication, supersession, store and index — does the store retain more of what
the user stated than the same pipeline without the repair, and at what cost in
unsupported content and requests?

The pilot (`results/prereg-specificity-repair-pilot.md`) answered this for the repair's
own loop on 60 holdout sessions: 44.8% -> 59.3%, 21 gained, 0 lost. It never ran inside
the pipeline. This does, on 780 namespaced sessions.

## Data

- Dataset: LongMemEval-S (`variant: s`), manifest `results/manifests/v2b-gate16.json`
  (16 questions, seed 20260916, SHA-256
  `31129e88209856434fc93b238a50abe191cec386648879e90f4476e1428a76d7`).
- 780 namespaced sessions (772 unique; sessions shared across questions are ingested once
  per question, as `namespaced_sessions` does).

## Arms

- **Baseline — existing, not re-run:** `stores/v2b-gate16.db`, built by the v2b gate at
  batch 8 without the repair. Read-only. Measured for free before this registration
  with `tools/store_fidelity.py` →
  `results/analysis/store-fidelity.v2b-gate16.{json,md}`:
  - 3,055 memories (3.92 per session);
  - recall **47.5%** (1,110 of 2,338 specifics), 0.363 kept per memory;
  - support **92.7%**; 284 of 2,003 checkable memories unsupported (**14.2%**).
- **Candidate — new:** `configs/v2b-batch8-repair.yaml` into a new store,
  `--store-name v2b-gate16-repair`. Same extractor model (`gemini-3.1-flash-lite`),
  extractor version (`two-stage-p10-v2`), prompts digest (`017e6620b1b9`), batch size
  (8) and deduplication (namespace, 0.92) as the baseline, checked by fingerprint diff.
  The only fingerprint differences are `schema_version` (the `observed_at` column added
  by the event-time split; the rulers read `content` and `object` only) and
  `specificity_repair: specificity-repair-v1:5022faf04eb0`.

## Measurements

All with `tools/store_fidelity.py` on `results/manifests/v2b-gate16.json`. No model calls.

1. **Candidate, whole store** → `results/analysis/store-fidelity.v2b-gate16-repair.*`,
   paired against the baseline reading.
2. **Candidate, repair excluded** (`--exclude-prefix g_`; the repair's ids start `g_`,
   the extractor's `mem_`) →
   `results/analysis/store-fidelity.v2b-gate16-repair.extracted.*`, paired against the
   baseline reading.

Measurement 2 against the baseline is **extraction drift**: the same configuration
extracting the same sessions twice. Measurement 1 against 2 is the **repair's own
contribution** inside one store. Measurement 1 against the baseline is **end to end**,
and is only attributable to the repair if drift is small.

The within-store comparison (1 against 2) cannot lose a specific by construction —
excluding memories only removes — so it carries no loss gate. It slightly understates the
repair where deduplication dropped an extractor memory as a near-copy of an earlier
repaired one; that case is counted, not corrected.

## Gates

Promote "batch 8 + specificity repair" to the ingestion configuration for the next store
build only if **all** hold:

1. **Repair contribution:** recall of measurement 1 minus measurement 2 is at least
   **+10 pp** (the pilot's own threshold; the pilot measured +14.5).
2. **End to end:** candidate recall minus baseline recall is at least **+10 pp**, and the
   paired comparison loses at most **22** specifics (2% of the baseline's 1,110 kept).
   Losses here can only come from drift or deduplication, so this is the gate that the
   pipeline did not undo what the repair added.
3. **Drift is small enough to read gate 2:** measurement 2's recall is within **±3 pp** of
   the baseline. If it is not, gate 2 is reported as confounded and the decision rests
   on gate 1 alone, stated as such.
4. **No new hallucination:** candidate support at least **91.7%** (baseline − 1 pp) and
   unsupported-memory rate at most **16.2%** (baseline + 2 pp).
5. **No dilution:** specifics kept per memory at least **0.363** (the baseline).
6. **Cost:** repair requests on at most **50%** of sessions (390), and repaired memories
   written at most **1.0 per session** (780), both from the ingest checkpoint.
7. **Tenant isolation, checked in the store:** every `g_` memory's `user_id` is the
   namespace it is stored under, its `source_session_id` is scoped by that namespace,
   and no memory id appears under two namespaces.
8. The run finishes without changing any frozen, sealed or archived artifact, or the
   baseline store.

Per-facet recall is descriptive. No significance test is applied or claimed; the rulers'
measured replicate noise is 0 of 134 specifics (arm R), which is what makes a
single-run comparison readable at all, not a test.

**What a pass permits:** building the next store at batch 8 with the repair on. Not a QA
claim, not a claim about 72%, and not a restatement of any v2 or v2b figure.

**What a failure records:** which gate failed and by how much, with both stores kept.

## Cost

Estimated from the baseline store, before running:

- extraction: ~212 requests (the baseline's own count), plus ~20 adjudications;
- repair: **343** requests — the number of baseline sessions that lost a specific under
  `missing_specifics`, i.e. what the repair would call for if the second extraction
  matches the first (44.0% of 780; the memo assumed 42%);
- total ≈ **575** successful requests against a 500/day extractor quota: at least **two
  quota days** with no provider failures. Provider `503 UNAVAILABLE` rates of 80–90%
  over the previous three days would multiply the wall time, not the request count.

**Stopping rule.** The ingest resumes after a quota stop or a provider failure from its
checkpoint. If it has not finished by **2026-09-30**, it stops, and the incomplete run is
recorded as incomplete: no gate is evaluated on a partial store.

## Code state

Uncommitted at registration, relevant to this run:

- `ingest/repair.py`, the pipeline wiring, `ingest.specificity_repair`, and the
  fingerprint key (`tests/test_specificity_repair.py`, `tests/test_repair_in_pipeline.py`).
- Found and fixed before any paid call: the repair's memory id was
  `hash((session_id, span))` — no tenant in it, and Python's per-process hash salt. A
  session shared by two questions would have written both tenants' repaired memory
  under one id, and `INSERT OR REPLACE` would have kept only the second. The id is now a
  SHA-1 of tenant, session, span and content; four tests fail on the old id and pass on
  the new one. Gate 7 checks the store for it anyway.
- `tools/store_fidelity.py` with `--exclude-prefix` and `--pair-with`
  (`tests/test_store_fidelity.py`).

Full suite at registration: 1,690 passed, 0 failed.
