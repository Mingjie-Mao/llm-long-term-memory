# Scoped t4 selection acceptance v2

Registered before fresh reader calls, 2026-10-10 Australia/Sydney. No commits.

## Evidence and scope

Class: exposed train150 known-case development acceptance, not population accuracy
or unseen final measurement. Fixed 21 IDs from `run_recursive_summary_cost.PILOT`,
manifest `results/manifests/train150.json`; seed 42, temperature 0.

Historical comparator: immutable `t4-selection-train21-v1`, original 16/21,
supported 15/18, auxiliary 0/3. All old rows/rules remain unchanged. The accepted
heldout100 t4 aggregate comparison (900 readers/grades) remains immutable and is
not used for question-level repair diagnosis.

Zero-call prerequisite: `t4-scoped-selection-train21-v2.preflight.json`.
It reproduces the second song at BM25 rank 5 but absent from packed context,
verifies source ownership, and estimates median context 6,374. These are composition
and provenance measurements, not accuracy. This enriched fixed set detects concrete
known failures; no significance or population improvement will be inferred from it.

## Explicit correction to the prior auxiliary rubric

`gpt4_2f8be40d` moves from ambiguous-source auxiliary to original-gold supported case.
The complete USER sources establish three distinct attended events:

- `answer_e7b0637e_1`, 2023-10-15: cousin's vineyard wedding in August; neighboring
  USER turns name cousin Rachel and say the user was a bridesmaid.
- `answer_e7b0637e_2`, same date: returned from college roommate Emily/Sarah's city
  rooftop wedding; the neighboring cousin-Emily mention describes the same setting.
- `answer_e7b0637e_3`, same date: returned from Jen/Tom's rustic barn wedding last
  weekend, repeated consistently in later USER turns.

Recent-event language anchors them in the question year; precise days are not needed
to distinguish these three. The previous auxiliary rubric incorrectly demanded an
abstention solely for missing exact dates. The original answer of three remains gold.
This correction is registered openly, not a retroactive change to the failed gate.

## Candidate

Baseline implementation remains `two_stage_raw_primary_t4`, config
`configs/fallback.yaml`, store `stores/train150.db`, existing MiniLM 384-D memory index.
Candidate experimental wrapper `tools/t4_selection_repair.py`, prompt version
`memory-t4-scoped-selection-v1`. No gold, IDs, or reference flags enter its answer path.

Same answerer `gemini-3.5-flash-lite`, extractor `gemini-3.1-flash-lite` (no new
extraction), judge `gemma-4-31b-it`. Changes are exact collection-category checks,
same-session action/place inference, contiguous ordinal artifact hydration, explicit
quantity-period disclosure, and Decimal sums with temporal-conflict disclosure.
Deterministic source-bound output can replace the model's proposed output. Each
question still records an actual call to the same answer model, supplied derivation,
unmodified model response, final output and provenance. No archive pagination.

## Procedure and gates

1. Save one fresh reader for every fixed question before any grading. No answer cache.
2. Save all 21 original-gold judgments unmodified.
3. Separately grade the two real source gaps using the same judge:
   `f9e8c073`: report periods and overlap missing; no exact combined 3+5 total.
   `gpt4_731e37d7`: quoted payments 20+200+500 = 720, distinguish quoted subtotal
   from requested four-month total; preserve March/February conflict, clarify date
   and year, invent no corrected year. These never substitute for original grades.
4. Pass requires all 19 supported original judgments correct, both faithful-gap
   judgments correct, median actual supplied context <= 6,500. Report original
   21-question score separately; conservative whole-reference coverage is diagnostic.
5. Failure blocks summary calls. Success permits preparing a new full regression
   before a t4-based recursive-summary comparison; it does not promote the wrapper
   to production or revive the failed v18 conditional summary experiment.

Resources: 21 fresh readers (at most 42 nominal model calls with fallback), 21+2
fresh grades, about 150k reader-input tokens plus 10k grading-input tokens. Retry
configuration: two API retries and two transport retries (three shared attempts
per generation invocation), three durable judge recoveries with 180-second maximum
cooldown. Maximum 126 reader attempts and 276 grade attempts; no quota reset bypass.
Record successful and failed attempts, tokens, per-model usage, end-to-end wall time.

Before execution the driver hashes source/config/data/manifests, this preregistration,
the zero-call report and previous failed gate, records git SHA/environment and saves
literal source snapshots. Resume rejects changed identities. Outputs use only
`t4-scoped-selection-train21-v2` under `results/raw/` and `results/analysis/`.
