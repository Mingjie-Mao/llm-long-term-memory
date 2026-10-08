# Fixed-model grounded answering — registration v1

2026-10-02, Australia/Sydney. New namespace; no provider requests before registration.
This supersedes the *planned execution* of the unrun raw-primary dev100 v4 bundle.
Its old registration and gate remain unchanged. No new accuracy figure is claimed.

## Requirement and first losses

Keep `gemini-3.5-flash-lite` as reader, the existing extractor/judge/embedder unchanged.
On exposed train150, the prior diagnosis found 14 incomplete/missing-evidence failures
and 9 sufficient-evidence answer failures. Existing v2d gate16 produced code results
on only 2 questions, with 9 refusals and no net accuracy gain; v4.2 citation enumeration
also did not qualify. Do not assume another citation prompt will improve accuracy.

Concrete defects addressed: raw turns had no operand handles; source-label validity
did not check a value against its quote; parallel operand arrays could misalign;
fallback discarded the structured computation path; conversation age was suggested as
event duration. New policy `memory-grounded-v1` uses whole source records, verifies
quoted values and units, retains all supplied operands/refusal causes, resolves dates
from their own source words, and computes only after validation. Same policy is
callable from the product AnswerEngine and MemoryRunner. Defaults are not promoted.

Retrieval reuses fact-keyed BM25+dense+stored-memory-anchor fusion. No new retrieval
method or datastore. Source headers/labels count against a hard 6000-token context
estimate (characters / 4.6). Whole sources only; dropped ids are recorded.

## Offline stage — train150 decides, before paid evaluation

Manifest `results/manifests/train150.json`, store `stores/train150.db`, already inspected,
development/regression evidence. Build `train150-turn-key-index` locally using the
existing MiniLM model. Compare actual baseline raw-primary 4k excerpts to the candidate's
final packed raw evidence; only full verbatim gold turns count, not memory anchors.

Generator `tools/grounded_context_replay.py`; report namespace
`results/analysis/grounded-context-v1.train150.{json,md}`. Gates: all-gold-turn coverage
gain at least 2 percentage points, no stored gold-text mismatch, median context <=6000.
Failing this gate prevents the full paid comparison. Preserve the result if it fails.
dev100 gets context-only preflight, never per-question failure/flip selection.

## Bounded diagnostic before dev100

After offline gates pass, use the already exposed 9-question manifest
`results/manifests/train150-raw-v1-reasoning-errors.json` on train150 once, candidate
only, label `grounded-diagnostic-v1-rep1`. At most 18 reader calls plus 9 judge calls,
before retries; capped recovery at one additional reader call. This checks actual
schema acceptance, operand grounding, calculation execution and refusals. It is
failure-enriched diagnostic evidence, never an accuracy estimate or adoption gate.
The prior lite rerun (0/9) is historical context, not a fresh paired control.
Any repair prompted by this diagnostic must receive a new prompt/config and execution
namespace; preserve this first run. Do not inspect dev100 failures for further tuning.
Driver `tools/run_grounded_experiment.py` defaults to a zero-provider-call cost report;
`--execute` freezes code/config/manifest/store/index hashes before the first call.

Offline cost: zero provider requests/tokens; local embeddings plus stub replay of two
arms. Report manifest, code/config hashes, exact models, git SHA, environment and usage.

## Conditional reader comparison — dev100, once

Only after offline gates, isolation/unit tests, full tests and lint/format pass, freeze
source/config/manifest hashes into a separate execution inventory before provider calls.
Dataset LongMemEval-S, manifest `results/manifests/dev100.json`, store `stores/dev100.db`,
development comparison recorded as regression. This set is exposed and is not unseen.
No later tuning against its per-question answers or failures.

- Baseline `two_stage_raw_primary` / prompt `memory-aware-v2`, output cap 512.
- Candidate `two_stage_raw_primary_grounded` / prompt `memory-grounded-v1`, output cap 2048.
- Config `configs/fallback.yaml` for both. Reader `gemini-3.5-flash-lite`, judge
  `gemma-4-31b-it`, existing extractor `gemini-3.1-flash-lite`, embedder
  `sentence-transformers/all-MiniLM-L6-v2`, same existing stores, no new ingestion.
- 100 exact questions, three repeats each arm, temperature 0; provider seed unsupported.
- Namespace `raw-primary-grounded-dev100-v1-rep{1,2,3}`; never overwrite old runs.
- Adoption rules unchanged in magnitude from v4: mean paired net >=4/100; every type's
  mean net >=-2; median candidate context <=6000. The +4 is an adoption threshold,
  not a guarantee of statistical significance. Existing dev100 resolution analysis
  estimates a one-run floor of 7, which motivates three repeats. Questions remain the
  independent units, not the 300 repeated answers.
- Gate must require exactly the manifest ids, no duplicates, boolean correctness,
  matching type/model/prompt/config/store identities and all six complete runs.
- Diagnostic only: abstentions, source-review completeness, validated calculation
  frequency/refusals, duplicates merged, date precision, fallback calls, token usage.

Upper bound is 1200 reader calls (at most two per question) plus 600 judge calls, before
provider retries. The historical ~680-reader-call estimate is not a quota guarantee:
strict validation may cause more recoveries or abstentions. Check available quota and
record provider errors/retries; stop rather than silently change the model or schema.
Do not schedule continuation unless the user requests it. Missing runs yield INCOMPLETE.

A pass makes this a candidate for a new unseen comparison, not a formal new accuracy
claim. A failure leaves v1 as the candidate and preserves the failed bundle.

## Interpretation boundaries

Source review is model-reported; it cannot prove no historical item was missed.
Quotes prove source membership, not that the selected item semantically answers the
question. Identity keys and scope still require model decisions and measured validation.
Approximate dates remain approximate; ambiguous dates are refused. The hard context
limit is an existing character estimate, not an exact tokenizer bound. Multiple
mechanisms are bundled, so a pass cannot attribute the gain to one component.
