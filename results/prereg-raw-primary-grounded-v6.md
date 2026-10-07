# Fixed model v6 — review facts before choosing an answer

2026-10-02, registered before v6 output. train150 is exposed development data;
heldout100 remains aggregate-only; dev100 remains context-only until adoption run.
No commits, model/config switch, gold-guided runtime logic or frozen edits.

## Evidence and repair

v5 stopped with eight graded rows (2 correct), repeated judge500/503; incomplete,
not overall accuracy. Preserve its raw/usage/inventory/source snapshot. All marked
raw sources reached the context, but the reader still chose the wrong dates/items:
lookup denied a contextual coupon store, song identity mixed with another style,
previous/current tennis schedules conflated, cumulative group attendance added,
and racket receipt-to-question-date substituted for purchase-to-receipt.
The validator also rejects QUESTION_DATE in reviewed_sources although it accepts
that endpoint as an operand. English word quantities lack numeric binding.

Small changes in the same reader:

- Group sources by actual session identity and turn order, with chat dates; put raw
  turns before derived memories within a turn. Preserve original body/id/provenance.
  Use compact headers, retain memory lifecycle/validity, count rendering in budget.
- Flat v6 schema lists short source-grounded observations/exclusions and reason
  before answer. Require event identities, original expressions, prior/current state,
  accumulated versus independent quantities, and conversational coreference.
- Validate copied observation quotes/ids before accepting lookup as well as arithmetic.
  Unsupported scope remains uncertain; do not assert a missing fact is proved absent.
  reviewed_sources lists relevant reviewed sources, rather than a census of unrelated
  ids. Aggregates require the source-grounded inclusion/exclusion review and explicit
  scope_complete. An id census never proved semantic completeness and repeatedly
  caused rejection before valid operands could be checked; keep the old rule in v1–v5.
- Permit reviewed QUESTION_DATE only for durations; reject it as a requested endpoint
  for a question explicitly asking between two historical events. Keep other versions.
- Bind simple English number words (zero through ninety-nine) to the adjacent original
  unit. No unconstrained text-to-number conversion or inferred quantities.
- One bounded same-pool correction for lookup validation/false abstention too; at most
  two same-model reader calls. No unconditional second pass of a valid supported answer.

## Verification before provider calls

Target source grouping/isolation/default compatibility, flat schema and field order,
quote/id validation, missing/contradictory facts, previous/current and accumulated
count examples, paired historical endpoint rejection, QUESTION_DATE review, written
quantities/currencies, product/eval identical requests, CLI construction. Full suite
and Ruff. train150 offline namespace grounded-context-v6: coverage >= v5 (146/146),
no body mismatch, median<=6000. dev100 context-only separate namespace, no gold tuning.
Failure of any gate stops calls and is retained rather than editing these rules.

## Real reader diagnostics and regression

The original nine reasoning failures plus the existing ten originally correct train
questions, same model once per item, new v6 namespace. <=38 reader calls excluding
internal retries. Retain reader output before invoking any judge. Because Gemma is
currently failing repeatedly, support a reader-only stage with explicit UNGRADED
artifacts, source-audited review on exposed train, and separate grading later with
unchanged judge. UNGRADED must not be called benchmark accuracy or gate PASS.
Record identity, prompt/schema/config/data/store/index SHA, usage/environment and
git SHA (dirty source inventory authoritative), temperature0, seed unsupported.
No further retry of the failed v5 judge run. Reader-only and graded resume use the
same saved answer rather than repeat paid reader work. No automatic model fallback.

Pass diagnostic mechanism review only if all supported known event/item selections
are correct and no mechanical rejection of a correct selection; document raw-source
ambiguity separately. Ordinary-correct sample must retain all source-supported
answers. Failure triggers a new version/namespace, not a rule change here. This is
failure-enriched regression, not a general accuracy or significance estimate.

Full dev100 only after valid diagnostic/regression and functioning original judge:
100 exact questions, 3 repeats/arm vs raw-primary, mean net>=4, per-type net>=-2,
median candidate context<=6000. No per-item dev100 tuning; 1200 reader/600 judge
upper bound before internal retries, original quotas. Unseen independent final still
required for a new official figure. Resolving these known errors does not guarantee
every future natural-language item has complete/consistent source evidence.
