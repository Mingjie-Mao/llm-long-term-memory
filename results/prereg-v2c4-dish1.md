# v2c.4 dish-1 preregistration

Date: 2026-09-17 (Australia/Sydney)

## Why v2c.3 is not accepted as 100%

The configured judge marked `778164c6` correct in both v2c.3 repetitions because
both answers contained the gold phrase “Grilled Snapper with Mango Salsa”. Manual
semantic inspection shows that each answer nevertheless led with “Escovitch Fish”
as the answer. That candidate matches Jamaican + snapper but fails the question's
fruit qualifier. The metric is therefore a false positive for a direct-answer
memory system.

## Frozen change

For an exact prior-reference question, add a deterministic instruction to resolve
all qualifiers jointly against the already-attached source-local evidence and not
present a partial-match sibling as the answer. No retrieval, source evidence,
models, store, or other routing changes.

## Minimal paid gate

Only `778164c6` is affected, so do not rerun the seven byte-equivalent contexts.
Run this one row twice. Pass only if both raw hypotheses directly identify
“Grilled Snapper with Mango Salsa” and do not identify Escovitch Fish as the answer,
regardless of the configured judge's label.

If it passes, combine these two row-level repetitions with the seven unaffected
v2c.3 rows for the gate-8 decision. Do not expand to 48 questions without separate
authorization.
