# v2c.7 dish-1 preregistration

Date: 2026-09-17 (Australia/Sydney)

## Observed v2c.6 failure

The first v2c.6 row returned only “Escovitch Fish” and was rejected by both the
strict criterion and configured judge. The second repetition was cancelled under
the early-stop rule. Prompt inspection confirms the v2c rule and evidence were
present, so context preservation was necessary but insufficient.

## Frozen change

Move the general exact-reference policy into the v2c system instruction, above the
question and evidence in instruction priority. Repeat a short form immediately before
the final question in the second-pass prompt. The policy is generic: resolve names,
titles, URLs, and links using the most specific source-backed distinguishing property,
not a conflicting broad-category sibling. No question id, dish, ingredient, answer,
retrieval, store, or model is encoded in the change.

## Gate

Run `778164c6` up to twice independently, stopping immediately on a strict failure.
Pass only if both raw hypotheses directly answer Grilled Snapper with Mango Salsa and
do not present Escovitch Fish as the requested answer. Judge labels are secondary.
