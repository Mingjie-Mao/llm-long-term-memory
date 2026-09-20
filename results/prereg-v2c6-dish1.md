# v2c.6 dish-1 preregistration

Date: 2026-09-17 (Australia/Sydney)

## Root cause isolated after v2c.5

v2c.5 failed its single directional row. Code audit showed that when the first pass
returned `no_evidence`, the generic fallback prompt discarded the complete first-pass
context. That dropped all v2c query-time rules and the deterministically hydrated
source evidence before the final answer was generated. The prompts added in v2c.4
and v2c.5 therefore could not govern the final answer on this execution path.

## Frozen change

For `answer_policy=v2c` only, the second-pass prompt now retains the entire first-pass
context and appends fallback evidence. Other answer policies retain their existing
templates. No ingestion, retrieval, evidence selection, store, or model changes.

## Gate

Local tests must prove that both the exact-reference rule and first-pass verbatim
evidence survive into the second prompt. Then run `778164c6` twice independently.
Pass only if both raw hypotheses directly answer Grilled Snapper with Mango Salsa
and do not present Escovitch Fish as the requested answer. The configured judge is
recorded but cannot override this strict criterion.
