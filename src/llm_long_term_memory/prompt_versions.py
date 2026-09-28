"""Prompt versions that both the product and the benchmark have to agree on.

The judge's prompt text belongs to evaluation, but its version does not only: a
recorded demonstration run carries the judge's verdict, and the service reports such a
recording as stale when the judge that graded it is no longer the current one. Keeping
the version here lets the product read it without importing the evaluation package.

The prompt text stays next to the judge. `tests/test_prompt_versions.py` pins a digest
of that text to this version, so editing the prompt without bumping the version fails
instead of silently relabelling old verdicts as current.
"""

#   reference-v1       one reference-answer prompt plus an abstention prompt
#   lme-type-aware-v2  2026-08-14: routed by question type, so a preference gold is
#                      graded as the rubric it actually is
JUDGE_PROMPT_VERSION = "lme-type-aware-v2"
