# Seed field clarification

2026-10-10, during grading, before inspecting any new correctness outcomes.

The v2 acceptance preregistration accidentally says `seed 42`. The frozen execution
inventory correctly records `seed: provider unsupported`, and the immutable train150
manifest records split seed 0. All 21 IDs and their order were fixed explicitly by
`PILOT`; no resampling or provider seed was used. Temperature is 0 as registered.

This corrects the prose description of actual execution, not the question set,
model calls, pass/fail rules, original grades or auxiliary requirements. The original
preregistration and execution snapshot remain unchanged.
