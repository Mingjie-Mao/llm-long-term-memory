---
name: answer-synthesis-debugger
description: Diagnose wrong or abstaining answers after sufficient evidence already reached the answer context, especially count, duration, comparison, temporal, preference, or multi-session synthesis. Do not use when extraction, eligibility, retrieval, or composition has not yet been shown successful.
---

# Answer synthesis debugger

First prove that the final answer context contains sufficient gold evidence. If it does
not, return to `memory-failure-debugger`; do not force an S5 diagnosis.

Classify the requested operation as lookup, temporal, count, duration, comparison,
preference transfer, or multi-session aggregation. Inspect operand selection, evidence
inclusion, duplicates, exclusions, time boundaries, arithmetic, aggregation,
instruction-following, and abstention. Compare the answerer's selected operands to the
context before changing prompts.

If retrieval succeeded, prefer a minimal operand-selection or deterministic-operation
repair over more retrieval complexity. For arithmetic, require auditable operands and
let code compute; for false premises, require source-grounded correction rather than a
forced nearby answer. Preserve negative results and do not modify frozen rows while
debugging.

Report operation, sufficient evidence, chosen versus required operands, exact failure,
smallest repair, regression risks, and the targeted evaluation needed to validate it.
