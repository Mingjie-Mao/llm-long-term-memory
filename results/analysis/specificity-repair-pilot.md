# Specificity repair pilot

> **DEVELOPMENT RESULT**: source-text fidelity mechanism experiment, not QA or unseen final evidence.

Decision: **PASS_TO_QA_GATE**

- baseline: 44.8%
- candidate: 59.3%
- delta: +14.5%
- paired specifics: 21 gains / 0 losses
- repair calls: 25/60
- added memories: 20 (0.33/session)
- grounding failures among accepted repairs: 0

## Registered gates

- PASS — `fidelity_gain_at_least_10pp`
- PASS — `at_least_10_gains`
- PASS — `zero_losses`
- PASS — `every_repair_is_grounded_in_a_user_span`
- PASS — `repair_calls_at_most_half`
- PASS — `added_memories_at_most_one_per_session`
- PASS — `cohort_complete`
