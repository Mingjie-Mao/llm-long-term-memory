---
name: memory-failure-debugger
description: Diagnose why one memory benchmark or custom memory question failed and locate its first-loss stage. Use for a wrong LongMemEval item, a first-loss request, or “the memory exists but the answer is wrong”; do not use for aggregate experiment design or for implementing a fix before diagnosis is requested.
---

# Memory failure debugger

Trace one question through the repository's actual evidence chain:

`raw source -> Stage A extraction -> Stage B temporal keying -> dedup -> lifecycle -> stored memory -> eligibility -> retrieval -> composition/raw fallback -> answer context -> final answer`

Read existing failure reports, result rows, logs, manifests, and diagnostic scripts before
reconstructing anything. Prefer `scripts/fact_lineage.py`, `scripts/stage_oracle.py`,
`scripts/session_recall.py`, `tools/failure_taxonomy.py`, and existing retrieval replay
tools when applicable. Do not modify frozen or sealed artifacts during diagnosis.

Assign exactly one earliest loss:

- S0: the expected fact is unsupported or ambiguous in the raw source.
- S1: extraction omitted or corrupted it.
- S2: temporal keying, dedup, or lifecycle corrupted/suppressed it.
- S3: it was stored but made ineligible by namespace/status/time filtering.
- S4: eligible evidence was not retrieved/ranked.
- S4b: retrieved evidence was dropped or damaged during context composition/fallback.
- S5: the answerer received sufficient evidence but selected operands, reasoned, computed,
  followed instructions, or abstained incorrectly.

Do not label S4 when the fact was never extracted or when sufficient evidence is already
in the final answer context. If fixing the first loss would still leave a later failure,
name that later issue as a secondary defect.

Report question ID, expected answer/evidence, source session, extracted facts, lifecycle
state, retrieval/fallback result, answer context, actual answer, first-loss stage,
secondary defects, the smallest plausible fix, and every file/artifact inspected.
