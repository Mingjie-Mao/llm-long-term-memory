# Scoped t4 dev100 failure: diagnosed before new candidate

2026-10-11. Exposed-development diagnosis, zero provider calls. Protocol:
[`prereg-t4-scoped-dev100-postmortem-v1`](../results/prereg-t4-scoped-dev100-postmortem-v1.md).
The generated [full100 request audit](../results/analysis/t4-scoped-dev100-postmortem-v1.json)
verifies all600 answer/grade bindings and reproduces the failed frozen gate. It keeps
every positive, negative and neutral question, rather than just selected examples.

## What the completed experiment established

- Original t4:88/89/88, mean88.33%; repaired wrapper:84/91/83, mean86.00%.
  Mean net−2.33 fails the registered operational gate. The paired interval includes
  zero; this is not a statistically established harmful effect.
- No dev100 question activated a source-bound deterministic repair. Nevertheless
  every first model request changed because SelectionRepairRunner appended its
  global RULES even when selection_plan returned no calculation.
- Within each arm, every question's first request was identical across its three
  repeats. The resulting answer/score variation must not be described as three
  independent question sets or treated as proof of a code effect per question.

## First-loss inspection of all eight negative questions

The saved candidate actual requests were compared to USER reference text and
original raw dates. Source paths are
`results/raw/t4-scoped-regression-dev100-v3-{control,candidate}-rep{1,2,3}.{readers,grades}.jsonl`.
The `actual_answer_calls` records preserve full system/schema/context/response.

| Question | Control/candidate correct out of3 | First failing layer | Evidence and secondary issues |
|---|---:|---|---|
| `0bc8ad93` |3/2|S5|Museum reference is in all actual contexts; one candidate confirms visiting but omits whether a friend accompanied the user.|
| `681a1674` |3/2|S5|Both rewatched films are present. One answer mixes the two rewatched films with four other watched films instead of answering the requested count.|
| `945e3d21` |2/0|S5|August USER says yoga twice weekly; November30 USER says three times weekly, both delivered. Candidate repeatedly chooses the older twice value. Stale structured memory also labels twice as current; it is a secondary conflict, not missing final evidence.|
| `aae3761f` |3/2|S5|All three gold driving durations4/5/6 are delivered. One candidate adds other trips to its explanation and never returns the requested15-hour total.|
| `d851d5ba` |1/0|S0 / gold-scope ambiguity|The four gold amounts1000/2000/250/500 are present, but another USER literally says a concert “back in April” raised over5000. The raw phrase omits a year and describes a past event; it is not proof of a future April2023 event. Structured memory adds April2023. Excluding the raw past amount solely to fit3750 would encode the rubric into the answerer; the underlying scope/year ambiguity remains unresolved.|
| `gpt4_59149c78` |2/1|S5|Correct Met event text is present and anchored “today” to January15. Candidate selects City Art Museum or MoMA based on nearby conversation dates; a conversation recounting an earlier visit is not an exact event-date statement. Secondary memory-date precision is also suspect.|
| `gpt4_e05b82a6` |1/0|S5|All four sources are delivered and the answers themselves enumerate3+3+1+3, but report6 or8 instead of10. Arithmetic after operand selection is wrong; more retrieval cannot fix that.|
| `gpt4_f2262a51` |3/2|Evaluation inconsistency, not established answer failure|All three doctors are delivered through source excerpts: Smith/primary physician, Patel/ENT, Lee/dermatologist. Candidate outputs3/3/three doctors with the same names. Only the third receives False. Whole-reference coverage1/5 does not imply missing required evidence: the available excerpts already identify all three. Original score is retained.|

No new grading, relabeling or gold changes were performed. S5 diagnoses refer to
the delivered answer evidence, not a claim that every earlier storage layer is
correct. Two cases remain benchmark/evidence ambiguity rather than a demonstrated
model repair opportunity. The doctor and charity judgments are not silently changed.

## Smallest supported repair

Add a separate `ScopedSavingsRepairRunner`: evaluate existing source-bound repair
applicability from tenant-owned evidence. If no repair is supported, delegate to
the accepted MemoryRunner answer/fallback with unchanged context; do not append
RULES. If supported, reuse the existing repaired path without changing its
operands, derivation or model requests. No query IDs or gold enter this router.

This fixes the proved scope leak. It does not claim to solve residual frequency
updates, vague relative-time selection, arithmetic or rubric ambiguity universally.
The next proof is strict300 baseline request/output replays plus fixed21 acceptance,
explicitly historical evidence. A new original test100 protocol is required; the
old protocol's failed dev gate must not be patched or bypassed.
