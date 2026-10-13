# Scoped-dispatch structural regression and known-case replay acceptance

2026-10-11 Australia/Sydney, before computing this new candidate's replay outcomes.
The separately registered dev100 postmortem found that the failed scoped wrapper
changed every first model request, even though none of the100 questions activated
its deterministic repair. Preserve the full failed comparison88.33% versus86.00%.
Removing the unrelated prompt suffix requires a new candidate and proof, not a
reinterpretation of that failed gate.

Candidate `ScopedSavingsRepairRunner`, `memory-t4-scoped-dispatch-v1`: activate the
previous source-bound repair only when its existing tenant-scoped operation returns
a supported result; otherwise call accepted t4's original answer/fallback method
with unchanged context. No IDs, reference flags, gold or diagnostic outcomes enter
the runtime. Models, extraction, store/index, temperature and quota are unchanged.

## Evidence and questions

LongMemEval-S, exposed dev100 and fixed21 train150 selection cases. This is a
zero-provider-call structural/replay acceptance, NOT a fresh600-answer regression,
population accuracy improvement, or statistically significant result.

- All300 saved original-t4 control readers (three repeats on all100 dev100 IDs)
  must replay every exact logical request, generation count and final answer with
  the new dispatcher. No mismatching request is sent to any provider. Historical
  grading can be retained only under unchanged question, gold, type, abstention,
  judge prompt, hypothesis and saved-reader SHA binding.
- Fixed21: choose the saved v4 repaired reader when the source-bound applicability
  function returns a mechanism; otherwise choose saved original t4 reader from
  `t4-selection-train21-v1`. This choice is made by runtime applicability, not its
  historical correctness. Require exact entire request sequence and final output.
- All18 supported known cases must pass their original saved grade, all3 faithful
  gaps must pass the previous separate faithful-gap rubric, and median candidate
  context must remain<=6500. Original21 gold scores remain separately reported.
- Any request/output mismatch, changed grade binding, missing row, foreign ID,
  failed known-case check, or source drift FAILS this acceptance. Mismatches are
  preserved; any later paid acceptance requires separate registration.

Successful exact replay proves the scoped dispatcher preserves the accepted t4
inference requests on these100 unactivated questions and the prior repaired paths
on activated known cases. It does NOT resolve all residual reasoning errors,
grading ambiguity, or generalize to new questions. Report the300 prior outcomes as
historical replay with zero fresh readers/grades. This acceptance is an explicit
different prerequisite for a separately registered original test100 comparison;
it cannot satisfy or alter the old v1 registration's failed fresh-dev prerequisite.

## Traceability and cost

Freeze `tools/t4_scoped_dispatch.py`, the replay generator/tests, old source receipt,
all source readers/grades/gap grades, the failed gate/postmortem, exact manifests,
config/store/index and this registration before replay. Save separate execution
identity and per-row source/input/output/grading receipts in
`results/analysis/t4-scoped-dispatch-replay-v1.*`. Record original model IDs
gemini-3.5-flash-lite/gemma-4-31b-it, unused extractor gemini-3.1-flash-lite,
MiniLM384, manifest seeds, Git SHA/Python and source prompt versions.
Fresh provider requests, paid tokens, extraction, grading and quota: zero.
Do not inspect heldout100 item errors or original test100 outcomes.
