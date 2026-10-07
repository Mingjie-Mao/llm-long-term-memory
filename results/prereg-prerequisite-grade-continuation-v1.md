# One supplemental judge call after v9 recovery exhaustion

2026-10-04 Australia/Sydney. User requested continuation after the bounded v9
grading process stopped. Original v9 readers, five grades, usage and exhausted
recovery state remain unchanged. The original failure is not reset or concealed.

Only the missing `eaca4986` grade may be requested, using its exact saved v9
answer, original Gemma judge/model/prompt and public audited corpus. No readers
run, no existing grade is repeated, no answer/gold is changed. New namespace:
`prerequisite-grade-continuation-v1`. One client attempt (max_retries=1,
max_transport_retries=1), no outer recovery. Usage or attempt state prevents
restarting this namespace for another call. Failure remains incomplete.

Freeze original inventory/readers/grades/usage/recovery, public source, this
protocol and generator hashes in a zero-call preview before executing. Require
six readers, five hash-bound grades, precisely this one missing question and
exhausted recovery. Validate every original inventory file hash.

On success generate a separate joined view of unchanged original readers and
five grades plus this single supplemental grade. Preserve all inputs and record
their hashes and combined usage. Apply the same v9 five-supported-plus-faithful
temporal-gap mechanism gate and report the original six-question score separately.
This is completion of missing grading, not another reader run or a new population
accuracy estimate. Do not repeatedly create further continuation namespaces.
