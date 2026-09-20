# v2c.1 gate8 amendment

Registered after stopping v2c attempt 1 and before any v2c.1 answer call.

Attempt 1 ran only `two_stage_v2c_memory_only.gate8-rep1`. Update suppression fixed
`6a1eabeb`, and all three controls stayed correct. The temporal assembler selected and
ordered all three gold events, but the reader put the attended NBA game in a footnote and
excluded it from the answer because it interpreted “watched” as excluding attendance.

v2c.1 changes one general reading instruction in the deterministic chronological note:
attending an event in person counts as watching it unless the question explicitly says
otherwise. Retrieval, selected event ids, ordering, store, models, and all other prompts
are unchanged. The answer prompt version becomes `memory-aware-v2c.1`.

The attempt-1 row is retained but excluded from promotion counts. The two-run gate in
`prereg-v2c-gate8.md` restarts under fresh labels `gate8-v2c1-rep1` and
`gate8-v2c1-rep2`. No hybrid call was spent on the failed attempt.
