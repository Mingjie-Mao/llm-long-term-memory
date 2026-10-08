# Exhaustive evidence pages — development prerequisite, zero-call gates

The allocation-only v16 experiment retained 146/146 at 1x, 144/146 at 2x and 134/146
at 4x train150 history. This is a preserved failed gate, not a successful fix.
The first loss is final rank/budget composition, with original turns intact.

Candidate v17 adds an opt-in exhaustive page review using the SAME answerer model.
Each page <=6000 estimated tokens; whole raw turns and source dates retained. Max32
pages by default. Each page must acknowledge every source, selecting relevant and
uncertain sources; final selected pool must also fit <=6000. Invalid selection, oversized
turns, page cap or selected-pool overflow => explicit incomplete answer + evidence gap.
No hidden truncation and no fabricated zero/total. Overlapping events are not deduplicated
by text; original source identities remain separate. Same-user raw hydration only.

Acceptance now: fault/source/tenant unit tests, zero-call actual train150 1/2/4x archive
page coverage, raw SHA equality, recorded page-count distribution and projected extra
answerer requests. Compare coverage across ALL review pages, not just final context;
this is different from the earlier single-context metric. No accuracy claim or inference
that a model will select every relevant source correctly. Verify final calculation with
synthetic multi-page source selection, wrong role/event and cap failure tests.

No paid calls authorized by this protocol run. Reader/judge accuracy comparison is a
separate next gate in a fresh namespace. Cost optimization/recursive summary must include
all page-selection calls and cold/update summary cost, with raw sources authoritative.
