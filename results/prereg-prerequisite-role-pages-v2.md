# Role page review r2 — development offline, no provider calls

Preserve failed v18 r1 routing: 140/139/138 of146 annotated sources delivered at1/2/4x.
Six assistant-source questions were misrouted to USER review; two personal event
queries were misrouted to ranking atgrowth. Broad first-person wording is not source intent.

r2 generic source intent rules distinguish references to previous assistant chat/advice
from personal events (including family birth counts and event order). No labels/IDs/gold
are read by product routing. Advice retains v15 composition with its known limits.
Oversized USER turns are split into exact contiguous spans with original turn ID and
character offsets; original text is reconstructable. No summaries. Selected spans remain
mandatory, no silent overflow; final context budget and page cap remain explicit.

Exposed train150 manifest, all150,146 annotated, synthetic1/2/4x growth; same baseline
and tools/role_paged_growth_replay.py. USER archive must be complete and contain every
annotated original source. Composite advice metric uses historical v15 growth coverage;
this is NOT a full new answer pipeline result or QA accuracy. Record page/token costs,
source hashes, exclusions, negatives. Max32 pages,6000 estimated tokens perpage/final.
If passed, test actual same-model selector on a preregistered small development set.
No official accuracy improvement or statistical claim may be made from this gate.
