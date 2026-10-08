# Six-question same-model selector gate r2

Preserve v1 failure (2 provider requests,0 saved readers,0 grades) and exact raw diagnostic
v1/v2. The structural failure: returned E80..E97 while page has105 sourcesE1..E105.
Explicit full receipt list fixed this on the one-page diagnostic. Same selector schema,
model, source page layout, question set, limits and pass criteria as v1; no gold in prompt.
Runner now returns and saves incomplete reviews instead of crashing and records invalid
raw selection output. Bounded reader and judge retry policy remains unchanged.

Use NEW namespace prerequisite-reader-v18-v2, never fill/overwrite v1. Same six original
train150 histories, same-model29 nominal reader-call preflight plus6 judge calls.
No claims about full QA,4x answer accuracy or unseen-final. Existing growth-v3 delivery
artifact applies to unchanged page construction; changes only concern ID acknowledgement
and failure propagation, verified in targeted tests. QA gate still independently requires
all relevant original evidence in final context and all6 correct grades. Frozen identity
records the current complete source/config/data/index snapshot and this protocol.
