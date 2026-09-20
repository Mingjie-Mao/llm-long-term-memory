# v2c.2 gate8 amendment

Registered after stopping v2c.1 hybrid rep1 and before any v2c.2 answer call.

v2c.1 memory-only passed twice. Its first hybrid run fixed both existing source-local
rows but failed the exact-date and temporal-composition targets.

The exact-date failure was mechanical: the precision trigger fired, but provenance
hydration expanded only the coarse fact's turn 2. “Valentine's Day” occurs in turn 8 of
the same located session and was absent from the answer context. v2c.2 ranks raw turns
inside the sessions already identified by selected memories and never crosses into an
unlocated session for this repair.

The temporal plan contained all three events in correct order, but the stochastic
reader again omitted the first. v2c.2 states that every item in the filtered plan must
appear in the main answer, in order. No event is added or removed.

The prompt version becomes `memory-aware-v2c.2`. All v2c.1 artifacts remain audit
evidence and are excluded from v2c.2 promotion counts. The same offline gate and two-run
memory-only/hybrid requirements restart under `gate8-v2c2-rep1/rep2`.
