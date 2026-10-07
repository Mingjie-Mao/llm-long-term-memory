# V13 bridge preflight — first loss before ordinal selection

Before v13 outputs, inspecting v12's original ledger confirms assistant artifact
turn1 andturn3 and initial user creation turn0 were delivered, but the user revision
turn2 was omitted. Thus contiguous-turn proof cannot be assumed from assistant
indices. This is missing contextual evidence even with annotated raw coverage146/146.

Before the first reader call, for the narrowly supported ordinal-artifact query,
add missing immediately preceding raw USER turns of selected assistant artifacts
from the same original tenant/session. Use existing whole-source ledger budget6000;
never evict existing sources or truncate words. Existing packing/gates remain.
No later hidden source fetch: the amended complete pool is hashed/delivered from
call1, then fixed for any repair. If required bridge does not fit, ordinal proof
refuses. Program selection still requires contiguous actual raw roles/turns through
the requested artifact, one source session and unique literal section progression.
Test correct bridge, wrong tenant, missing/intervening assistant, max budget and
retained body/role/source provenance. All19 and17+2 acceptance stays unchanged.
This is provenance-neighbor recovery, not a new vector/reranker/RAG component.
