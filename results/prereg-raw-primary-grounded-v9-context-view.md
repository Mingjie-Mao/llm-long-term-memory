# V9 context view clarification — before v9 outputs

The offending source bodies and dated event candidates will be shown first WITHIN
the existing ledger's source rendering, preserving every selected body exactly once.
Feedback lists their ids and checks; it must not duplicate large bodies on top of
the 6000 evidence budget. Session labels and actual turn/date provenance continue
to determine chronology/ordinal, regardless of display order. Keep both drafts in
the saved audit, but omit the first draft's confident rationale from the verification
prompt so it does not reinforce an erroneous reading. This implements the registered
source-specific repair and event candidate view with bounded evidence size.

The source renderer must remain <=6000, retain all selected sources, and fall back
to original rendering if a focus view cannot fit. No candidate retrieval or tenant
boundary changes. Test that focus rendering never evicts or repeats source bodies.
