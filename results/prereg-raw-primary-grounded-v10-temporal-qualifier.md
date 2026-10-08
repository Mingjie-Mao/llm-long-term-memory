# V10 local temporal qualifier repair — before external v10 calls

The provisional zero-call v10 selection replay exposed a code defect: full original
event text containing "tell me more about AI" falsely turns an exact dated duration
into "About8weeks". The inherited calculator checks any approximate word anywhere
in the source, rather than whether it modifies the selected temporal expression.
Retain grounded-selection-replay-v10 and its exact source snapshot; no external v10
reader/judge or completed v10 context preflight has occurred with that source.

V10 alone scopes about/around/approximately/roughly to the original date expression
they directly modify; genuine approximately dated phrases and the temporal parser's
approximate-day precision stay approximate. Old policies preserve their original
behavior. Add positive/negative date qualifier regression tests. Re-run the same
saved v9 selections under grounded-selection-replay-v10-r2 before final v10 context
preflight, freezing matching code and this addendum for the same registered19 calls.
No new model, retrieval or benchmark scope. This fixes temporal code, not model
accuracy; ambiguous event identity/year still cannot be fabricated from mention time.
