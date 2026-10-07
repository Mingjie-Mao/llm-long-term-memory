# Fixed-model v8 — bind fact answers and review selection

2026-10-02 before v8 outputs. Preserve v7 ungraded19 answers/inventory/source snapshot
and usage25requests225075tokens2failures. v7 mechanically solves taxi/sculpting/charity/
racket/attendance selections, but lookup still copies first-song notes while citing
second song, chooses old eggs despite a later quantity, skips chess next move and
counts a vague cousin-wedding mention as a distinct fourth wedding. Three-book first
selection contains six correct event quotes, but mislabels derived-memory id E2 instead
of raw E22 and omits review of one endpoint; second rewrites date words as calendar
dates. No dev100 adoption run; original10 regression has failures. No accuracy claim.

## Changes

Same typed schema/model and evidence pool. Every question gets one bounded same-model
verification pass: review the previous draft and code result against sources for
identity, chronology, duplicate mentions, attribute/argument roles and missing facts.
At most2 reader calls; code/schema/config/namespace frozen first. Keep both drafts.
No hidden gold/model swap or adaptive third call. For factual lookup/current_state,
require selected operand objects with source/quote/value/key; values must occur in the
actual source and in the final prose. Recommendation/suggestion/advice synthesis may
use prose grounded by reviewed sources. Review prompts explicitly require the actual
latest state, immediately following step, or requested ordinal, rather than first match.
Different mentions alone do not prove different events; resolve names/context and
keep uncertainty when raw identity/dates conflict. A vague unnamed reference cannot
automatically add a distinct named event.

Safe mechanical canonicalization only: a quote mistakenly attached to derived memory
may rebind to a unique matching raw turn with the EXACT same session+turn provenance;
never across tenants/events. Selected operands themselves are evidence review entries
when not explicitly excluded/uncertain. A model-written calendar date may be replaced
by actual full source text only when the full original source has one independently
resolved date and that date equals the supplied expression; undated/ambiguous sources
still refuse. Preserve canonical source id/body/expression and original provider draft.
Detailed repair feedback identifies offending source/value/quote and prints canonical
calculator citations/date/result rather than a bare refusal label.

## Verification

Target rebinding uniqueness/provenance/tenant isolation, unresolved dates, false numeric
quotes, lookup value binding, synthesis exemption, exclusion, two-pass usage/model
bound, latest/sequence review instructions, default/old-policy compatibility, CLI and
product/eval parity. Full suite and Ruff. Offline train150 grounded-context-v8 coverage
>=v7(146/146), no body mismatch, median<=6000. dev100 context-only separate namespace;
no gold inspection/tuning. Zero-call canonicalization replay of saved v7 operands may
diagnose mechanical repair, not accuracy. Keep all failures and stop if gates fail.

Same19 public train150 questions, nine errors+existing ten original-correct sample,
new grounded-reader-v8 label, <=38 reader calls before internal retries, then original
Gemma judge on saved outputs when functioning. Authorizations unchanged: all necessary
work, same previously-approved19 scope, no commits. Do not edit prior evidence.
Record sources/config/data/index/store/tool SHA, environment/git/model/temp0/seed
unsupported, usage and incomplete outcomes. Review supported selections individually,
distinguish ambiguous/contradictory raw evidence from S5. Ordinary-correct source-
supported regressions must retain correctness; no nine-question general accuracy claim.
Recurrent original judge500/503 stops grading, preserves all reader work; no model switch.
Full dev100 only after valid diagnostic/regression and functioning judge,100x3/arm vs
raw-primary, mean net>=4, per-type net>=-2, median<=6000, up to1200 reader/600judge
before retries. No per-item dev tuning; independent unseen final still required.
