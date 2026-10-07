# Fixed-model v7 — typed records and source-derived dates

2026-10-02, before v7 outputs. Keep v6 ungraded19 answers, usage (36 requests,
320527 recorded tokens,2 failures), request preview, execution inventory and source
snapshot. v6 not adopted: observations were frequently prose instead of encoded JSON,
apostrophes overescaped in operand strings, and resolved calendar dates replaced
literal "today". The model often identified correct operands in reason, but mechanical
record decoding refused. Original-correct cases also refused; preserve regression.

## Minimal repair

Replace JSON-inside-strings with typed provider record objects in v7 only. Use simple
object fields and arrays, no unions/extra-properties/minItems constraints. Check local
SDK conversion and actual provider acceptance before expanding any run. No schema/model
fallback on error; keep old schemas and behavior. Observations select source id,
interpretation and include/exclude/context/uncertain. Program attaches actual source
body and provenance, avoiding redundant model quote copying. Operand objects retain
source/quote/value/unit/key; optional empty quote means the complete actual source body.
For dates code resolves the actual quote against its source chat date and preserves the
original expression; the model no longer generates a resolved date or separate time
field. Multiple/undated/ambiguous expressions still refuse; copied narrow clauses are
allowed. Numeric units/values still bind to actual text and excluded items cannot be used.
Named-member counts may discard a redundant descriptive unit label; numeric quantities
still require sum. A unit may be the head noun in an adjacent short noun phrase (e.g.
5 tomato plants), without crossing another number, preposition, verb or different
measurement unit (e.g. gallon). Preserve the actual quoted phrase in the audit.
The observation's same-source proof covers a narrowed operand clause, not necessarily
an exact repeated copy of that clause. Keep v6 session grouping, reason-before-answer,
identity/time/scope checks and bounded two calls. Add general relation-role instructions
to distinguish a fact's origin from where it was used; contextual inference must have
user/session support, not merely an assistant's example. No question-id/gold rule.

## Frozen validation

Test typed SDK-compatible schema, source-only record binding, original relative words,
ambiguous multi-date rejection, unit/quantity binding, scope/exclusion, copied quote
validation, historical endpoints, tenant isolation, legacy compatibility, identical
product/eval requests and CLI build. Targeted then full suite and Ruff. Offline train150
namespace grounded-context-v7: coverage>=v6 (146/146), no body mismatch, median<=6000;
dev100 context-only namespace separately. No overwrite/edits to v6 or frozen evidence.

Registered reader-only diagnostic: same19 exact questions (9 known errors + existing
10 original-correct manifest), same model once/item, <=38 reader calls before internal
retries. Authorizations include user's "全部授权，继续完成所有。不要提交", and previously
approved identical19-question scope. Persist outputs before grading. Copy schema/config/
data/store/index/tool SHA plus model/environment/git SHA/temp0/seed unsupported; new
grounded-reader-v7 namespace. New requests may be reviewed locally before approved calls.
No result is benchmark accuracy until graded. Source-audit exposed train selections,
document raw ambiguity separately. Known supported selections must be correct and no
mechanical refusal of a correct selection; all source-supported original-correct answers
retained before adoption comparison. Failures require new namespace/version.

Original Gemma judge still required for benchmark scores. Grade the saved answers once
if functioning; recurrent500/503 stops and preserves answer/usage/partial grades without
rerunning readers. No model switch. Full dev100 only after valid diagnostics/regressions
and functioning judge: exact100,3 repeats/arm vs raw-primary, mean net>=4, per-type net>=-2,
median<=6000; <=1200 reader/600 judge before retries. No per-item dev100 tuning or unseen
claims. All artifacts remain local and uncommitted. A new independent final is required
for an official improved accuracy, no universal100% guarantee.
