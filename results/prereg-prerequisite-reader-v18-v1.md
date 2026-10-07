# Same-model page-selector mechanism gate — 2026-10-03

Development/regression on six exposed train150 questions, one fresh candidate execution,
not a new benchmark or paired accuracy estimate. Model selection remains frozen: answerer
Gemini3.5flash-lite, judge Gemma4-31b-it, existing extraction unchanged. No new extraction.

Question IDs are fixed before new provider outputs: b46e15ed (event identity/time),
9d25d4e0 (multi-session jewelry), gpt4_731e37d7 (workshop sum), gpt4_a1b77f9c
(named reading duration),51a45a95 (redemption lookup),eaca4986 (assistant-song control).
Three growth-risk examples and three selection controls. Source: public LongMemEval-S,
train150; all exposed. No dev100 per-item inspection. No significance/overall QA claim.
This enrichment tests the mechanism, not a plausible +4pp change in the population.

Preflight: role-paged-growth-v3 must deliver146/146 gold sources at1/2/4x; complete
USER archives/pages <=6000. Composite advice uses existing v15 coverage; its limits persist.
Actual provider test uses ORIGINAL history (1x). New larger-history QA remains unmeasured.
Audit every outbound raw turn against exact public source and every memory's same-user
public-session origin. Freeze sources/config/store/index/data/protocol before calls.

Ceiling40 nominal answerer calls (page-selection plus at most2 grounded reads/question),
6 judge calls,6000 estimated tokens perpage/final. All page/reader/provider retry token usage
and latency recorded. Bounded judge outer recovery3,180seconds total; quota/permanent/schema
stop. Save complete reader outputs before any grades. Resume exact saved readers, never
replace a failed or graded answer. Keep incorrect grades. No automatic daily wakeup.

Pass requires all6 judged correct, all personal-question annotated raw sources in final
reader context with exact hashes (not merely in review pages), no foreign sources, complete
review, context bound, and assistant advice control unchanged/correct. Failures are classified
by first-loss before a new protocol/candidate is created. v18 is not adopted as default even
if this small gate passes. Full paired regression and unseen-final remain separate.

Provider retry policy for this small gate: max_retries2, max_transport_retries2,
timeout180s unchanged. Outer grading recovery as above. Record every failed attempt.
No endless provider retries. Global learned quota remains in force.
