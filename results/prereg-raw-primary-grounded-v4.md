# Fixed-model remaining-gap repair v4 — source-role budgeting

2026-10-02, registered before v4 replay or model calls. Keep all models/config unchanged.
Development/regression only. Exposed train150 may be inspected; no dev100 item tuning.

## Measured first loss

v3 actual context covers 133/146 fully marked train150 sources (91.1%), median5975.5,
remaining13. Gap trace puts missing user turns at fused ranks21–84, body38–121 tokens;
selected raw assistant advice consumes3008–4458 tokens while user turns consume552–1732.
The remaining failures have no missing stored source body; ranking/packing is the
identified loss. Full marked-turn reach is a proxy: repeated gold annotations need
not all be semantically necessary. Never equate this with answer accuracy.

Small repair: for questions explicitly about I/my/me/we/our and not assistant's past
advice, reserve75% of raw4000 budget for user turns in existing fused rank order, then
fill with other roles and remaining users if space permits. No new queries, embedding,
reranker or datastore; all sources remain namespace-bound. Keep v3 lexical recovery
under total6000 and its quote/unit/grouped-duration/same-pool checks. Advice-history
questions retain original packing. Route uses question text only, no benchmark type,
answers, gold flags or question ids. Output2048, temp0, seed unsupported.

Candidate two_stage_raw_primary_grounded_v4 / grounded_v4 / memory-grounded-v4.
Preserve v3 original source snapshot and prior negative/partial results unchanged.

## Verification and replay

Targeted tests must exercise advice/user routing, capacity, role isolation, unchanged
default packing and product/eval identity, then full suite/Ruff. Actual stub train150
namespace grounded-context-v4.train150. Report paired raw-primary baseline and v3,
gold-body consistency, total median/p90/max, raw-budget omissions and model hashes.
Offline advancement: coverage>=v3, no stored-text mismatch, median<=6000.
Desired diagnostic: cover all13; preserve residuals if this is not achieved. New ratio
selection after seeing outcomes requires a new registration; do not tune this label.
dev100 only context preflight in grounded-context-v4.dev100, no per-item gold analysis.
All offline provider request/token cost0; cached local embeddings and stub answers.

## Model validation boundary

Nine exposed train150 reasoning-error questions, candidate only once, at most18 reader
and9 judge calls before provider internal retries. Requires explicit outbound-data
authorization because automatic approval rejected that action earlier. Stop on recurring
judge500/503; preserve incomplete usage and rows. Freeze source/config/store/index/data
inventory before calls. The nine questions cannot estimate overall accuracy.
Reader gemini-3.5-flash-lite, judge gemma-4-31b-it, extractor gemini-3.1-flash-lite,
embedder sentence-transformers/all-MiniLM-L6-v2, dim384, configs/fallback.yaml.

Only a mechanically sound completed diagnostic permits larger comparison. Conditional
dev100 exact100, baseline raw-primary vs candidate, three repeats per arm, mean net>=4,
per-type net>=-2, median context<=6000; upper1200 reader/600 judge before retries.
This is an adoption rule, not significance or unseen final. No subsequent dev100
failure-based repairs. Missing runs produce INCOMPLETE, not a pass.

Complete evidence coverage alone cannot guarantee event selection, semantic item scope,
abstention or correct final answers. Conflicting/missing source facts must be identified,
not invented to match a reference. Fully resolving known supported errors requires
actual model outputs plus ordinary-correct regressions; no100% universal claim.
