# Fixed-model remaining-gap repair v5 — actual asked subject

2026-10-02, registered before replay/model calls. Same model/config/budgets as v4.
Candidate two_stage_raw_primary_grounded_v5 / grounded_v5 / memory-grounded-v5.
Keep all v4 artifacts/source snapshot; no overwritten history. Exposed train150 only
for item-level diagnostics; dev100 context-only, no per-item gold tuning.

## Observed failure and minimal repair

v4 covers145/146 fully marked train questions (99.3%), 13 wins/1loss against v3,
median5925,max5999. Remaining58470ed2 is an assistant's literary explanation.
Its question introduces "I ... our previous conversation", then asks "what did Borges
say ...". Treating incidental I/our as a request for user evidence suppressed the
long assistant source. Fix source routing using the explicit actual question clause
where present (what/where/when/who + auxiliary), rather than introductory pronouns.
Keep current personalized recommendations and existing past-advice exemptions.
No author/question-id-specific rule, new query, model, index or budget tuning.

## Verification

Targeted route regressions: quoted authors/third-party statements, current personal
recommendations, explicit personal questions and assistant past advice. All v4 role,
tenant, quantity/unit, grouped-duration and bounded-repair checks retained, followed
by full suite and Ruff. Actual train150 stub replay namespace grounded-context-v5.train150:
coverage>=v4, no stored-text mismatch, median<=6000. Goal146/146 if source-supported;
retain failure rather than silently changing this registered rule. Report comparisons
against v4/v3/raw-primary; full marked-turn coverage is not answer accuracy.
dev100 context-only in separate grounded-context-v5.dev100 namespace.
Offline provider requests/tokens0; local cached MiniLM embeddings, temp0, no seed support.

## Model and adoption verification

Pending explicit authorization for outbound public benchmark questions/evidence remains
required after prior automatic rejection. If authorized, nine exposed train150 reasoning
errors once, candidate only, newv5 label, <=18 reader/9 judge before internal retries.
Keep gemini-3.5-flash-lite answerer, gemma-4-31b-it judge, gemini-3.1-flash-lite extractor,
sentence-transformers/all-MiniLM-L6-v2 embedder(dim384), configs/fallback.yaml unchanged.
Freeze code/config/store/index/data/environment inventory first, preserve usage/errors,
stop on recurrent judge500/503, mark partial as INCOMPLETE. This nine-question diagnostic
cannot establish general accuracy or fully correct semantic item selection.
Full dev100 only after mechanically sound completed diagnostic: exact100questions,
three repeats/arm against raw-primary, mean net>=4, per-type net>=-2, median<=6000;
upper1200 reader/600 judge before retries. No per-item dev100 tuning. This is development
adoption, not statistical significance or unseen-final evidence.

Completion still requires actual grounded event/operand selections and ordinary-correct
regressions, not just retrieval coverage or unit tests. Missing/ambiguous/conflicting
facts must be represented as such, not invented to force a reference match. No universal
100% guarantee is implied by resolving known supported failures on an exposed corpus.
