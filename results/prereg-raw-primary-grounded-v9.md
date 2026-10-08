# Fixed-model v9 — repair from the offending source and explicit event candidates

2026-10-02 before v9 outputs. Preserve v8 answers, grades/partial grades, usage,
execution inventory and source snapshot. V8's 19 real answers expose four concrete
failures: wrong first-song text attached to second-song source survives generic
review; a valid five-session total is rejected as grouped count and then five/5
prose mismatch; the next numbered move is selected two entries late despite an
explicit immediate next entry; a finished book is treated as ongoing to question
date despite its dated completion source being in the pool. Coupon location is a
contextual inference rather than an explicit redemption-place sentence. This is
development/regression evidence, not overall or unseen accuracy.

## Candidate changes, same models

Keep V8's source pool, typed provider schema, 6000 evidence budget and bounded two
calls. No new retrieval/database/framework and no question-id, named-book, store,
chess-piece or gold-answer hardcoding. Keep old variants/default policy unchanged.

1. When source/quote/value validation fails, show the ACTUAL offending source body
   in repair feedback, not just a source id and an error label. Preserve the full
   original pool and both drafts; do not move a mismatched quote to a different
   event simply because that text occurs elsewhere.
2. For duration review, expose a compact list of existing dated raw user sources
   containing start/end/completion event verbs. If the question quotes named
   objects, select literal matching names (including a title prefix before colon).
   Keep source id, session date and original text. This is an auditable view of
   sources already delivered, not a new retrieval or resolved-event assertion.
   A completed event cannot use the question date as a fabricated endpoint.
3. Numeric count operands with quoted quantities and units may canonicalize to
   sum only when all selected records qualify; run the SAME quote, role, quantity-
   unit, identity, exclusions and completeness validators before code computes.
   Bound equivalent English-number/digit spellings for lookup prose only after
   actual source value validation; never accept another numeric value or unit.
4. A narrow structural successor guard applies to a question literally asking
   after an explicit N. entry. Require an exact anchor in a raw assistant source,
   one matching session, and a unique subsequent assistant source beginning with
   N+1. Use original source provenance and turn order; reject a selected later
   entry and show the actual successor in review. If any prerequisite is absent
   or ambiguous, do not infer an entry or apply this guard.
5. Distinguish explicit facts from a place inferred from one consistent same-
   session context. Label such answers as likely, and cite actual context plus
   action evidence; conflicting locations must remain uncertain. An email coupon
   origin alone does not identify its redemption location. Do not force a gold
   store name or resolve contradictory unnamed wedding mentions by assumption.

## Gates and execution

Target tests cover wrong-source repair content, current/default compatibility,
numeric count vs members/mixed records/units/negatives, digit/word equivalence,
successor uniqueness/session/role/order, completed vs ongoing event candidate
feedback, and tenant isolation. Run targeted then full tests and Ruff.

Zero-call train150 replay must retain v8-r2 146/146 marked-raw coverage, no body
mismatch, median<=6000. dev100 only context preflight, never inspect its gold or
individual failures. New grounded-context-v9 and grounded-reader-v9 preview/output
namespaces; freeze code/config/data/index/manifest/store/model and registrations.

Use the SAME registered19 public train150 questions (9 known errors,10 originally
correct); <=38 reader calls before internal retries; gemini-3.5-flash-lite remains
answerer, gemma-4-31b-it remains judge. Save answers before grading. Record usage,
incomplete outcomes and ambiguity separately. Stop repeated judge500/503, preserve
reader outputs and never switch models. No blind claim of statistical significance
or improvement from this enriched set. Known source-supported selections must be
correct and original source-supported correct cases retained before dev adoption.
User has authorized all necessary work, no commits.

Only after those gates pass, use the previously registered dev100 3 repetitions
per arm and paired mean net>=4, per-type net>=-2, median<=6000. A genuinely independent
unseen-final set is still required for a new public accuracy number. No arbitrary
natural-language 100% accuracy guarantee; source contradictions remain limitations.
