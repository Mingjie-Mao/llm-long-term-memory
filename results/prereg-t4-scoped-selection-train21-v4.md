# Scoped selection v4: quoted-price repair and exact unchanged-case replay

Registered 2026-10-10 Australia/Sydney before v4 provider calls. Same models; no commit.

## Historical evidence and purpose

The all-fresh v3 fixed21 acceptance failed: original18/21, supported17/18,
faithful3/3, median6393. Preserve it and previous failures. Only supported failure
was airport train/taxi savings (`09ba9854`): original USER quotes around$10/around$60
both reached the first answer context. The answerer wrongly marked the planned
train quote OUTSIDE as though calculating purchases, discarded first-pass evidence
in fallback, and used generic assistant estimates. First loss S5, secondary fallback
composition failure. Fix eligibility for a **price comparison**, not a purchase count.

Candidate `SavingsRepairRunner`, prompt`memory-t4-scoped-selection-v3`, inherits all
unchanged v3 source/identity/time repairs. It binds two USER quotes to the requested
alternatives and route, requires one unambiguous price per alternative and identical
currency, computes original minus chosen with Decimal, preserves approximate wording.
Assistant estimates, hypothetical/negative clauses, different routes, conflicting
prices and historical queries without temporal selection cannot be operands.
No question IDs, gold or reference flags enter this runtime module.

## Zero-call equivalence proof and reuse (explicitly not 21 fresh answers)

LongMemEval-S exposed train150, same fixed21 `PILOT` and manifest, no resampling.
Split seed0, provider seedunsupported, temperature0, existing store/MiniLM384 index,
`configs/fallback.yaml`. Answerer`gemini-3.5-flash-lite`, judge`gemma-4-31b-it`,
extractor`gemini-3.1-flash-lite` unchanged and not called.

Before new calls, `t4-scoped-selection-train21-v4.replay-preflight-r2.json` checks
source/config/data unchanged from the v3 frozen inventory. For every fixed case it
uses archived responses only after **all logical generation arguments match exactly**,
including prompt, system, schema, model, temperature and output limit. It also requires
the full request sequence, final output and normalized source/calculation audit to
match. Eligibility is solely request/output/provenance equivalence, never prior grade.
Changed cases get new generations, even if their previous grade was correct.

The proof admits20 historical reader replays; only savings is changed and fresh.
The first proof (19 replays) is retained: a tuple/list representation mismatch in
an otherwise identical serialized audit was normalized using canonical JSON SHA.
One fresh price answer plus20 CPU replays is a component/regression acceptance,
not21 new model samples. Reader origins and exact historical row hashes are recorded.
Replay input/output tokens are zero for new billing; CPU replay time is not model latency.

For replay-eligible readers, reuse original/fidelity grades only with exact historical
reader SHA and unchanged question/gold/hypothesis/type/abstention/judge prompt, as
proved by frozen dataset/config/source and request hashes. Preserve failed original
gold judgments too. Replayed grades are labeled historical; no new judge calls.
The changed savings answer is freshly graded. No old grade may substitute for it.

## Fixed gate and resources

Keep21 original-gold judgments separately; all18 supported cases must pass and all3
faithful source-gap cases must pass, median actual supplied context<=6500. Gaps and
gold are identical to v3; no denominator/rule correction this round. Preserve original
scores for the ambiguous cases. No population or significance claim from this set.

The zero-call composition median is6393. Nominal new work: one reader (<=2 generations,
<=6 shared client attempts) plus one original grade (<=12 attempts with three durable
recoveries and180-second cooldown); zero new gap grades. Client retries2 and transport
retries2. All actual failures/tokens consume quota and are saved. No quota-reset bypass.

Freeze this preregistration, source/config/data, exact replay proof, historical readers
and grades, git SHA/Python and literal source snapshot in a fresh v4 namespace before
calls. Resume rejects changes. A mismatch during replay stops execution, not silently
spending unregistered calls. Failed/incomplete acceptance blocks full regression and
summary. Passing permits registering a new **all-fresh** dev100 comparison, original
t4 versus this candidate, three repeats each (600 fresh answers+600 fresh grades).
No replay is allowed in that comparison; recursive-summary calls remain blocked until
the full regression passes.
