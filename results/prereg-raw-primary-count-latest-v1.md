# Raw-primary t2 (latest value wins; counts item by item) against t1 — heldout100, three runs per arm — pre-registration

Date: 2026-10-07 (Australia/Sydney). Written before any answer or judge call. Runs on
the same set and in the same schedule as `results/prereg-raw-primary-time-notes-v1.md`
(t1 against v1), as its third arm. Same classification: **a development comparison on an
exposed set, recorded as regression evidence**; nothing here is an unseen result.
Registration hashes are those listed in the t1 registration.

## Why

`results/analysis/error-taxonomy-dev100-v18-v1.md`, v1's 26 failed dev100 questions,
all with the gold turns in context except where noted:

- **knowledge update, 4** — the older value given (gym at 7 pm, later 6 pm; yoga twice a
  week, later three times; "no change" to a gym routine that had increased; a restated
  running total of short stories added to an earlier one);
- **multi-session count or sum, 5** with complete evidence (+3 incomplete) — items missed
  (4 classes of 5; 7–8 rides of 10), items outside the question added ($8,750 of
  charity for $3,750; 4 re-watched films for 2), one leg mis-summed.

## What t2 is

`two_stage_raw_primary_t2` = t1 with answer policy `v2_cl` (prompt
`memory-aware-v2-cl`, `src/llm_long_term_memory/answering.py`); everything else is t1's.

1. **Every question**: v2's system prompt plus one rule — when conversations give
   different values for the same thing, the most recent conversation's value is current
   and earlier ones are history, unless the question asks about an earlier time
   (the latest-wins update that Zep and Mastra apply in their state layers, here stated
   to the reader because the excerpts carry both values).
2. **Questions asking for a count or total** (`asks_for_aggregate`: "how many", "how
   much", "total", "combined", "altogether", "number of"; "how many … ago" excluded as a
   date question): the reader first writes one note per candidate item, from every
   conversation, with its date and the user's words, each marked COUNTS, REPEAT or
   OUTSIDE (another period, someone else's, planned not done, only suggested); then
   counts or adds only COUNTS. Structured output with the notes first
   (`NotedAnswerVerdict`), and 1,024 output tokens instead of 512 for these questions
   only (512 truncates the notes JSON, as measured for `v2_notes`). Chain-of-Note
   (LongMemEval's reading strategy) restricted to the questions that need enumeration.

Retrieval, context, fallback trigger, models: unchanged from t1. The fallback second
call uses the latest-value system prompt without the notes instruction (it has no schema).

## Evidence before this run (zero calls)

`asks_for_aggregate` selects 72 / 150 train150 questions (33 multi-session, 15
knowledge-update), 38 / 100 dev100, 37 / 100 heldout100 (21 multi-session, 8
knowledge-update). All five complete-evidence count failures on dev100 are selected; the
three knowledge-update failures it does not select receive the latest-value rule. Unit
tests: `tests/test_count_latest.py` (classification, per-question switching of prompt,
schema and output limit, reset between questions). Full suite at registration: 2,389
passed, 0 failed, 0 skipped. This policy has never been measured with a call; there is
no estimate of its effect.

## Resolution

As in the t1 registration: three runs per arm, per-question mean correctness. Multi-
session and knowledge-update together are 30–40 heldout100 questions; on dev100 v1 had
2 unstable questions among its 43 of these types (8 of 100 overall), so +2.0 on the
combined types is well above what no change produces there; it is kept equal to t1's
floor rather than lowered to fit that estimate. No significance is claimed.

## Decision rules (t2 against t1)

t2 passes if **all** hold (`tools/raw_primary_gate_t2.py`):

1. multi-session + knowledge-update mean net **≥ +2.0**;
2. overall mean net **≥ 0**;
3. no question type's mean net below **−2.0**;
4. median context of t2 **≤ 6,000** tokens.

Reported, not deciding: per-run scores, per-type nets, median output tokens (t2 will spend
more), fallback second calls, sign test. Also reported, descriptively: t2 against v1.

A **pass** makes `v2_cl` part of the next candidate, to be confirmed with the other
changes on a set not yet used. A **failure** is recorded; the policy is not tuned on
heldout100. If t1 fails its own gate, t2's comparison against t1 still measures the
policy, and the next candidate is v1 + `v2_cl` alone, confirmed elsewhere.

## Cost

A third arm: about 340 more answer calls (≈ 1,020 for all three arms) and 300 more
judge calls (900 in all) — about three answerer quota days. Same deadline as t1:
**2026-10-12**, else recorded as incomplete.

## Reproduction

The loop in the t1 registration, which runs all three arms per repeat, then:

```bash
python tools/raw_primary_gate_t1.py
python tools/raw_primary_gate_t2.py
```

## Withdrawn before any call — 2026-10-07

No run of this registration was started. Two small probes on train150
(`results/t-probe-v1-decision.md`, `results/prereg-t-probe-v2.md`) showed that the date
notes alone did not move temporal failures and that the notes path did; the plan is
replaced by `results/prereg-raw-primary-t3-t4-heldout100-v1.md` (v1, t3, t4 on heldout100).
