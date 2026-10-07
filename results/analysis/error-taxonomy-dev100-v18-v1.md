# Error taxonomy — `paged-paired-dev100-v18-v1` (zero calls)

2026-10-07. Tool: `tools/error_taxonomy_dev100_v18.py`; rows:
`error-taxonomy-dev100-v18-v1.json`. Read after the run's gate was written; the run's
outcome is unchanged. **From here on dev100 is development evidence with per-item
failures read** — a change motivated by these items must be validated elsewhere.

Method: for each wrong row, first whether every gold turn (LongMemEval `has_answer`)
reached the final answer context verbatim; only then what the answer did. Baseline
context is replayed from the read-only store (excerpt turn counts match all 300 rows);
candidate context is its final grounded call's sources. Causes for the baseline were
then read by hand, one question at a time.

## Evidence delivery

| | wrong rows | questions ever wrong | always wrong (3/3) | wrong rows with **all** gold turns in context |
|---|---:|---:|---:|---:|
| baseline (v1) | 67 | 26 | 18 | 52 of 61 non-abstention rows |
| candidate (v18) | 77 | 34 | 18 | 72 of 72 non-abstention rows |

No wrong row of either arm is a retrieval miss or an extraction miss. Three baseline
questions had part of the evidence (9 rows); every other failure is downstream of the
context: reading, choosing, counting, dating or declining.

## Baseline (v1) — 26 questions, read by hand

| cause | questions | ids |
|---|---:|---|
| temporal: order of two events | 3 | gpt4_213fd887, gpt4_483dd43c, gpt4_65aabe59 |
| temporal: "N days/weeks ago" computed from the wrong date | 2 | gpt4_7bc6cf22, gpt4_85da3956 |
| temporal: picked the event outside the asked window | 2 | 0bc8ad93, gpt4_e414231f |
| temporal: relative dates not resolved, declined | 1 | gpt4_4ef30696 |
| temporal: arithmetic slip | 1 | gpt4_93159ced |
| multi-session count/sum wrong, evidence complete | 5 | 2788b940, d851d5ba, gpt4_e05b82a6, aae3761f, 681a1674 |
| multi-session, evidence incomplete | 3 | gpt4_59c863d7, gpt4_7fce9456, ba358f49 |
| knowledge update: older value chosen | 4 | 59524333, 945e3d21, c4ea545c, a1eacc2a |
| should abstain, answered | 2 | 09ba9854_abs, 6456829e_abs |
| single session: hedged, mis-compared, or judged strictly | 3 | 5d3d2817, gpt4_5501fe77, ec81a493 |

Two grades are arguable: `ec81a493` answers 500 (the gold) about a poster the user
described as from the album; `6456829e_abs` says the chili peppers were never mentioned
but also gives the tomato count. Neither is re-graded here.

## Candidate (v18) — why it lost

48 of its 77 wrong rows (20 questions, 13 multi-session) are "I do not know" with every
gold turn in context. In 24 of those 48 rows the baseline answered the same question
correctly in the same repeat. The refusals come from v18's own answer checks:

| check that refused | rows |
|---|---:|
| sum: unit / number not in quote | 14 |
| lookup: value not in answer / not quoted / quote invalid | 14 |
| duration: date expression not quoted / event date unresolved | 8 |
| count / sum / difference: operand not in review | 5 |
| count: member not in quote | 4 |
| difference: number and unit not bound | 3 |

The quote-binding checks meant to stop wrong computations reject correct ones far more
often than they catch errors.

## What this points at

The first loss is the reader, not retrieval: temporal resolution (9 of 26), aggregation
across sessions (5 + 3), and latest-state selection (4). v18's strict verification turned
reading errors into refusals. These are dev observations; they suggest where to look,
and nothing here is a measured improvement.
