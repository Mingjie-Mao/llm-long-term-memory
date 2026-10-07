# `paged-paired-dev100-v18-v1` — decision: FAIL

2026-10-07 Australia/Sydney. Registration: `results/prereg-paged-paired-dev100-v18-v1.md`.
Amendments, both written before the calls they allowed:
`results/amendment-paged-paired-dev100-v18-v1-judge-recovery.md` (used twice, both on
baseline rep2) and `results/amendment-paged-paired-dev100-v18-v1-reader-transport.md`
(one restart, candidate rep3). Gate: `results/analysis/paged-paired-dev100-v18-v1.gate.json`;
report: `results/analysis/paged-paired-dev100-v18-v1.report.json`.

**Exposed dev100 regression evidence — not an unseen or final accuracy.** All 600
answers and 600 grades are fresh answers of this run; 280 grades reused an identical
saved judge request under the registered cache rule.

## Gate

| rule | value | needed | holds |
|---|---:|---:|:---:|
| mean net, candidate − baseline (per 100) | **−3.33** | ≥ +4 | no |
| worst question type, mean net | multi-session **−3.33** (temporal −2.00, at the limit) | ≥ −2 | no |
| median candidate source context | 5,978 | ≤ 6,000 | yes |

Outcome: **fail**. Per the registration, the conditional recursive-summary cost
comparison does not start, and the candidate is not changed on dev per-item failures.

## Runs (correct of 100)

| | rep1 | rep2 | rep3 | mean |
|---|---:|---:|---:|---:|
| baseline `two_stage_raw_primary` | 78 | 80 | 75 | 77.67 |
| candidate `two_stage_raw_primary_grounded_v18` | 74 | 73 | 76 | 74.33 |

Questions whose mean moved: 15 up, 21 down (no significance claimed).

| type | n | mean net |
|---|---:|---:|
| multi-session | 27 | −3.33 |
| temporal-reasoning | 26 | −2.00 |
| single-session-preference | 6 | −0.33 |
| single-session-assistant | 11 | 0.00 |
| knowledge-update | 16 | +1.00 |
| single-session-user | 14 | +1.33 |

## Cost (answerer, three runs per arm)

| | calls | per question | input tokens per question | output tokens per question |
|---|---:|---:|---:|---:|
| baseline | 336 | 1.12 | 6,740 | 56 |
| candidate | 1,203 | 4.01 | 31,372 | 2,068 |

The candidate's paged archive review (83 personal questions per run, all complete)
used about 3.6x the calls and 4.7x the input tokens of the baseline, and did worse.
End-to-end latency was not captured in this namespace.

## What it says

The plain raw-primary arm (v1: memory state + 4,000 tokens of question-found turns)
remains the candidate. Reading the whole personal archive in pages with a grounded
selector cost more, was less stable between runs, and lost most on multi-session and
temporal questions — the two types where it was meant to help.
