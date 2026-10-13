# Raw-primary t4 (and t3) against v1 on heldout100 — registered gate

> Regression-class evidence on an exposed set. Rules as registered in
> `results/prereg-raw-primary-t3-t4-heldout100-v1.md`; not an accuracy claim.

| run | correct / 100 |
|---|---:|
| `two_stage_raw_primary.heldout100-t3t4-rep1` | 82 |
| `two_stage_raw_primary.heldout100-t3t4-rep2` | 81 |
| `two_stage_raw_primary.heldout100-t3t4-rep3` | 80 |
| `two_stage_raw_primary_t3.heldout100-t3t4-rep1` | 87 |
| `two_stage_raw_primary_t3.heldout100-t3t4-rep2` | 86 |
| `two_stage_raw_primary_t3.heldout100-t3t4-rep3` | 87 |
| `two_stage_raw_primary_t4.heldout100-t3t4-rep1` | 89 |
| `two_stage_raw_primary_t4.heldout100-t3t4-rep2` | 86 |
| `two_stage_raw_primary_t4.heldout100-t3t4-rep3` | 88 |

Mean over three runs: v1 81.00, t3 86.67, t4 87.67. **t4 mean net +6.67** (floor +4). Attribution: t3 - v1 +5.67, t4 - t3 +1.00 (contributes at +2).

| question type | n | mean net |
|---|---:|---:|
| knowledge-update | 15 | +1.67 |
| multi-session | 27 | +2.00 |
| single-session-assistant | 11 | +0.00 |
| single-session-preference | 6 | +0.67 |
| single-session-user | 14 | -1.00 |
| temporal-reasoning | 27 | +3.33 |

Median total context of t4: 6,147 (limit 6,500). Median output tokens: v1 48, t3 106, t4 114. Fallback second calls per run: v1 13.7, t3 7.7, t4 7.7.

## Rules

- PASS — `t4_mean_net_at_least_4`
- PASS — `no_type_loses_more_than_2`
- PASS — `median_total_context_at_most_6500`

**Decision: PASS**

Descriptive: questions whose mean moved up 19, down 6; exact sign test p = 0.0146 (no significance claimed).
