# Small t3 probe — registered reading

> Diagnostic (`results/prereg-t-probe-v2.md`); one run per arm; v1 rerun and t2
> rows from probe v1. Not an accuracy figure.

| train150 | v1 rerun | t2 | t3 |
|---|---:|---:|---:|
| correct on 8 temporal failures | 0 | 5 | 5 |
| correct on 11 count/KU failures | 2 | 4 | 5 |
| wrong on 12 controls | 0 | 1 | 1 |

Moves against the v1 rerun: {'temporal': 5, 'count_ku': 3, 'harm': 1}

- t3_moves_temporal: **True**
- t3_moves_count_ku: **True**
- t3_harm_signal: **False**
- **register t3 on heldout100: True**
- t3 answer modes on train150: {'count_time_notes': 8, 'latest_only': 5, 'time_notes': 9, 'count_notes': 9}
- dev100 design set (reported only): 11 / 18 against v1's expected 3.0
