# Small t1/t2 probe — registered reading

> Diagnostic (`results/prereg-t-probe-v1.md`); one run per arm. Not an accuracy
> figure; the heldout100 registrations decide t1 and t2.

## train150, out of design

| | v1 rerun | t1 | t2 |
|---|---:|---:|---:|
| correct on temporal failures (of 8) | 0 | 1 | 5 |
| correct on count/KU failures (of 11) | 2 | 5 | 4 |
| wrong on controls (of 12) | 0 | 1 | 1 |

## Readings

- t1_moves_its_target: **False**
- t2_moves_its_target: **True**
- t1_harm_signal: **False**
- t2_harm_signal: **False**
- t1 move +1, t2 move +2 (floor +2); harm vs v1 rerun {'t1': 1, 't2': 1}
- **proceed to heldout100: False**

## dev100, design set (reported only)

| arm | questions | correct | v1 expected (3 runs) |
|---|---:|---:|---:|
| t1 | 9 | 4 | 2.33 |
| t2 | 18 | 9 | 3.0 |
