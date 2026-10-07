# Stronger answerer — regression probe on flash-lite's correct answers

> Diagnostic, train150. Registered in
> `results/prereg-strong-answerer-regression-probe-v1.md`. Not an accuracy figure.

- regressions: **1 / 10** ['a89d7624']
- expected from flash-lite's own run-to-run flips: **0.59**
- projected net on train150: conservative **-7.7**, noise-adjusted **-0.2** (fixes counted: 5)
- **reading: regression risk is low — proceed to the full strong vs flash-lite comparison**

| type | sampled | regressed |
|---|---:|---:|
| knowledge-update | 2 | 0 |
| multi-session | 2 | 0 |
| single-session-assistant | 1 | 0 |
| single-session-preference | 2 | 1 |
| single-session-user | 1 | 0 |
| temporal-reasoning | 2 | 0 |
