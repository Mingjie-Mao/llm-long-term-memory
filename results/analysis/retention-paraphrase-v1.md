# Delayed paraphrase retention v1

> Regression on 16 inspected development questions. No LLM calls or answer scores.

| query | delay | durable all | candidate any/all | top-20 any/all |
|---|---:|---:|---:|---:|
| original | 1 days | 16/16 | 16/16 / 16/16 | 16/16 / 16/16 |
| paraphrase | 1 days | 16/16 | 16/16 / 16/16 | 16/16 / 16/16 |
| original | 7 days | 16/16 | 16/16 / 16/16 | 16/16 / 16/16 |
| paraphrase | 7 days | 16/16 | 16/16 / 16/16 | 16/16 / 16/16 |
| original | 30 days | 16/16 | 16/16 / 16/16 | 16/16 / 16/16 |
| paraphrase | 30 days | 16/16 | 16/16 / 16/16 | 16/16 / 16/16 |

Cross-tenant retrieved hits: **0**.

Delayed dates are simulated ranking references on a reopened store, not elapsed wall-clock time; source-session recall is not answer accuracy. The configured recency weight is zero, so changing only the delay cannot change this ranking.

## Per-item top-20 source coverage at day 1

| question | original | paraphrase | gold source sessions |
|---|---:|---:|---:|
| `1f2b8d4f` | 2 | 2 | 2 |
| `3a704032` | 3 | 3 | 3 |
| `45dc21b6` | 2 | 2 | 2 |
| `561fabcd` | 1 | 1 | 1 |
| `57f827a0` | 1 | 1 | 1 |
| `58ef2f1c` | 1 | 1 | 1 |
| `6613b389` | 3 | 3 | 3 |
| `6a1eabeb` | 2 | 2 | 2 |
| `6e984301` | 2 | 2 | 2 |
| `778164c6` | 1 | 1 | 1 |
| `95228167` | 1 | 1 | 1 |
| `a82c026e` | 1 | 1 | 1 |
| `a96c20ee` | 2 | 2 | 2 |
| `d905b33f` | 2 | 2 | 2 |
| `gpt4_45189cb4` | 3 | 3 | 3 |
| `gpt4_7a0daae1` | 2 | 2 | 2 |
