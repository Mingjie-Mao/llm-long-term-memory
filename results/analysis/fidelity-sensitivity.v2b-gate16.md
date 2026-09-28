# Recall ruler sensitivity on `stores/v2b-gate16.db`

> Zero calls. The published ruler is unchanged; this measures its literal-match
> bias and the effect of ShareGPT task sessions. It still cannot see paraphrase.

| sessions | specifics | literal (published) | normalised |
|---|---:|---:|---:|
| all | 2,338 | 47.5% | 50.2% |
| personal (non-ShareGPT) | 1,490 | 64.3% | 67.8% |
| ShareGPT task | 848 | 17.9% | 19.2% |

## Personal sessions by facet

| facet | specifics | literal | normalised |
|---|---:|---:|---:|
| date | 2 | 50.0% | 100.0% |
| duration | 140 | 55.0% | 58.6% |
| money | 89 | 66.3% | 85.4% |
| proper_noun | 673 | 66.3% | 66.4% |
| quantity | 413 | 80.4% | 87.2% |
| relative_time | 173 | 24.9% | 24.9% |

Examples counted lost literally and kept after normalisation: `800,`, `$800,`, `50,`, `$50,`, `hotel moderne saint germain

check`, `10,`, `$10,`, `2022,`, `5,`, `6`, `3,`, `1`.
