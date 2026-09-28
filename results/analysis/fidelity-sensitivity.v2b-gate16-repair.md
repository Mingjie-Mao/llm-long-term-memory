# Recall ruler sensitivity on `stores/v2b-gate16-repair.db`

> Zero calls. The published ruler is unchanged; this measures its literal-match
> bias and the effect of ShareGPT task sessions. It still cannot see paraphrase.

| sessions | specifics | literal (published) | normalised |
|---|---:|---:|---:|
| all | 2,338 | 60.5% | 63.4% |
| personal (non-ShareGPT) | 1,490 | 75.7% | 79.2% |
| ShareGPT task | 848 | 33.7% | 35.7% |

## Personal sessions by facet

| facet | specifics | literal | normalised |
|---|---:|---:|---:|
| date | 2 | 50.0% | 100.0% |
| duration | 140 | 72.9% | 76.4% |
| money | 89 | 66.3% | 85.4% |
| proper_noun | 673 | 73.3% | 73.3% |
| quantity | 413 | 88.9% | 95.9% |
| relative_time | 173 | 61.3% | 61.3% |

Examples counted lost literally and kept after normalisation: `800,`, `$800,`, `50,`, `$50,`, `hotel moderne saint germain

check`, `10,`, `$10,`, `2022,`, `8,`, `5,`, `3,`, `3,`.
