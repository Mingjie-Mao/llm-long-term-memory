# Extraction rulers on `stores/v2b-gate16-repair.db`

> Manifest `results/manifests/v2b-gate16.json`. Read from the built store: after deduplication,
> supersession and any repair. No model calls.

- sessions: **780**, memories: **3,299** (4.23 per session)
- recall: **60.5%** — 1,414 of 2,338 specifics kept, 0.429 per memory
- support: **93.0%** of asserted specifics are in the conversation; 297 of 2224 checkable memories carry one that is not (13.4%)

| facet | recall |
|---|---:|
| date | 64.7% |
| duration | 69.7% |
| money | 61.9% |
| proper_noun | 56.9% |
| quantity | 62.8% |
| relative_time | 59.6% |

Paired against `results/analysis/store-fidelity.v2b-gate16.json` over 2,338 shared specifics: **+304 gained, −0 lost** (0 only there, 0 only here).
