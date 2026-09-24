# Batch 8 cohort attribution v1

> Retrospective development analysis; zero model calls. Historical results unchanged.

| cohort | stage | sessions | specifics kept | recall | memories/session | median source chars |
|---|---|---:|---:|---:|---:|---:|
| A_first60_extraction | archived batch8 extraction | 60 | 87/134 | 64.9% | 3.72 | 12838 |
| B_second60_extraction | pilot baseline extraction | 60 | 65/145 | 44.8% | 5.85 | 9270 |
| C_gate16_store | built store after deduplication | 780 | 1110/2338 | 47.5% | 3.92 | 9776 |

## Facet composition

| facet | A stated/kept | B stated/kept | C stated/kept |
|---|---:|---:|---:|
| quantity | 54/36 | 62/23 | 852/416 |
| duration | 14/10 | 14/5 | 178/82 |
| money | 1/0 | 16/10 | 97/60 |
| date | 0/0 | 0/0 | 17/3 |
| relative_time | 12/5 | 13/2 | 183/44 |
| proper_noun | 53/36 | 40/25 | 1011/505 |

## Cohort and stage limits

- Session-ID overlap `A_first60_extraction__B_second60_extraction`: 0.
- Session-ID overlap `A_first60_extraction__C_gate16_store`: 1.
- Session-ID overlap `B_second60_extraction__C_gate16_store`: 1.
- `C_reweighted_to_A_facet_mix`: 46.9% over 134/134 specifics in shared facets.
- `C_reweighted_to_B_facet_mix`: 48.1% over 145/145 specifics in shared facets.

The three readings use different sessions. C has no archived pre-dedup extraction for those same sessions, so cohort and pipeline effects cannot be causally separated from these artifacts.

## Historical-total checks

- A_matches_history: PASS
- B_matches_history: PASS
- C_matches_history: PASS
