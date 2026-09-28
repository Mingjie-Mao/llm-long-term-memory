# Batch 8 readings by session kind

> DEVELOPMENT, zero calls, **exploratory and post hoc** — the split was chosen
> after seeing one per-kind table. Same reconstructions as
> `batch8-cohort-attribution-v1`.

## Recall by kind of session

| kind | A first 60 | B second 60 | C gate16 store |
|---|---:|---:|---:|
| `simulated` | 63.8% (74/116) | 65.3% (47/72) | 65.5% (858/1310) |
| `evidence` | 100.0% (7/7) | 47.4% (9/19) | 55.8% (48/86) |
| `ultrachat` | 75.0% (6/8) | 62.5% (5/8) | 55.3% (52/94) |
| `sharegpt` | 0.0% (0/3) | 8.7% (4/46) | 17.9% (152/848) |
| **all** | **64.9%** | **44.8%** | **47.5%** |
| **without `sharegpt`** | **66.4%** | **61.6%** | **64.3%** |
| `sharegpt` share of specifics | 2% | 32% | 36% |

## Composition, standardised

| cohort | own recall | its per-kind recall at C's composition | C's per-kind recall at its composition |
|---|---:|---:|---:|
| A first 60 | 64.9% | 42.4% | 63.3% |
| B second 60 | 44.8% | 44.0% | 48.6% |
| C gate16 store | 47.5% | 47.5% | 47.5% |

## Post-extraction processing

The gate16 baseline store's ingest record: 2 memories dropped as duplicates against 3,055 written. Deduplication cannot account for a gap of this size in C.
