# Grounded candidate offline replay — train150

Development evidence; stub reader, zero provider requests. Reach is not accuracy.

| arm | median context | p90 | max |
|---|---:|---:|---:|
| two_stage_raw_primary | 5683.0 | 5787 | 5935 |
| two_stage_raw_primary_grounded_v10 | 5815.5 | 5925 | 5999 |

All-gold-turn coverage: {'baseline': 0.7808219178082192, 'candidate': 1.0}; paired {'wins': 32, 'losses': 0}.

Offline gates: {'coverage_gain_at_least_2pp': True, 'no_content_mismatch': True, 'median_context_at_most_6000': True}.

Questions with context-budget omissions: 150.
Both arms use configs/fallback.yaml and its unchanged model assignments.
