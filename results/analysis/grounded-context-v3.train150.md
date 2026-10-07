# Grounded candidate offline replay — train150

Development evidence; stub reader, zero provider requests. Reach is not accuracy.

| arm | median context | p90 | max |
|---|---:|---:|---:|
| two_stage_raw_primary | 5683.0 | 5787 | 5935 |
| two_stage_raw_primary_grounded_v3 | 5975.5 | 5996 | 5999 |

All-gold-turn coverage: {'baseline': 0.7808219178082192, 'candidate': 0.910958904109589}; paired {'wins': 21, 'losses': 2}.

Offline gates: {'coverage_gain_at_least_2pp': True, 'no_content_mismatch': True, 'median_context_at_most_6000': True}.

Questions with context-budget omissions: 144.
Both arms use configs/fallback.yaml and its unchanged model assignments.
