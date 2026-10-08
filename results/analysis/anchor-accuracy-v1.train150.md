# Memory anchor accuracy — `train150`

> Zero calls, no gold labels. Registered in `results/prereg-anchor-v2-offline-v1.md`.
> Reference: the only turn in the source session that carries one of the memory's
> specifics.

Checkable memories: **6,456**. v1 recomputed equals the stored anchor for 100.0%.

| anchor | accuracy |
|---|---:|
| stored | 87.8% |
| v1 recomputed | 87.8% |
| v2 | 96.5% |
| **v3** (v2 among the memory's own speaker's turns) | **95.6%** |

v1 wrong, v2 right: 611; v1 right, v2 wrong: 53 (v2 keeps 99.1% of v1's correct anchors).

- PASS — `v2_at_least_3_points_above_v1`
- PASS — `v2_keeps_99pct_of_v1_correct`

**Decision: switch new ingests to v2**

Fixed by v2:
- turn 1 -> 0: The user's LiFePO4 battery cell measures 205mm height, 174mm width, and 72mm thick.
- turn 10 -> 9: The assistant recommended L'Occitane, Dermalogica, The Body Shop, Burt's Bees, and Kiehl's as eco-friendly beauty brands.
- turn 7 -> 9: The assistant recommended Leesa Hybrid, Casper Wave Hybrid, and WinkBeds Plus as hybrid mattresses with cooling features.
- turn 5 -> 7: The assistant recommended M25 for Middle Eastern dishes in the Jaffa Flea Market.
- turn 1 -> 7: The assistant recommended Sabich Tchernichovsky for Israeli sandwiches in the Jaffa Flea Market.
