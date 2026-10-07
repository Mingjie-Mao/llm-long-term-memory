# Memory anchor accuracy — `dev100`

> Zero calls, no gold labels. Registered in `results/prereg-anchor-v2-offline-v1.md`.
> Reference: the only turn in the source session that carries one of the memory's
> specifics.

Checkable memories: **4,272**. v1 recomputed equals the stored anchor for 100.0%.

| anchor | accuracy |
|---|---:|
| stored | 87.8% |
| v1 recomputed | 87.8% |
| v2 | 96.2% |
| **v3** (v2 among the memory's own speaker's turns) | **95.2%** |

v1 wrong, v2 right: 395; v1 right, v2 wrong: 35 (v2 keeps 99.1% of v1's correct anchors).

- PASS — `v2_at_least_3_points_above_v1`
- PASS — `v2_keeps_99pct_of_v1_correct`

**Decision: switch new ingests to v2**

Fixed by v2:
- turn 3 -> 5: The assistant recommended Bell Helmets, Specialized, and Giro for bike helmets.
- turn 3 -> 5: The assistant recommended Cygolite, Planet Bike, and NiteRider for bike lights.
- turn 3 -> 5: The assistant recommended Leatherman, Gerber, and Park Tool for bike multi-tools.
- turn 0 -> 1: The assistant recommended Tofuya Ukai in Shinjuku for traditional Japanese cuisine.
- turn 0 -> 1: The assistant recommended Kyubey Ginza in Shinjuku for traditional Japanese cuisine.
