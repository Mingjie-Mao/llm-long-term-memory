# Memory anchor accuracy — `heldout100-anchor-v2`

> Zero calls, no gold labels. Registered in `results/prereg-anchor-v2-offline-v1.md`.
> Reference: the only turn in the source session that carries one of the memory's
> specifics.

Checkable memories: **4,285**. v1 recomputed equals the stored anchor for 88.9%.

| anchor | accuracy |
|---|---:|
| stored | 96.6% |
| v1 recomputed | 87.0% |
| v2 | 96.6% |
| **v3** (v2 among the memory's own speaker's turns) | **95.7%** |

v1 wrong, v2 right: 432; v1 right, v2 wrong: 24 (v2 keeps 99.4% of v1's correct anchors).

- PASS — `v2_at_least_3_points_above_v1`
- PASS — `v2_keeps_99pct_of_v1_correct`

**Decision: switch new ingests to v2**

Fixed by v2:
- turn 0 -> 1: The assistant recommended visiting Shakespeare's birthplace, the Royal Shakespeare Theatre, Anne Hathaway's cottage, and other family homes 
- turn 3 -> 1: The assistant recommended Seiko 5 Sports, Citizen BM8475-03E, and Fossil Grant for watches between $100 and $300.
- turn 0 -> 4: The most popular exhibit at the College Football Hall of Fame is the Game Day exhibit.
- turn 1 -> 3: The assistant recommended the Data Science Certification by DataCamp for the user.
- turn 2 -> 3: The assistant summarized a text about Lean Six Sigma (LSS) implementation in an R&D configuration control process.
