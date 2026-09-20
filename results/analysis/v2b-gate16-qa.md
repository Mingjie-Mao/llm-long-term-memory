# v2b gate16 — paired QA analysis

This is a targeted mechanism gate, **not** a representative accuracy estimate.

| Endpoint | Frozen batch15 | Batch8 | Wins | Losses | Net | Exact p |
|---|---:|---:|---:|---:|---:|---:|
| Memory-only proxy | 8/16 | 9/16 | 4 | 3 | +1 | 1.0000 |
| Final fallback | 9/16 | 10/16 | 3 | 2 | +1 | 1.0000 |

The archived memory-only value is a proxy: a row counts only when its final answer was correct and no raw fallback ran; old pre-fallback drafts were not saved.

Raw fallback triggered on **4/16** rows.

The first run is directionally positive but ambiguous. Per the preregistration, only the **9** discordant rows are eligible for one repeat; there is no basis yet for an automatic 48-question expansion.

## Discordant-row repeat

- Memory-only: 3 wins / 3 losses (net +0); first/repeat agreement 8/9.
- Final fallback: 4 wins / 2 losses (net +2); first/repeat agreement 8/9.

**Decision: `stop_no_48`.** The discordant-row repeat did not reproduce a positive memory-only net; the preregistered promotion rule requires memory-only improvement.
The repeat set was selected from first-run disagreements, so no inferential p-value is attached to it.
