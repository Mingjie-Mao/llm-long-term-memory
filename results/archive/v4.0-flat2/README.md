# v4.0-flat2: retrospective evidence closure

Archived on 2026-09-12 from existing development results. Outcome: **exploratory_not_promoted**.
No provider calls or regrading were performed. This archive preserves original bytes; it does not reconstruct a missing historical source snapshot.

| operation | v4.0-flat | v4.0-flat2 | correct-count difference |
|---|---:|---:|---:|
| comparison | 19/33 | 16/33 | -3 |
| count | 5/30 | 8/30 | +3 |
| current_state | 42/49 | 36/49 | -6 |
| duration | 21/30 | 20/30 | -1 |

These are saved verdicts over store-derived development probes, not benchmark accuracy. The archive deliberately has no pooled headline.

Gate 0: PASS: identical retrieval and question identity on 142 rows.

The old one-probe regression rule would flag differences below -1. This is only a retrospective comparison: small strata and observed run disagreement make that rule a weak decision instrument. It does not establish a causal regression. Count remains 8/30; these records do not establish a reliable benefit warranting promotion.

Historical source/configuration identity, per-run usage and observed routes were not captured in these rows. Equal variant names and retrieval do not prove byte-identical prompts or runtime configuration. Consequently, disagreement cannot be attributed exclusively to provider sampling. Current code and current preregistration text must not be passed off as their historical versions.

The copied split shows all 142 rows belong to development and none to the 87 held-out probes. `conclusion.json` binds each copied evidence file by SHA-256. The hash proves preservation at audit time, not that a preregistration was frozen before the original run.

Next action: use the existing enumeration diagnosis and v4.2 candidate, validate the statistical assumptions and repeat scheduling, then freeze source/configuration/store and request budget before a prospective run. No hidden-set run is authorised by this archive.
