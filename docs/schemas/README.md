# Artifact schemas

These draft version-1 schemas define the contracts used by the planned product/research/artifact
separation. They are documentation-only while the registered v2 experiment is active; no current
artifact is moved or reclassified by adding them.

| Schema | Contract |
|---|---|
| `experiment-artifact.schema.json` | immutable run identity, fingerprints, files, outcome and evidence |
| `negative-results-index.schema.json` | discoverable decisions for negative, inconclusive and invalid runs |
| `legacy-path-map.schema.json` | byte-preserving old-path to new-path migration proof |

The schemas deliberately reject undeclared fields. Schema evolution therefore requires a new
`schema_version`, rather than silently changing the meaning of an existing audit record.

