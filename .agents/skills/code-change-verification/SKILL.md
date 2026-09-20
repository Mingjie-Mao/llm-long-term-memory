---
name: code-change-verification
description: Select and run proportionate verification after source, behavior, schema, test, API, or release-figure changes in this repository. Use whenever code behavior changed; do not use for read-only diagnosis or documentation-only edits with no executable effect.
---

# Code change verification

Run focused tests first, then expand according to blast radius. Reuse repository tests and
scripts; do not add a dependency merely to verify a change.

- Extraction: extraction, fidelity, schema, provenance, zero-yield, fingerprint/resume.
- Temporal: resolver/lifecycle, update/replacement, termination/removal, `as_of`,
  out-of-order ingestion, idempotence.
- Dedup: DUPLICATE/UPDATE/DISTINCT, tenant isolation, fingerprints, resume compatibility.
- Retrieval/composition: semantic/hybrid ranking, user/status filtering, fallback routing,
  hydration, context selection.
- Storage: SQLite, vector consistency, deletion, export, backup/restore, tenant isolation.
- API/MCP: auth, trusted identity, tenant isolation, budgets, delete/export, failure
  semantics.
- Evaluation: manifest/protocol safety, schema sent to providers, reproducibility,
  frozen/verifier invariants.

Always run the relevant targeted tests and Ruff lint/format checks. Run the full suite
when shared runner/store/protocol code changes or when targeted scope cannot bound the
impact. If a release figure changed, run `public-demo/check_release_figures.py` and the
release/site consistency tests.

Report commands and outcomes exactly. Separate passed, failed, skipped, and
environment-blocked checks. Never say the full suite passed when it was not run.
