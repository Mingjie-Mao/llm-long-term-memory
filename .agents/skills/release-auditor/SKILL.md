---
name: release-auditor
description: Audit benchmark, test, coverage, documentation, and release-manifest consistency before publishing or changing reported figures. Use for releases, README/project-report/evaluation numbers, release.json, CI counts, or generated figures; do not use for ordinary feature development with no release claim.
---

# Release auditor

Inspect `README.md`, `README.zh-CN.md`, `docs/PROJECT_REPORT*`, `docs/EVALUATION*`,
configs, `public-demo/site/release.json`, generated figures, CI, test/coverage outputs,
experiment artifacts, git SHA, and version metadata relevant to the claim.

Find the authoritative source before proposing an edit. Do not reconcile a mismatch by
blindly changing the visible number. Confirm that accuracy, context, p-value, sample,
model, and split belong to the same experiment, and that development evidence is not
presented as final. Distinguish passed and skipped tests.

Reuse `public-demo/check_release_figures.py`, `public-demo/check_site.py`, release
manifest tests, and artifact verifiers. Report blocking inconsistencies, non-blocking
inconsistencies, verified numbers, the source artifact for every major figure, and a
release-readiness verdict. Never modify frozen/sealed history.
