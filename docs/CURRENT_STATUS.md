# Current status — 2026-08-31

This is the one current-state page. The long reports and execution log are evidence
and history; older entries in them are intentionally not rewritten.

## Experiment

| item | state |
|---|---|
| v1 heldout100 | complete: registered single shot 70/100; repeats 71 and 73 |
| v2 train150 | complete: 7,180/7,180 sessions; 18,519 memories |
| v2 selected context | `mean`, radius 1, cap 30; 95.3% Top-3 and assembled source recall |
| v2 dev100 ingest | complete: 4,791/4,791 terminal; 12,436 memories; state checker reports `COMPLETE`, `issues: []` |
| v2 post-ingest freeze | captured as `v2-candidate` and preserved at `eecb736`; current preflight passes |
| v2 dev validation | in progress: 1,185/1,500 sealed rows; all three repeats of `flat20`, `coherent-auto`, and `coherent-oracle` are complete; `naive_rag` has two complete repeats plus 85/100 in repeat 3; `memory-only` has not started |
| v2 test100 | sealed; no v2 result exists yet |

The exact sealed order remains in [`results/v2-runbook.md`](../results/v2-runbook.md).
That hash-bound protocol record intentionally is not edited while validation is active,
so its old progress annotation is historical rather than the live counter. The next safe
action is to resume the identical 15-run validation after the Pacific-midnight provider
reset (2026-08-31 17:00 AEST; use a small safety margin). The 315 remaining rows should
fit inside one fresh day for both the answer and judge models, absent provider failures.
No dev100 score exists until all
arms complete; do not inspect sealed rows, replace
the checkpoint, add `--fresh`, or change the registered order.

The current sealed checkpoint consists of eleven complete repeats and 85 rows in
`naive_rag.rep3`. Only file counts and checksums have been inspected; individual rows and
scores remain sealed. The most recent committed immutable count-and-checksum snapshot is
[`results/audit/dev100-validation-checkpoint-20260829T174608Z.json`](../results/audit/dev100-validation-checkpoint-20260829T174608Z.json),
which is historical and therefore lower than the live append-only count above.
A stalled provider request is interrupted only after ten minutes without a new sealed row,
then the same command resumes from the append-only checkpoint.

### Review findings that must be handled at the phase boundary

- The aggregate produced by the registered five-arm run binds 30 sealed row/usage files,
  but `load_dev_decision` currently validates only the 18 files from the three
  decision-making arms. Add the two report-only baselines to that inventory and add a
  five-arm regression test after dev validation completes and before the test100 freeze.
- Do not make that repair while validation is incomplete: `src/` and `scripts/` are part
  of the `v2-candidate` source hash, so changing them now would invalidate the remaining
  checkpoint. The validation preflight still passes against the frozen source.
- The current offline suite has 560 passing tests, Ruff reports no violations, and core
  package line coverage is 83%. Including one-off research scripts lowers the combined
  number to 62%; this is not treated as a reason to write low-value tests for retired
  analyses.

## Product code

The post-freeze repair branch at `06b567e` fixes the default REST/MCP write composition, uses the
real batch extractor through a single-turn adapter, serializes access to the shared
SQLite/index resources, avoids mutating a shared answer runner, externalizes session
ids consistently, and makes `/healthz` a readiness check. The public playground now
persists expiries, sweeps after restart and in the background, validates inputs, and
uses a hashed dependency lock without the provider SDK.

These source changes are deliberately separate from the in-flight experiment. Merging
them into the experimental branch before dev100/test100 finish invalidates the frozen
source lineage; either finish the registered run first or explicitly abandon and
restart the frozen experiment.

The public showcase was rebuilt on `main` at `3026b78`: the browser-local guided tour
works before the backend wakes, the real engine is isolated as a second layer, mobile
uses a three-step pager, and published figures come from a release manifest. The planned
physical separation of product, research, negative results, and audit evidence is in
[`ARCHITECTURE_SEPARATION_PLAN.md`](ARCHITECTURE_SEPARATION_PLAN.md); it intentionally
does not move frozen source files yet.

A static-site-only truthfulness and accessibility hotfix was deployed on 2026-08-29 as
Cloudflare Pages deployment `18c9bea7`. The production page now describes the 60-minute
lifetime as access expiry rather than a restart-durable deletion guarantee, exposes the
release id/date/source commit/manifest, uses the step pager at tablet widths, creates a
fresh namespace for every preset run, and translates accessible names with the visible
copy. This dirty-worktree deployment deliberately changed no frozen experiment source;
its exact seven-file checksums are recorded under `public-demo/deployments/`.

## Still not a production service

The current API has no trusted tenant identity: `user_id` is supplied by the caller.
Main-service deletion is a provenance-preserving soft delete, not a data-subject hard
delete. PostgreSQL/pgvector, OIDC/JWT, export, backup/restore drills, quotas, alerts and
SLOs require deployment and policy choices and remain the P0–P5 productization work in
[`PRODUCTIZATION_V2_PLAN.md`](PRODUCTIZATION_V2_PLAN.md). They must not be represented
as complete merely because the local prototype passes tests.
