# Current status — 2026-08-28

This is the one current-state page. The long reports and execution log are evidence
and history; older entries in them are intentionally not rewritten.

## Experiment

| item | state |
|---|---|
| v1 heldout100 | complete: registered single shot 70/100; repeats 71 and 73 |
| v2 train150 | complete: 7,180/7,180 sessions; 18,519 memories |
| v2 selected context | `mean`, radius 1, cap 30; 95.3% Top-3 and assembled source recall |
| v2 dev100 ingest | 3,660/4,791 terminal (76.4%); 1,131 remain; state checker reports no issues |
| v2 dev validation | not started: five arms × three repeats after post-ingest freeze |
| v2 test100 | sealed; no v2 result exists yet |

The exact resumable command and one-shot order remain in
[`results/v2-runbook.md`](../results/v2-runbook.md). Do not use `--fresh`, delete a
ledger, or treat 76.4% as an accuracy score.

## Product code

The post-freeze repair branch fixes the default REST/MCP write composition, uses the
real batch extractor through a single-turn adapter, serializes access to the shared
SQLite/index resources, avoids mutating a shared answer runner, externalizes session
ids consistently, and makes `/healthz` a readiness check. The public playground now
persists expiries, sweeps after restart and in the background, validates inputs, and
uses a hashed dependency lock without the provider SDK.

These source changes are deliberately separate from the in-flight experiment. Merging
them into the experimental branch before dev100/test100 finish invalidates the frozen
source lineage; either finish the registered run first or explicitly abandon and
restart the frozen experiment.

## Still not a production service

The current API has no trusted tenant identity: `user_id` is supplied by the caller.
Main-service deletion is a provenance-preserving soft delete, not a data-subject hard
delete. PostgreSQL/pgvector, OIDC/JWT, export, backup/restore drills, quotas, alerts and
SLOs require deployment and policy choices and remain the P0–P5 productization work in
[`PRODUCTIZATION_V2_PLAN.md`](PRODUCTIZATION_V2_PLAN.md). They must not be represented
as complete merely because the local prototype passes tests.
