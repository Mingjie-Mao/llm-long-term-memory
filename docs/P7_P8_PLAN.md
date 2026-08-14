# P7–P8 Delivery Plan

This plan follows the current repository rather than the original phase checklist.
P1–P6 now have implementation and automated tests, but the new two-stage store has
not yet produced a replacement benchmark table. P7 must package a *pinned,
measured* configuration; it must not turn an unmeasured variant into a service
default.

## Preflight: finish the P1–P6 evidence loop

1. Run `chronomem ingest temporal-gate` with `configs/baselines.yaml`. Archive the
   JSON gate report and stop if it misses a threshold.
2. Ingest into the isolated `two-stage` store, then evaluate `two_stage` and
   `two_stage_no_temporal`. Keep the frozen v1 artifacts unchanged and compare the
   paired results before selecting a default.
3. Run `chronomem lifecycle run` and `chronomem lifecycle consolidate` only as
   separately recorded variants; lifecycle policies change retrieval and therefore
   must not be mixed into an earlier row.
4. Run `chronomem influence measure`, `chronomem influence fit`, then
   `chronomem eval budget-sweep`. Publish the resulting JSON and SVG with the
   selector (relevance or fitted utility), model path, dataset split, and commit
   SHA. Choose a Pareto point only after it completes.

The selected configuration, extraction prompt fingerprint, store name, model IDs,
and feature layout become the P7 service release manifest.

## P7 — Service, operations, and reproducibility

### 1. Service boundary and configuration

- Add an `api` package that constructs one store, index, encoder, retriever,
  lifecycle service, and ingestion service at application startup; HTTP handlers
  must not call Typer commands or duplicate domain logic.
- Version the API under `/v1`, validate all request and response models with
  Pydantic, and expose `GET /healthz` plus a read-only `GET /v1/config` manifest.
- Require a namespace/user ID on every stateful endpoint and reject cross-namespace
  reads. Keep SQLite as the single-process default; document that horizontal,
  multi-writer deployment requires the future Postgres adapter.

### 2. HTTP contract

- `POST /v1/messages`: validate and persist turns, run the configured extraction
  pipeline, return created/skipped memory IDs and extraction usage.
- `POST /v1/recall`: return the ranked memories, temporal status, strength, and all
  five normalized retrieval signals; include no generated answer.
- `POST /v1/pack`: take a query and token budget, return selected and rejected
  memories with utility/relevance, token accounting, type-floor decisions, and
  prompt-ready context.
- `GET /v1/memories`: paginate one namespace and support status/type filters;
  include evidence links and validity windows.
- `POST /v1/admin/consolidate`: make consolidation an explicit, authenticated
  background operation and return a job ID/status rather than holding an HTTP
  request open for LLM calls.

### 3. Observability and failure semantics

- Emit one structured JSON event per request with request ID, route, status,
  latency, LLM calls, input/output tokens, retrieved count, packed tokens, and
  selected configuration fingerprint.
- Record cost as `null` when no verified price schedule is configured; never invent
  a USD value from a free-tier run. Add an optional versioned pricing table later.
- Map validation errors, quota pauses, and provider failures to stable API errors;
  preserve resumable job checkpoints and never expose API keys or memory content in
  error logs.

### 4. Packaging and CI

- Add a minimal non-root Docker image with a mounted data/store/results volume,
  explicit environment variables, health check, and a `docker compose` development
  profile. Do not bake datasets, models, credentials, or SQLite state into the image.
- Extend the existing `.github/workflows/ci.yml` rather than replacing it: retain
  `ruff` and `pytest`, then add API-contract tests and a deterministic five-question
  mini-eval using fake client/encoder fixtures (no network or API key).
- Add a startup smoke test that creates a temporary store, calls health/recall/pack,
  and verifies structured logs contain token and latency fields.

### P7 acceptance gate

- `docker compose up` starts a healthy service with an empty mounted store.
- Contract tests cover success, namespace isolation, malformed input, quota pause,
  and lifecycle job initiation.
- CI remains offline, deterministic, and green; a release manifest reproduces the
  exact measured P6 configuration.

## P8 — MCP integration, demo, and project narrative

### 1. MCP server

- Add an `mcp` package that reuses P7 service/domain request models rather than
  implementing a second memory pipeline.
- Expose exactly three focused tools: `memory_write`, `memory_search`, and
  `memory_pack`. Their schemas must expose namespace and token budget explicitly.
- Return compact, model-usable results by default, with an opt-in inspector payload
  for scores, timeline status, evidence, and packing rejections. Test tool schemas,
  namespace isolation, and error mapping without a live MCP client.

### 2. Streamlit inspector

- Build the demo against the P7 API (or an injected local service client), never a
  second direct store implementation.
- Left panel: submit a conversation turn and ask a recall question. Right panel:
  show memories written this turn, hybrid signal breakdown, validity window,
  strength, evidence sources, packing outcome, and final prompt token use.
- Include a deterministic seeded walkthrough that demonstrates an update over time
  and a tight budget; live provider mode is optional and visibly labelled.

### 3. README and release evidence

- Update both English and Chinese READMEs only from generated artifacts: retain the
  frozen v1 loss, add the two-stage/P6 table when its runs complete, and link the
  sweep JSON/SVG instead of copying console numbers.
- Add an architecture diagram, local/Docker/MCP quickstarts, configuration and data
  persistence notes, API/MCP examples, privacy boundaries, and known limitations.
- Keep the project claim honest: it is an evaluated, explainable prototype unless a
  clean held-out result says otherwise; do not present dev-50 tuning as a benchmark
  win.

### P8 acceptance gate

- A fresh clone can start the API and demo with fixture data in one documented flow.
- The MCP tools complete the seeded write/search/pack walkthrough.
- In under two minutes, the demo explains why a memory was stored, retrieved,
  superseded, selected, or excluded under the token budget.

## Order and non-goals

Implement P7 in the order: service composition → read endpoints → write endpoint →
logging/failures → Docker/CI. Implement P8 only after those contracts are stable:
MCP → inspector → README/release artifacts. Postgres/pgvector, multi-worker SQLite
writes, authentication/tenant billing, and hosted deployment remain post-P8 work;
they must not be smuggled into the interview-demo milestone.
