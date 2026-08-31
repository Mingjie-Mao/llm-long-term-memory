# Separation readiness package

**Status:** implementation-ready design; source migration is blocked until the registered v2
dev/test sequence is complete.  
**Purpose:** turn the architecture proposal into a reviewable migration checklist without changing
the active experiment's source or artifact lineage.

## 1. Non-negotiable invariants

1. The active `v2-candidate` continues to verify against the same source, configuration, inputs,
   protocol records and store.
2. Product core contains no LongMemEval types, evaluation prompts, judge code, benchmark runners,
   pre-registration logic or result rendering.
3. Research may depend on public core contracts; core may never import research.
4. Every experiment remains immutable and receives one explicit outcome: `positive`, `negative`,
   `inconclusive` or `invalid`.
5. Audit evidence is checksummed proof, not a second copy of a conclusion.
6. Public release manifests contain aggregate publishable facts only; raw rows remain sealed.
7. Moves preserve bytes with `git mv` and a generated legacy-path map. Behaviour changes and moves
   never share a review unit.

## 2. Verified dependency violations to remove

| Current consumer | Current dependency | Required replacement |
|---|---|---|
| `api/service.py` | evaluation prompt versions and `MemoryRunner` | core-owned `AnswerPipeline` and prompt versions |
| `api/service.py` | LongMemEval `Instance` | neutral `MemoryQuery` |
| product ingest composition | dataset-shaped sessions | neutral `ConversationSession` and `ConversationTurn` |
| monolithic `cli.py` | evaluation, judge, manifests, fidelity and diagnostics | `lltm` product CLI plus `lltm-exp` research CLI |
| product wheel | `evaluation/`, `influence/`, research-only ingest diagnostics | exclude from core wheel |
| showcase | copied presentation values | release-manifest-only rendering |

The current CLI is approximately 1,790 lines and is the largest coupling point. It is split only
after the runtime contracts are extracted; splitting command registration first would hide the
dependency rather than remove it.

## 3. Ownership inventory

| Current area | Owner after split | Migration rule |
|---|---|---|
| `store/`, `retrieve/`, `temporal/`, `pack/`, `embed/`, `locking.py`, `lifecycle.py` | core | move unchanged after Phase 1 contracts pass |
| runtime extraction, deduplication, provenance and temporal keying in `ingest/` | core | first replace dataset-shaped inputs with neutral types |
| `ingest/coverage.py`, `fidelity.py`, `stage_b_quality.py`, `temporal_gate.py`, `temporal_pairs.py`, `zero_yield*` | research diagnostics | no import from product API/MCP/CLI |
| `evaluation/` and dataset adapters | research | product runner becomes a thin adapter over core public APIs |
| `influence/` | research | optional research dependency group only |
| `api/`, `mcp_server.py` | core | depend on `MemoryWriter` and `AnswerPipeline` |
| product subcommands in `cli.py` | core CLI, `lltm` | serve, inspect, remember, import, search, forget |
| benchmark/freeze/judge/report subcommands and `scripts/` | research CLI, `lltm-exp` | importable commands with isolated tests |
| inspector application | `apps/inspector` | core-only consumer |
| public demonstration | `apps/showcase` | consumes release manifest, never research rows/database |
| pre-registrations and data protocol | `artifacts/registry` | immutable protocol inputs |
| run manifests, config, raw outputs, usage and summary | `artifacts/experiments/<run-id>` | one immutable directory per run |
| failed, disabled and no-gain decisions | `artifacts/negative-results/index.json` | navigation index pointing to experiment ids |
| freezes, hashes, coverage and integrity reports | `artifacts/audit` | immutable verification evidence |
| publishable aggregate metrics | `artifacts/releases/<version>` | generated, aggregate-only public contract |

## 4. Execution slices and rollback points

| Phase | Entry gate | Change set | Acceptance gate | Rollback boundary |
|---:|---|---|---|---|
| 0 | v2 final ledger and report complete | record final experiment state | freeze verifies and final report is immutable | abandon separation branch; never rewrite experiment |
| 1 | Phase 0 | neutral domain types, LongMemEval adapters, forbidden-import checker | old/new fixtures are byte-equivalent; all existing tests pass | remove only new contracts/adapters |
| 2 | Phase 1 | extract `MemoryWriter` and `AnswerPipeline` | fixed golden outputs unchanged; REST/MCP use core interfaces | revert runtime extraction without moving files |
| 3 | Phase 2 | create core/research packages; mechanical `git mv` | core wheel imports with research absent | reverse path map; byte hashes must still match |
| 4 | Phase 3 | split `lltm` and `lltm-exp` | product help contains no research commands | restore command registration only |
| 5 | Phase 4 | split tests and CI jobs | core, research, contracts and showcase pass independently | restore CI matrix; packages remain intact |
| 6 | Phase 5 | migrate artifacts and generate legacy map | every tracked old path maps once with identical SHA-256 | reverse only verified map entries |
| 7 | Phase 6 | generate release manifest and bind showcase to it | all displayed values equal validated manifest | serve previous release manifest |
| 8 | Phase 7 | clean-install, container, REST, MCP and Demo release gates | no forbidden wheel contents; all smoke tests pass | do not tag or deploy |

Each phase is one review unit when commits are permitted. Until then, do not stack phases in one
uncommitted worktree because a failure could not be isolated reliably.

## 5. Planned enforcement commands

The implementation phase must provide commands equivalent to these gates; names may be finalized
when the packages exist.

```text
check-import-boundaries     core must not import research/evaluation/datasets
check-core-wheel            wheel contains no evaluation, influence, prereg or LongMemEval path
test-core-clean-install     install and import core without research dependencies
test-research-install       install research with core and run adapters/evaluation fixtures
verify-artifacts            validate schemas, checksums, outcomes and evidence links
verify-legacy-map           require one verified mapping for every moved artifact
check-showcase-release      require every public number to come from one release manifest
```

## 6. Artifact decision flow

1. Create the experiment directory before the first provider call.
2. Record producer, source/config/input hashes and expected arms.
3. Append raw output and usage without rewriting completed rows.
4. Write an aggregate summary only after the registered completion gate.
5. Classify the run as `positive`, `negative`, `inconclusive` or `invalid`; never infer the class
   from its directory name.
6. Add negative/inconclusive/invalid decisions to the negative-results index when they affect a
   product or research choice.
7. Put integrity proofs in audit and aggregate public claims in releases. Both point back to the
   immutable experiment id.

Schemas prepared before migration:

- [`schemas/experiment-artifact.schema.json`](schemas/experiment-artifact.schema.json)
- [`schemas/negative-results-index.schema.json`](schemas/negative-results-index.schema.json)
- [`schemas/legacy-path-map.schema.json`](schemas/legacy-path-map.schema.json)

The first schema-valid historical classification is
[`results/negative-results-index.draft.json`](../results/negative-results-index.draft.json).
It remains a draft navigation layer: the evidence it cites stays at its frozen legacy path until
Phase 6 performs the checksummed migration.

## 7. Definition of ready to implement

Implementation may start only when all of the following are true:

- dev100 aggregate and automatic decision exist;
- the one-shot test100 ledger is complete;
- the final v2 report has been rendered without changing the registered arms;
- the final experiment freeze verifies;
- the product/Demo repair change has a conflict-resolution plan against the final experiment tree;
- a clean baseline test run is recorded;
- the user permits the review units to be committed, or explicitly accepts an uncommitted staged
  migration with reduced rollback quality.
