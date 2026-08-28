# Architecture separation plan

**Status:** planned; do not execute against the frozen v2 lineage yet  
**Scope:** product core, research code, experiment outcomes, negative results, audit evidence,
and showcase publishing  
**Repository strategy:** one Git repository, two installable Python packages, explicit artifact
classes; no multi-repository split

## 1. Why this is still open

The repository documents the difference between product and research, but the dependency graph
does not enforce it:

- `api/service.py` imports evaluation prompts, the LongMemEval `Instance`, and `MemoryRunner`;
- the production ingest path accepts LongMemEval `HaystackSession` values;
- `evaluation/`, `influence/`, and experiment-only ingest diagnostics ship inside the product
  wheel;
- one 1,790-line CLI exposes product operations, benchmark execution, adjudication, diagnostics,
  migrations, and final-test commands;
- `results/` mixes registered inputs, raw outputs, negative results, audit snapshots, and public
  conclusions;
- the showcase copied release numbers into HTML, allowing it to drift from verified artifacts.

Documentation is not an architecture boundary. This plan makes the desired dependency direction
testable.

## 2. Freeze constraint

The active v2 experiment hashes `src/**/*.py`, `src/**/*.sql`, `scripts/*.py`, `pyproject.toml`,
and `uv.lock`. Moving or refactoring those files before the registered dev/test sequence finishes
changes the source lineage and invalidates the comparison.

Therefore:

1. The current frozen branch continues unchanged until v2 finishes or is explicitly abandoned.
2. Planning, `public-demo/`, and other non-frozen presentation work may proceed independently.
3. The separation starts from the final registered experiment commit, on a new branch.
4. Historical artifacts are moved with `git mv`; a generated legacy-path map preserves links.
5. No separated product release is tagged until product-only and research-only installations both
   pass their independent test gates.

## 3. Target layout

```text
packages/
  core/
    pyproject.toml
    src/llm_long_term_memory/
      domain/              neutral ConversationSession, Turn, Query, Memory
      ingest/              extraction, deduplication, provenance, temporal keying
      retrieve/            hybrid/coherent retrieval, hydration, fallback
      temporal/            validity and supersession resolution
      store/               protocols, SQLite, vector index, session keys
      runtime/             MemoryWriter and AnswerPipeline composition
      llm/                 optional provider client, quotas, usage
      api/                 REST application
      mcp/                 MCP application
      cli.py               serve, inspect, import, remember, search, forget

  research/
    pyproject.toml
    src/lltm_research/
      datasets/            LongMemEval adapters and manifests
      evaluation/          harness, judge, comparisons, reports
      runners/             full-context, naive-RAG, product adapter
      diagnostics/         fidelity, coverage, zero-yield, failure attribution
      influence/           utility measurement and predictor experiments
      reproducibility/     freeze, validation, state checks
      cli.py               ingest experiments, eval, judge, aggregate, freeze

apps/
  inspector/               product operator UI
  showcase/                public fictional demonstration

artifacts/
  registry/                pre-registrations, data protocol, selection records
  experiments/<run-id>/    immutable manifest, config, raw output, usage, summary
  negative-results/        indexed lessons pointing to immutable experiment runs
  audit/                    hashes, coverage snapshots, integrity/state reports
  releases/<version>/      public metrics manifest and release notes

tests/
  core/
  research/
  contracts/               adapters and artifact-schema compatibility
  showcase/
```

`packages/research` may depend on `packages/core`. The reverse dependency is forbidden.
`apps/showcase` may consume only `artifacts/releases/<version>/manifest.json`, never raw benchmark
rows or a research database.

## 4. Stable contracts to extract first

### 4.1 Neutral conversation types

Create product-owned types:

```python
ConversationSession(id, user_id, occurred_at, turns)
ConversationTurn(index, role, content, occurred_at)
MemoryQuery(namespace, text, occurred_at=None)
```

The LongMemEval adapter converts `HaystackSession` and `Instance` into these values in research.
The core package must not know the dataset's class names or question ids.

### 4.2 MemoryWriter

Move single-turn and batch write composition behind one product interface:

```python
MemoryWriter.write_session(session) -> WriteOutcome
MemoryWriter.write_turn(namespace, session_id, turn) -> WriteOutcome
```

`WriteOutcome` owns extracted memories, deduplication decisions, temporal changes, and provider
usage. Research ingestion checkpoints wrap this outcome instead of living inside it.

### 4.3 AnswerPipeline

Extract retrieval, packing, structured insufficiency judgement, raw fallback, and answering from
`evaluation.runners.memory.MemoryRunner` into a product `AnswerPipeline`.

- REST and MCP call `AnswerPipeline` directly.
- `ProductRunner` in research converts an evaluation `Instance` to `MemoryQuery`, calls the
  pipeline, and converts the outcome back to the harness `Answer`.
- evaluation prompts and runner bookkeeping no longer appear in the product service.

### 4.4 Release manifest

Define one versioned JSON schema containing only publishable aggregate results:

```json
{
  "schema_version": 1,
  "release": "v1",
  "source_commit": "...",
  "tests": 565,
  "coverage": 83,
  "metrics": {},
  "limitations": [],
  "evidence": []
}
```

The showcase renders this file. A release command creates it from verified aggregate artifacts;
hand editing generated figures is prohibited.

## 5. File migration map

| Current area | Target | Rule |
|---|---|---|
| `store/`, `retrieve/`, `temporal/`, `pack/`, `lifecycle.py` | core | move without behaviour change |
| runtime parts of `ingest/` | core | replace dataset types before move |
| `ingest/coverage.py`, `fidelity.py`, `stage_b_quality.py`, `temporal_gate.py`, `zero_yield*` | research diagnostics | must not ship in product wheel |
| `evaluation/` | research | `MemoryRunner` becomes a thin adapter over core `AnswerPipeline` |
| `influence/` | research | optional research dependency only |
| product commands in `cli.py` | core CLI (`lltm`) | no benchmark flags or dataset concepts |
| eval/freeze/judge commands in `cli.py` and `scripts/` | research CLI (`lltm-exp`) | scripts become importable commands with tests |
| `results/prereg-*`, `data-protocol.md` | artifact registry | immutable protocol records |
| `results/raw/*`, usage files | experiment run directories | manifest names the producer and checksum |
| failure analyses and shipped-but-disabled | negative-results index | retain evidence links and decision outcome |
| frozen snapshots, coverage and state reports | audit | immutable, checksummed |
| final tables and public numbers | releases | aggregate-only publishable data |

## 6. Execution phases and gates

| Phase | Change | Required gate |
|---|---|---|
| 0 | Finish or abandon frozen v2 explicitly | final experiment commit and state report recorded |
| 1 | Add neutral domain contracts and adapters without moving files | old and new adapters produce byte-equivalent sessions/queries |
| 2 | Extract `MemoryWriter` and `AnswerPipeline` | existing golden/evaluation results unchanged on fixed fixtures |
| 3 | Create core/research packages and move modules with `git mv` | core wheel imports with no research package installed |
| 4 | Split `lltm` and `lltm-exp` | product CLI help contains no eval/freeze commands; research CLI retains them |
| 5 | Re-home tests and CI | core, research, contract, and showcase jobs pass independently |
| 6 | Re-home artifacts and generate legacy map | every old tracked artifact has one new path and identical SHA256 |
| 7 | Add release-manifest generator and point showcase at it | page values equal manifest; manifest links to immutable evidence |
| 8 | Tag first separated product release | clean installs and Docker/API/MCP smoke all pass |

Each phase is a separately reviewable commit. Do not combine code moves with behaviour changes;
otherwise review cannot distinguish a path error from a semantic change.

## 7. Enforcement

CI must make regression difficult:

1. **Import boundary:** an AST/import-linter rule fails on any `llm_long_term_memory` import of
   `lltm_research`.
2. **Wheel contents:** build the core wheel and assert no path contains `evaluation`, `dataset`,
   `influence`, `prereg`, or `LongMemEval`.
3. **Clean core install:** install only core base dependencies and import every public core module.
4. **Research install:** install the research package with core and run dataset/evaluation tests.
5. **Artifact schema:** validate manifests, checksums, outcome classification, and evidence links.
6. **Negative-result discoverability:** every disabled/failed arm appears in
   `artifacts/negative-results/index.json` with `decision`, `reason`, and `experiment_ids`.
7. **Showcase contract:** the public site consumes only a release manifest and has no hard-coded
   metric cards.

## 8. Artifact policy

An experiment is never classified by moving only its favourable rows. Every run directory contains:

```text
manifest.json
config.yaml
inputs.sha256
source.sha256
raw/<outputs>
usage.json
summary.json
outcome.json       # positive | negative | inconclusive | invalid
```

Negative results are first-class outcomes, not discarded branches. The negative-results index is a
curated navigation layer; the underlying immutable run remains beside positive and inconclusive
runs. Audit evidence is separated because it answers a different question: not "what happened?",
but "can this claim be traced and reproduced?"

## 9. Definition of done

The separation is complete only when all statements below are true:

- installing the core wheel does not install or expose research modules;
- product API, MCP, and CLI import no evaluation or dataset code;
- research reproduces the registered product result through public core contracts;
- one command can enumerate every experiment, including negative and invalid outcomes;
- one command verifies every artifact checksum and legacy-path mapping;
- the showcase has one versioned source for every displayed metric;
- `README` presents product usage first and links research as evidence, not as the product surface;
- the frozen v2 result retains an unbroken source/config/input hash chain through the migration.

## 10. First executable slice after v2

The first implementation PR should do only this:

1. add neutral conversation/query types;
2. add LongMemEval-to-core adapters under research;
3. make extractors and the answer path accept the neutral types;
4. add the forbidden-import CI rule;
5. prove all current tests and registered fixture outputs are unchanged.

No directory move belongs in that first slice. Once the dependency direction is clean, the physical
move becomes mechanical and reviewable.
