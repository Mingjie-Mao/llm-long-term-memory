[English](README.md) · [中文](README.zh-CN.md)

# llm-long-term-memory

A persistent memory layer for LLM applications: extract structured facts from
conversations, track changes over time, and recover original turns when a question
needs a detail that extraction lost.

**Research prototype.** The frozen v2 system answered 72% of `test100` with 574 median
context tokens; full history reached 86% with 109,059. The tradeoff is smaller answer
context with lower accuracy than full history. [Results and counting conventions](#results).

[Interactive demo](https://lltm-memory.pages.dev) · [Architecture atlas](docs/ARCHITECTURE.md) ·
[Experiment history](docs/EXPERIMENT_HISTORY.md) · [Current research status](docs/CURRENT_STATUS.md)

The demo uses fictional scenarios. Its guided tour runs in the browser; its live
playground exercises local embeddings, retrieval and temporal updates on explicit
structured facts. It does not demonstrate LLM extraction or generated answers.

![System overview: conversations become structured memories while raw turns remain available for conditional source recovery](docs/figures/overview.svg)

[Editable SVG](docs/figures/overview.svg) · [中文架构图](docs/figures/overview.zh-CN.svg) · [Implementation and optional paths](docs/ARCHITECTURE.md)

## Quick start

Python 3.11+ and `uv` are required for the local route. Run from the repository root:

```bash
uv sync --group dev --extra api --extra llm --extra embed --extra mcp
uv run uvicorn llm_long_term_memory.api.app:app --host 127.0.0.1 --port 8000
```

Open the [inspector](http://localhost:8000) or [API reference](http://localhost:8000/docs).
The local embedder is downloaded on first use if it is not cached. In a second terminal:

```bash
curl http://localhost:8000/healthz
curl http://localhost:8000/v1/config
curl -X POST http://localhost:8000/v1/memories/search \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"alice","query":"where do I live?","explain":true}'
```

The default service opens `stores/two-stage-p10.db` using `configs/fallback.yaml`.
**A fresh checkout has no conversation data:** search returns an empty list until a
store is populated. Search needs the embedder, but no Gemini API key. Live answering
at `/v1/answer` additionally needs `GEMINI_API_KEY` in the environment or `.env`.

**Service writes are connected.** With the `llm` and `embed` extras and
`GEMINI_API_KEY`, `/v1/messages` and MCP `remember` lazily compose the real batch
extractor through `LiveTurnExtractor`, then deduplicate and resolve temporal updates.
Read-only deployments still work without a key. Batch ingestion remains available via
`uv run lltm ingest run --help`.
For a no-key write/search demonstration, use the [separate playground](public-demo/README.md#run-it-locally).

<details>
<summary>Docker</summary>

```bash
docker compose up --build
curl http://localhost:8000/healthz
```

The compose profile mounts `./stores` and `./configs`. The image contains code and
dependencies; datasets, databases, credentials and embedding model weights are not
baked in. Default configuration files are included; Compose can override them.

</details>

## How it works

- **Facts with source context.** Each memory has a subject, predicate, object, scope
  and speaker. `subject` identifies who the fact is about; `source_role` identifies
  who said it. Source session references and optional sentence spans link back to turns.
- **Updates with history.** Replacement intent closes earlier validity intervals
  without erasing the memory row. Resolution rebuilds the affected key in event order;
  coexisting facts can remain active together. Currently, date columns come from the
  source session date, while `ingested_at` records when the fact was learned.
- **Memory-first answering.** Retrieve and rank facts, assemble context, then ask for
  a structured verdict. Both `need_source` and `no_evidence` can trigger raw recovery;
  a second answerer call happens only when turns are recovered.

| Default path | Checked-in behavior |
|---|---|
| Extraction | Stage A facts → Stage B keys/update intent → rules/source spans; near-neighbour dedup may add LLM calls |
| Storage | SQLite + FTS5; `<store>-index.npy` and `<store>-index.ids.json` vector sidecars |
| Retrieval | Semantic and lexical candidate union; five signals computed, only semantic has nonzero default score weight |
| Candidate budget | Up to 50 per retrieval path, at most 100 after union; final top-k follows scoring |
| Context | Ranked facts; service top-k 10, v2 evaluation top-k 20 |
| Raw recovery | Source-local or archive-wide BM25; at most 3 turns and 2,400 source characters |
| Interfaces | REST and MCP share `MemoryService`; the batch pipeline and playground have separate write paths |

<details>
<summary>Write path: extraction, deduplication and temporal updates</summary>

![Batch write path with extraction, source registration, LLM deduplication, persistence, temporal resolution and checkpointing](docs/figures/write-path.svg)

Nonempty two-stage extraction usually takes two requests, plus any dedup adjudication.
The 0.92 similarity threshold identifies neighbours; it does not itself delete facts.
[Full caption and code mapping](docs/ARCHITECTURE.md#2-批量写入).

</details>

<details>
<summary>Storage: relational model, source links and vector files</summary>

![SQLite storage schema and separately persisted NumPy vector sidecars](docs/figures/storage.svg)

Source-session foreign keys and heuristic turn/span anchors provide different levels
of assurance. Vector sidecars are saved separately from SQLite.
[Full caption and code mapping](docs/ARCHITECTURE.md#3-存储模型).

</details>

<details>
<summary>Read path: candidate ranking and the three verdict branches</summary>

![Retrieval and conditional source recovery, including answer, need_source and no_evidence verdicts](docs/figures/read-path.svg)

The [complete atlas](docs/ARCHITECTURE.md) also explains out-of-order temporal updates
and the boundary between the runtime, playground and research evaluation.

</details>

Reranking, session-coherent context, unconditional hydration, decay, consolidation and
utility-based packing are available but absent from the default answer route. v3
reasoning/hydration and v4 synthesis/scanning are explicit research variants.
See the [branch map](docs/ARCHITECTURE.md#可选分支的真实状态) and
[disabled-feature evidence](results/shipped-but-disabled.md).

## Results

**Frozen v2 final test: 100 LongMemEval-S questions, one run per arm.** These are the
archived system's results, not a measurement of every current research variant.

| Arm | Accuracy | Median context tokens | Answer + judge tokens |
|---|---:|---:|---:|
| Full history | **86.0%** | 109,059 | 10,932,294 |
| **v2 memory + conditional fallback** | 72.0% | **574** | **159,458** |
| Naive RAG over raw sessions | 65.0% | 12,763 | 1,352,847 |

The paired v2 gain over naive RAG was +7 points (`p=0.3368`), which is not statistically
conclusive. Full history was 14 points more accurate than v2 (`p=0.0043`).
[Final report](results/final/test100-aggregate.md).

Context counts follow the original runners: memory and RAG use character-based
estimates; full history reports provider input tokens. The last column is recorded
API usage including grading, and excludes the shared extraction stage's **13,757,713
tokens**. Context ratios are approximate, not a billing estimate. The final test has
no repeat-variance measurement.

**Separate research validation: v3.3 on dev60.** Majority accuracy over three runs was
55.0% versus 46.7% for its v2 control, with median context 1,115 versus 573 tokens
(`p=0.1797`). This is a different, deliberately harder question set; do not compare its
55.0% directly with v2's 72.0% above. A later schema audit found the confidence gate
was vacuous because confidence stayed at a default value. Accuracy is unaffected;
eight other gates remain interpretable. [Aggregate](results/validation/v3-dev60.md),
[schema audit](results/audit/v3-verdict-schema-never-sent-20260906.json).

Source-session recall is an **any-gold-session hit** metric. It does not prove that
all answer-bearing facts survived extraction. Detailed development results, negative
results and the already-measured v4 probes are in the [experiment history](docs/EXPERIMENT_HISTORY.md).

## Interfaces

With `LLTM_API_TOKENS` configured, REST derives the tenant from a bearer token and
rejects a conflicting request `user_id` with 403. Without configured tokens, open
mode lets the caller choose a namespace. Nonempty malformed authentication settings
fail closed. Tokens are environment-managed and do not yet have automatic expiry or
a complete rotation lifecycle.

| REST endpoint | Purpose / availability |
|---|---|
| `GET /healthz` · `GET /v1/config` | Readiness and resolved service configuration |
| `POST /v1/memories/search` | Ranked memories, signals, provenance and optional rejection trace |
| `GET /v1/memories` · `GET /v1/memories/{id}` | Browse or inspect memory and available source evidence |
| `GET /v1/timeline` | History for a subject/predicate |
| `POST /v1/raw/search` | Search the raw conversation archive |
| `POST /v1/answer` | Live memory-first answer; requires an API key and LLM dependency |
| `POST /v1/messages` | Extract and persist a turn; requires the LLM/embed extras and an API key |
| `DELETE /v1/memories/{id}` | Mark an active memory evicted |
| `GET /v1/export` | Export this namespace's data and provenance |
| `DELETE /v1/data` | Hard-delete this namespace's online database and vector data; excludes backups |

<details>
<summary>MCP setup</summary>

```bash
uv run lltm mcp                    # stdio
uv run lltm mcp --transport http   # streamable HTTP
```

```json
{
  "mcpServers": {
    "long-term-memory": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/llm-long-term-memory", "lltm", "mcp"]
    }
  }
}
```

Tools: `search_memory`, `search_conversations`, `get_timeline`, `forget`, and `remember`.
All take `user_id`; `remember` uses the same configured write path as REST.
MCP exposes memory tools rather than a separate generated-answer tool.

MCP stdio is for trusted local clients. Its HTTP transport has no equivalent to
REST's credential-derived identity boundary and should remain local-only.

</details>

## Limitations

- Extraction is lossy. Date columns inherit session dates, and source spans are
  heuristic. A source-session hit is weaker than exact evidence coverage.
- Full history is more accurate on the frozen final test. The reported token savings
  concern answer context and do not remove ingestion cost.
- The service is a prototype: empty token settings allow open access, writes are
  serialized within one process, and SQLite and vector files persist separately.
  Cross-resource recovery and multi-process writes remain unfinished.
- Ordinary `forget` preserves stored rows; REST namespace erasure removes online
  data. Backup retention, replaying erasures after restore, and production recovery
  drills remain unfinished.

## Documentation

| Document | Contents |
|---|---|
| [Architecture atlas](docs/ARCHITECTURE.md) | Six vector figures, exact code mappings, default and optional branches |
| [Experiment history](docs/EXPERIMENT_HISTORY.md) | Consolidated phase results and qualifications |
| [System report](docs/REPORT.md) · [中文](docs/REPORT.zh-CN.md) | Detailed design rationale and earlier measurements |
| [Decisions](docs/DECISIONS.md) | Measured design decisions |
| [Current status](docs/CURRENT_STATUS.md) · [Roadmap](docs/ROADMAP.md) | Evolving research log and planned work |
| [Separation plan](docs/ARCHITECTURE_SEPARATION_PLAN.md) | Proposed core/research split; not the current module layout |
| [Data protocol](results/data-protocol.md) | Permitted uses of question sets |
| [README review](docs/README_REVIEW.zh-CN.md) | Specific documentation corrections and remaining suggestions |

[MIT License](LICENSE)
