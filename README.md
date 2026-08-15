[![English](https://img.shields.io/badge/English-2962FF?style=for-the-badge)](README.md)
[![中文](https://img.shields.io/badge/%E4%B8%AD%E6%96%87-555555?style=for-the-badge)](README.zh-CN.md)

# llm-long-term-memory

**A persistent long-term memory layer for LLM applications.**

It remembers compactly, updates facts as they change, and falls back to the original
conversation when compression turns out to have lost the exact detail a question
needs.

---

## The problem, in one exchange

Three months ago the assistant answered a question. Today the user asks about it:

> **Q:** What was the Mayo Clinic YouTube video you recommended?

The memory store holds nothing about Mayo Clinic — extraction kept the gist and
dropped the link. A vector store would fail here and stay failed. This does not:

```
Structured memory          insufficient
Raw conversation fallback  triggered — archive-wide
Evidence                   session answer_sharegpt_81riySf_0 · turn 1 · assistant

Answer
  "How to Sit Properly at a Desk to Avoid Back Pain"
  https://www.youtube.com/watch?v=UfOvNlX9Hh0
```

Compression is lossy. The design accepts that and requires only that the loss be
**recoverable**.

### See it yourself

Start the service (below) and open:

| Demo | What it shows |
|---|---|
| [`/?demo=mayo`](http://localhost:8000/?demo=mayo) | Compression dropped a URL; the archive got it back |
| [`/?demo=battery`](http://localhost:8000/?demo=battery) | Memory sufficed — the reply *applies* a remembered preference, no fallback |
| [`/?demo=timeline`](http://localhost:8000/?demo=timeline) | One fact, five values over time, with the supersession chain |
| [`/?demo=collectibles`](http://localhost:8000/?demo=collectibles) | A multi-valued key whose facts coexist rather than replace |

Each opens a **recorded run** — real output from a real model, labelled as recorded,
so the page costs nothing to open. A **Run live answer** button re-executes it on
demand. Recordings carry a fingerprint of the store and prompt versions that
produced them; if either changes, the page says **stale** instead of pretending to be
current.

---

## What it does

| | |
|---|---|
| **Remember** | Turns conversations into typed, time-bounded facts — who said it, who it is about, and why it is worth keeping |
| **Update** | Tracks facts that change. "I moved to Sydney" supersedes "I live in Canberra" without deleting it |
| **Retrieve** | Five weighted signals — semantic, BM25, recency, importance, entity overlap |
| **Recover** | When structured memory is insufficient, searches the original conversation instead of failing |
| **Explain** | Every memory traces to the turn it came from; every *omission* has a stated reason |

The last one is the differentiator. Search returns not just what matched but what was
**rejected and why** — `superseded` (the fact is no longer true) or `below_rank`
(still true, lost on score). An absence without an explanation is indistinguishable
from a bug.

---

## Architecture in 30 seconds

```
                     Conversation
                          │
              ┌───────────┴───────────┐
              ↓                       ↓
      Raw conversation            Extractor
      archive (turns)        (2 LLM calls / batch)
              │                       ↓
              │              Structured memory
              │             typed · bi-temporal
              │              provenance-anchored
              │                       │
  Query ──────┼───────────────────────┘
              │        hybrid retrieval (5 signals)
              │                       ↓
              │               enough to answer?
              │              ┌────────┴────────┐
              │             yes                no
              │              ↓                  ↓
              └──────────> answer      raw-source recovery
                                                ↓
                                       source-cited answer
```

**The raw archive is a storage decision, not a retrieval algorithm.** Keeping the
original turns is what makes lossy extraction recoverable; *how* they are found —
source-local lookup, BM25, dense, hybrid — is an independent choice.

| Layer | Choice | Why |
|---|---|---|
| Store | SQLite (WAL) + FTS5 | One file, no service dependency; BM25 for free |
| Vectors | Exact flat inner-product (numpy) | Thousands of memories; an ANN index would add a dependency to save microseconds |
| Embeddings | `all-MiniLM-L6-v2`, local | Corpus-wide embedding is ~53M tokens; API quota is the binding constraint |
| LLM | Gemini, three independent roles | Extractor, answerer and judge have different requirements |
| Service | FastAPI, one composition root | The API and inspector are clients of the same service object |

---

## Run it in 2 minutes

```bash
docker compose up --build
```

```bash
curl localhost:8000/healthz
```

Then open <http://localhost:8000> for the inspector.

The image bakes in no datasets, models, credentials or database — the store arrives
on a mounted volume. A `GEMINI_API_KEY` is optional: without one the service runs
read-only and says so.

The first build downloads the embedding model's dependencies, so budget a few
minutes for it; subsequent starts are seconds. For a smaller read-only image without
semantic search, build with `--build-arg EXTRAS="api"` — `/healthz` will then report
`degraded` and search will return 503 rather than failing obscurely.

**Image size: 2.95GB.** Most of that is PyTorch, which the embedding model needs.
The default build pulls the CUDA wheel — 24.4GB of GPU runtime for a container that
will never see a GPU — so the Dockerfile installs the CPU wheel and deletes the CUDA
packages orphaned by the swap. It is still not small; a deployment that wants small
should run the encoder out of process, which is [on the roadmap](docs/ROADMAP.md) and
is not something to change while an evaluation is mid-flight.

<details>
<summary>Without Docker</summary>

```bash
uv sync --group dev --extra api --extra embed
uv run uvicorn llm_long_term_memory.api.app:app --port 8000
```
</details>

---

## Integrate in 5 minutes

```bash
curl -X POST localhost:8000/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"user_id": "alice", "role": "user", "content": "I stopped drinking coffee last month."}'
```

```bash
curl -X POST localhost:8000/v1/memories/search \
  -H 'Content-Type: application/json' \
  -d '{"user_id": "alice", "query": "what does she drink?", "explain": true}'
```

```json
{
  "memories": [
    {
      "content": "The user stopped drinking coffee.",
      "scope": "profile",
      "source_role": "user",
      "score": 0.81,
      "signals": {"semantic": 0.79, "bm25": 0.92, "recency": 0.4, ...},
      "source": {"session_id": "s_4f2a", "turn_index": 0, "char_start": 0, "char_end": 41}
    }
  ],
  "rejected": [
    {"memory_id": "m_9c1", "reason": "superseded", "superseded_by": "m_a20",
     "content": "The user drinks two coffees a day."}
  ],
  "candidates_considered": 37
}
```

| Endpoint | |
|---|---|
| `POST /v1/messages` | Ingest a turn; returns the memories it created |
| `POST /v1/memories/search` | Retrieve with signals, provenance and rejections |
| `POST /v1/answer` | Full answer path, including fallback when needed |
| `POST /v1/raw/search` | The fallback layer, queryable directly |
| `GET /v1/memories` | Browse a namespace; filter by status / type / scope / speaker |
| `GET /v1/memories/{id}` | One memory with its source turn |
| `GET /v1/timeline` | The supersession chain for a `(subject, predicate)` |
| `DELETE /v1/memories/{id}` | Forget — marks evicted, never a hard delete |
| `GET /healthz` · `GET /v1/config` | Health and the running manifest |

Every read requires a `user_id` and is namespace-isolated. Reading another
namespace's memory returns **404, not 403** — 403 would confirm the id exists.

---

## Connect from an MCP-compatible client

[Model Context Protocol](https://modelcontextprotocol.io) is how an agent connects
to this, where REST is how a program does. Both are thin wrappers over the same
service object, so the tools cannot drift from the behaviour that was measured.

```bash
uv sync --extra mcp --extra embed
uv run lltm mcp                    # stdio
uv run lltm mcp --transport http   # streamable HTTP
```

For Claude Desktop or Cursor, add to the MCP config:

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

| Tool | |
|---|---|
| `search_memory` | Recall what is known, with the reason each memory was selected — and with `explain`, why others were not |
| `remember` | Store a turn; returns the memories it produced, so the agent can confirm what was understood |
| `search_conversations` | Recover the original wording when a memory is on topic but lacks the exact detail |
| `get_timeline` | How one fact changed over time |
| `forget` | Mark a memory evicted; never a hard delete |

Every tool takes an explicit `user_id`. There is no ambient session identity: an
agent serving several people must say which one it is acting for, and the store
enforces the boundary rather than trusting the caller.

---

## Evidence

The design above is not a preference. Each choice below was measured, and two
plausible additions were measured and **not shipped**.

### Structured memory as compression

On 31 fully-ingested questions from [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval),
same answerer and judge throughout ([full write-up](results/a2-pilot.md)):

| Variant | Accuracy | Median context tokens |
|---|---:|---:|
| `full_context` | 64.5% | 109,605 |
| `naive_rag` | 51.6% | 13,057 |
| **`two_stage` (k=10)** | **51.6%** | **242** |
| v1 structured memory | 19.4% | 465 |

**~54x less context than naive RAG, with no detectable accuracy difference**
(6W-6L, paired exact McNemar, p = 1.000).

> **Read this as a pilot.** 31 questions, unstratified, one run each. `p = 1.000`
> means *no difference was detected*, not that the systems are equivalent. A
> stratified 50-question run is pending, and no held-out result exists yet.

### Where the gain came from

An early draft credited evidence hydration. A causal ablation reversed that:

| Step | Δ | Paired |
|---|---:|---|
| extraction rewrite | **+29.0pp** | 11W-2L, **p = 0.022** |
| evidence hydration | +3.2pp | 2W-1L, p = 1.000 |

Source-session recall is 93.5% for *both*, which confirms it — hydration runs after
retrieval and cannot change what is recalled. So hydration was demoted from an
always-on stage to the conditional fallback described above.

### What was measured and not shipped

**Cross-encoder reranking.** At k=10 the reranked and plain arms answered all 31
questions **identically** — zero disagreements — while reranking *lowered* source
recall at every k (93.5% → 90.3% at k=20). It stays in the repo behind an optional
`rerank` extra so the result is reproducible, and off by default.
[Details](results/rerank-pareto.md).

**Dense retrieval over the raw archive.** A constructed query set suggested BM25
collapses on paraphrases (6.9% R@1). Manual inspection showed the construction had
deleted the *questions*, not just their vocabulary — *"How long have I been
collecting vintage cameras?"* became `"long"`. That evidence was discarded.
Hand-written paraphrases retrieve the gold turn at **rank 1**, including phrasings
containing no distinctive noun from the source. Deferred, not rejected, with the
conditions that would reopen it. [Details](results/raw-recall-diagnostic.md).

**A learned utility predictor for budget packing.** Held-out RMSE 0.310 against a
mean-baseline 0.263 — it lost to predicting the mean. [Details](results/p6-pilot.md).

### Live regression

Seven questions that every variant previously failed now score **6/7, identical
across three runs** — including the Mayo case end to end.
[Details](results/live-regression-v2.md).

---

## Engineering

- **352 tests**, CI across ubuntu / windows / macos
- **Result versioning** — every evaluation row records its answerer prompt, judge
  prompt and extractor version; the extractor version comes from the *store*, not
  the checkout, because it describes the data being evaluated
- **Frozen manifests** — reported runs name an explicit question set, because a
  sample size is not an experiment identity
- **Structured logging** — one JSON event per request with request id, latency and a
  config fingerprint; no memory content, no credentials
- **Encoder warm-up** at startup: first request 12,486 ms → 181 ms

Full write-up: [Engineering Report](docs/ENGINEERING_REPORT.md) ·
[Design decisions](docs/DECISIONS.md) · [Roadmap](docs/ROADMAP.md)

---

## Development

```bash
uv sync --group dev --extra api --extra llm --extra embed
uv run pytest
uv run lltm --help
```

<details>
<summary>Benchmark workflow</summary>

```bash
uv run lltm data download --variant s
uv run lltm doctor
uv run lltm ingest run --store-name two-stage-hydrated
uv run lltm eval freeze dev50 --n 50
uv run lltm eval run two_stage --questions results/manifests/dev50.json
uv run lltm eval compare naive_rag two_stage
```

`--questions` takes a frozen manifest. `--limit` re-samples and is for exploration
only — a stratified subset is scattered across the split, not a prefix of it.
</details>

```
src/llm_long_term_memory/
  api/         FastAPI service, inspector, recorded demo runs
  store/       schema.sql, SQLiteMemoryStore, NumpyFlatIndex
  ingest/      two-stage extraction, dedup, checkpointed pipeline, quality gates
  retrieve/    hybrid retrieval, raw-conversation fallback, reranker (optional)
  evaluation/  benchmark loaders, runners, judge, manifests, reporting
  llm/         quota-aware rate limiter (RPM / TPM / RPD), retrying client
```

## Limitations

- No held-out result. Every number comes from development questions also used for
  prompt iteration and gate tuning.
- The store is 64% ingested and was built by the pre-P10 extractor, so
  `source_role` / `scope` are not yet exercised on real data.
- Single-writer SQLite; no concurrency lock on ingestion.
- Temporal arithmetic and cross-session aggregation are unsolved — see the failure
  taxonomy in the [engineering report](docs/ENGINEERING_REPORT.md).

## License

MIT
