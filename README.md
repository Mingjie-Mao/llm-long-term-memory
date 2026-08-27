[![English](https://img.shields.io/badge/English-2962FF?style=for-the-badge)](README.md)
[![中文](https://img.shields.io/badge/%E4%B8%AD%E6%96%87-555555?style=for-the-badge)](README.zh-CN.md)

# llm-long-term-memory

A persistent long-term memory layer for LLM applications. It turns conversations into
structured, time-bounded facts, updates those facts when they change, and falls back to
the original conversation when compression has dropped the detail a question needs.

It is built around three constraints: a full chat history in the context window is
expensive and grows without bound; a vector store over raw conversation cannot express
that facts change; and compressing conversations into facts fixes both but is lossy.

**Tech stack** · Python · FastAPI · SQLite + FTS5 · all-MiniLM-L6-v2 · Gemini · MCP · Docker

**Core features** · Temporal facts · Supersession · Provenance to source turn ·
Conditional raw fallback · Explainable retrieval

**Interactive demo — <https://lltm-memory.pages.dev>** (bilingual). Three questions
show a fact being superseded, a dropped link recovered from the raw archive, and
memory changing what the answer says. The data on it is fictional and written for the
demo; the measured numbers are below.

## Core design

**Structured memory.** Each fact is a typed row keyed by `(subject, predicate,
object)`, with a `scope` saying why it is worth keeping. Who said it (`source_role`) and
who it is about (`subject`) are separate columns, so a third-party fact and an assistant
recommendation both have somewhere to live.

**Fact updates.** Facts are bi-temporal: `event_time` is when something was true in the
world, `ingested_at` when the system learned it. A new value on the same
`(user_id, subject, predicate)` key closes the old one's validity window and marks it
`superseded` rather than deleting it.

**Raw-conversation fallback.** The original turns stay in the same database with a BM25
index. The answerer returns a structured verdict rather than prose, and only a
`need_source` verdict pays for a second pass over raw text. Attaching evidence to every
answer was measured instead, and cost 3x the context for no detectable gain.

The case this exists for: a memory reading *"the assistant recommended a Mayo Clinic
resource about desk posture"* is true and ranks first for a question about that
recommendation, but cannot answer *"what was the URL?"* — extraction kept the gist and
dropped the link. The archive recovers the original turn.

**Provenance.** Every memory resolves to a source session, turn index and character
span. Search returns what matched *and what was rejected, with a reason* —
`superseded` or `below_rank`.

## Architecture

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
              │           memory retrieval
              │                       ↓
              │               enough to answer?
              │              ┌────────┴────────┐
              │             yes                no
              │              ↓                  ↓
              └──────────> answer      raw-source recovery
                                                ↓
                                       source-cited answer
```

| Layer | Choice |
|---|---|
| Store | SQLite (WAL) + FTS5, one file, no service dependency |
| Vectors | Exact flat inner-product over normalized 384-dim embeddings (numpy) |
| Embeddings | `all-MiniLM-L6-v2`, run locally — API quota is the binding constraint |
| LLM | `gemini-3.1-flash-lite` extracts, `gemini-3.5-flash-lite` answers, `gemma-4-31b-it` judges. Pinned ids, never `-latest`, and three separate quota pools |
| Service | FastAPI; REST, MCP and the inspector share one service object |

## Results

**Held-out set** — 100 questions from
[LongMemEval-S](https://github.com/xiaowu0162/LongMemEval) that informed no decision,
run once on a frozen and hashed system:

| | |
|---|---:|
| Final accuracy | **70.0%** |
| From structured memory alone | 50.0% |
| Recovered by the raw archive | 20.0% |
| Median context tokens | 1,468 |

Two further runs of the identical configuration score 71 and 73, so the band is 70–73,
mean 71.3. `70.0%` is reported because the protocol committed to the single shot before
the repeats existed.

**Development set (`dev50`), with baselines** — every design decision was made here:

| Variant | Accuracy | Median context tokens |
|---|---:|---:|
| `full_context` — the whole transcript | 56.0% | 109,260 |
| `naive_rag` — vector search over raw sessions | 54.0% | 13,057 |
| Structured memory only | 56.0% | 1,415 |
| Structured memory + conditional fallback | **72.0%** | 1,455 |
| v1 structured memory | 26.0% | 465 |

Two limits belong next to these numbers rather than in a footnote. `dev50` was used for
prompt iteration and threshold tuning, so its 72.0% measures the fit of those choices as
much as the system. And **the baselines were never run on the held-out set**, so
"structured memory ties naive RAG" has no unseen evidence behind it; closing that gap is
the first registered arm of the next round.

An ablation attributes the gain to the extraction rewrite rather than to attaching raw
evidence by default; the [system design report](docs/REPORT.md) has the paired numbers.

## Quick start

```bash
docker compose up --build
curl localhost:8000/healthz
```

The inspector is at <http://localhost:8000>. The image contains no datasets, models,
credentials or database — the store arrives on a mounted volume. `GEMINI_API_KEY` is
optional; without one the service runs read-only and says so.

Ingest a turn and search it back:

```bash
curl -X POST localhost:8000/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"user_id": "alice", "role": "user", "content": "I stopped drinking coffee last month."}'

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

<details>
<summary>Running without Docker</summary>

```bash
uv sync --group dev --extra api --extra embed
uv run uvicorn llm_long_term_memory.api.app:app --port 8000
```

For development and the benchmark harness:

```bash
uv sync --group dev --extra api --extra llm --extra embed
uv run pytest
uv run lltm --help
```
</details>

## Interfaces

Every read requires a `user_id` and is namespace-isolated. Reading another namespace's
memory returns **404, not 403** — 403 would confirm the id exists.

<details>
<summary>REST endpoints</summary>

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
</details>

<details>
<summary>MCP server</summary>

```bash
uv sync --extra mcp --extra embed
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

| Tool | |
|---|---|
| `search_memory` | Recall what is known, with the reason each memory was selected |
| `remember` | Store a turn; returns the memories it produced |
| `search_conversations` | Recover original wording when a memory lacks the detail |
| `get_timeline` | How one fact changed over time |
| `forget` | Mark a memory evicted; never a hard delete |

Every tool takes an explicit `user_id`; there is no ambient session identity.
</details>

## Limitations

- **No baselines on the held-out set.** `heldout100` was run for this system only, so
  `70.0%` has no unseen point of comparison. Every baseline number here is `dev50`.
- **The extractor is known-lossy and was not changed.** A randomised control shows
  batch size 15 yields 2.6 memories per session against 12.7 at batch size 1. Every
  number here sits under that ceiling.
- **Paired results come from one run per arm.** Three repeats of `heldout100` flip 6 of
  100 verdicts, which is enough to move a comparison on tens of questions across a
  significance threshold by itself.
- **Four of five retrieval signals carry zero weight, and enabling them measured worse.**
  `recency` was also retested at a corrected half-life and still carries no information —
  the gold session is not preferentially recent.
- **Not a product.** `user_id` comes from the request body rather than a trusted token,
  deletion is a status change rather than an erase, SQLite is single-writer, and no
  restore drill has been run.

## Documentation

| | |
|---|---|
| [System design report](docs/REPORT.md) · [中文](docs/REPORT.zh-CN.md) | Full data-flow walkthrough: write path, storage, retrieval, evaluation protocol, ablations, failure analysis |
| [Design decisions](docs/DECISIONS.md) | 30 numbered decisions with the measurement behind each |
| [Engineering report](docs/ENGINEERING_REPORT.md) | Earlier chronological write-up, including the mixed-extractor-store incident |
| [Roadmap](docs/ROADMAP.md) | What is done, what is next |
| [Data protocol](results/data-protocol.md) | How the five question sets are allowed to be used |
| [`results/`](results/) | Pre-registrations, raw runs and per-experiment write-ups |

## License

[MIT](LICENSE)
