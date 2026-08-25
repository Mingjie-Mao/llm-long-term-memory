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
dropped the link. A vector store would fail here and stay failed.

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

---

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
              │      hybrid retrieval (5 signals built,
              │       1 weighted in the shipped config)
              │                       ↓
              │               enough to answer?
              │              ┌────────┴────────┐
              │             yes                no
              │              ↓                  ↓
              └──────────> answer      raw-source recovery
                                                ↓
                                       source-cited answer
```

| | |
|---|---|
| **Remember** | Conversations become typed, time-bounded facts — who said it, who it is about, why it is worth keeping |
| **Update** | "I moved to Sydney" supersedes "I live in Canberra" without deleting it |
| **Retrieve** | Five weighted signals — semantic, BM25, recency, importance, entity overlap. **The shipped configs weight only semantic** ([why](#limitations)) |
| **Recover** | When structured memory is insufficient, search the original conversation instead of failing |
| **Explain** | Every memory traces to its source turn; every *omission* has a stated reason |

The last row is the differentiator. Search returns what matched **and what was
rejected, with the reason** — `superseded` (no longer true) or `below_rank` (still
true, lost on score). An absence without an explanation is indistinguishable from a
bug.

**The raw archive is a storage decision, not a retrieval algorithm.** Keeping the
original turns is what makes lossy extraction recoverable; *how* they are found is
an independent choice.

| Layer | Choice | Why |
|---|---|---|
| Store | SQLite (WAL) + FTS5 | One file, no service dependency; BM25 for free |
| Vectors | Exact flat inner-product (numpy) | Thousands of memories; ANN would add a dependency to save microseconds |
| Embeddings | `all-MiniLM-L6-v2`, local, 384-dim | Corpus-wide embedding is ~53M tokens; API quota is the binding constraint |
| LLM | Three roles, three model ids | `gemini-3.1-flash-lite` extracts, `gemini-3.5-flash-lite` answers, `gemma-4-31b-it` judges. Pinned, never `-latest` — a floating alias silently invalidates comparisons between runs |
| Service | FastAPI, one composition root | REST, MCP and the inspector are clients of the same service object |

Three separate model ids, not one — the judge must not grade prose from the model
that wrote it, and the free tier meters requests **per model**, so separate ids are
what keeps a large ingest from starving evaluation. All four are pinned in
[`configs/fallback.yaml`](configs/fallback.yaml).

---

## See it running

| Demo | What it shows |
|---|---|
| [`/?demo=mayo`](http://localhost:8000/?demo=mayo) | Compression dropped a URL; the archive got it back |
| [`/?demo=battery`](http://localhost:8000/?demo=battery) | Memory sufficed — the reply *applies* a remembered preference, no fallback |
| [`/?demo=timeline`](http://localhost:8000/?demo=timeline) | One fact, five values over time, with the supersession chain |
| [`/?demo=collectibles`](http://localhost:8000/?demo=collectibles) | A multi-valued key whose facts coexist rather than replace |

Each is a **recorded run** — real output from a real model, labelled as recorded, so
the page costs nothing to open; **Run live answer** re-executes it. Recordings carry a
fingerprint of the store and prompt versions behind them, and say **stale** rather
than pretending to be current when either moves.

---

## Results

Two numbers matter. The first is the only one measured on data no decision was made
against.

**Held-out, 100 unseen questions from [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval), run once on a frozen and hashed system:**

| | |
|---|---:|
| **final accuracy** | **70.0%** |
| answered from structured memory alone | 50.0% |
| rescued by the raw archive | 20.0% |
| median context tokens | 1,468 |

Two further runs of the identical configuration score 71 and 73, so the band is
**70–73, mean 71.3**. `70.0%` is what is reported, because the protocol committed to
the single shot before the repeats existed.

**Development set, with baselines — `dev50`, where every design decision was made:**

| Variant | Accuracy | Median context tokens |
|---|---:|---:|
| `full_context` — the whole transcript | 56.0% | 109,260 |
| `naive_rag` — vector search over raw sessions | 54.0% | 13,057 |
| Structured memory only | 56.0% | 1,415 |
| **Structured memory + conditional fallback** | **72.0%** | **1,455** |
| v1 structured memory | 26.0% | 465 |

**75x less context than the transcript, and more accurate on this set.**

Two caveats that belong next to these numbers, not in a footnote: `dev50` was used
for prompt iteration and threshold tuning, so its 72.0% measures the fit of those
choices as much as the system — and **the baselines were never run on the held-out
set**, so "structured memory ties naive RAG" has no unseen evidence behind it.
[Full evidence](#evidence) · [Limitations](#limitations)

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

<details>
<summary>Image size, and a smaller build</summary>

**2.95GB**, most of it PyTorch for the embedding model. The default PyTorch install
pulls the CUDA wheel — 24.4GB of GPU runtime for a container that will never see a
GPU — so the Dockerfile installs the CPU wheel and deletes the orphaned CUDA
packages. Still not small; a deployment that wants small should run the encoder out
of process, which is [on the roadmap](docs/ROADMAP.md).

For a read-only image without semantic search, build with `--build-arg EXTRAS="api"`.
`/healthz` then reports `degraded` and search returns 503 rather than failing
obscurely.
</details>

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

<details>
<summary>What <code>search</code> returns, and the full endpoint list</summary>

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
</details>

Every read requires a `user_id` and is namespace-isolated. Reading another
namespace's memory returns **404, not 403** — 403 would confirm the id exists.

---

## Connect from an MCP-compatible client

[MCP](https://modelcontextprotocol.io) is how an agent connects to this; REST is how a
program does. Both wrap the same service object, so the tools cannot drift from the
behaviour that was measured.

```bash
uv sync --extra mcp --extra embed
uv run lltm mcp                    # stdio
uv run lltm mcp --transport http   # streamable HTTP
```

<details>
<summary>Claude Desktop / Cursor config</summary>

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
</details>

| Tool | |
|---|---|
| `search_memory` | Recall what is known, with the reason each memory was selected — and with `explain`, why others were not |
| `remember` | Store a turn; returns the memories it produced |
| `search_conversations` | Recover the original wording when a memory is on topic but lacks the detail |
| `get_timeline` | How one fact changed over time |
| `forget` | Mark a memory evicted; never a hard delete |

Every tool takes an explicit `user_id`. There is no ambient session identity: an
agent serving several people must say which one it is acting for, and the store
enforces the boundary rather than trusting the caller.

---

## Evidence

### The held-out run

100 questions no decision was ever made against, run **once** on a
[frozen and hashed system](results/frozen/p10-final/README.md), behind five gates —
one of them asserting the result file did not yet exist.

| | dev50 — every decision made here | **heldout100 — unseen** | |
|---|---:|---:|---:|
| **final accuracy** | 72.0% | **70.0%** | -2.0pp |
| answered from structured memory alone | 54.0% | 50.0% | -4.0pp |
| rescued by the raw archive | 18.0% | 20.0% | +2.0pp |
| fallback fired | 32% | 36% | +4pp |
| …and was right | 56.2% | 55.6% | -0.6pp |
| source-session recall | 93.5% | 94.0% | +0.5pp |

**The headline generalises. Almost nothing underneath it does.**

Against the three-run mean the gap to `dev50` is **-0.7pp**, not -2.0pp: the single
shot landed at the bottom of its own range, so the system generalises better than the
pre-registered number says. Six of 100 questions disagree across three runs, and
**all six are `temporal-reasoning` or `multi-session`** — the two types whose answers
are derived rather than looked up. The other four types are bit-identical across all
three runs. [The measurement](results/heldout-variance.md).

| question type | dev50 | heldout100 | Fisher p |
|---|---:|---:|---:|
| knowledge-update | 8/8 = 100% | **10/15 = 66.7%** | 0.122 |
| single-session-user | 7/7 = 100% | 12/14 = 85.7% | 0.533 |
| single-session-assistant | 6/6 = 100% | 9/11 = 81.8% | 0.515 |
| multi-session | 7/13 = 53.8% | 19/27 = 70.4% | 0.480 |
| single-session-preference | 2/3 = 66.7% | 4/6 = 66.7% | 1.000 |
| **temporal-reasoning** | 6/13 = 46.2% | **16/27 = 59.3%** | 0.509 |

Every `dev50` cell holds 3 to 13 questions. Its three 100% categories all fell, its
two worst both rose, and none of it is distinguishable from noise — **`dev50`'s
per-type numbers never carried information**, which nobody said out loud while they
were being used to choose what to build. Two movements change what to work on next:

- **`temporal-reasoning` is 59.3%, not 46.2%** — still the worst category, no longer
  the wall this README used to call it. The old estimate rested on 13 questions.
- **`knowledge-update` fell from 100% to 66.7%**, now joint-worst and invisible on
  `dev50` because 8 of 8 is not evidence. All five failures recalled the gold session
  *and* the gold evidence; four are downstream of retrieval, in how supplied facts are
  read and combined.

One question had its evidence refused by the provider's content filter; excluding it,
the numbers are 69.7% / 49.5% / 20.2% over 99. `heldout100` is now spent — nothing may
be tuned against it, and `dev100` was frozen before this result was read.

### Where the gain came from

An early draft credited evidence hydration. A causal ablation reversed that:

| Step | Δ | Paired |
|---|---:|---|
| extraction rewrite | **+29.0pp** | 11W-2L, **p = 0.022** |
| evidence hydration | +3.2pp | 2W-1L, p = 1.000, at 3x the context |

Source-session recall is 93.5% for *both*, which confirms it — hydration runs after
retrieval and cannot change what is recalled. So hydration was demoted from an
always-on stage to a conditional fallback.

### What the 72% is made of

54.0% comes from structured memory alone and 18.0% is rescued by the archive. Memory
on its own **ties `naive_rag`** — the archive is what puts the product ahead. Quoting
the total undivided reads as though the memory layer answered them all.

The two arms differ in three `fallback` settings and nothing else, checked before the
run rather than asserted after. Paired over the same 50 questions the fallback fixed 9
and broke 1, exact McNemar **p = 0.022**.

> **That p-value is not established.** It is one run per arm, on the same 50
> questions that exposed the defect whose fix produced it (66.0% → 72.0%). Repeats
> measured 6 of 100 questions flipping between identical runs; on 50 questions one
> flip turns 9W-1L into 8W-2L, `p = 0.109`. The effect size is large and probably
> real. The significance as published is beyond what one run per arm can support.

### Measured and not shipped

**Cross-encoder reranking.** At k=10 the reranked and plain arms answered all 31
questions **identically** — zero disagreements — while reranking *lowered* source
recall at every k (93.5% → 90.3% at k=20). Kept behind an optional `rerank` extra,
off by default. [Details](results/rerank-pareto.md).

**Dense retrieval over the raw archive.** A constructed query set suggested BM25
collapses on paraphrases (6.9% R@1). Manual inspection showed the construction had
deleted the *questions*, not just their vocabulary — *"How long have I been
collecting vintage cameras?"* became `"long"`. That evidence was discarded;
hand-written paraphrases retrieve the gold turn at **rank 1**. Deferred, not
rejected. [Details](results/raw-recall-diagnostic.md).

**A learned utility predictor for budget packing.** Held-out RMSE 0.310 against a
mean-baseline 0.263 — it lost to predicting the mean. [Details](results/p6-pilot.md).

### Live regression

Seven questions that every variant previously failed score **6/7, identical across
three runs**, including the Mayo case end to end.
[Details](results/live-regression-v2.md).

<details>
<summary>The clean store briefly broke the Mayo case, and the obvious diagnosis was wrong</summary>

After a rebuild that *improved* retrieval, the question this README opens with failed
in both arms. The tempting explanation was the level branch — `RawFallback.recover`
took source turns whenever retrieval returned anything, with nothing checking the
memories were about the question. A genuine defect, but not this one.

What broke it was truncation. Retrieval **had** found the right conversation. But
`turns_for_memories` returns turns ordered by session id, the code kept
`[:max_turns]`, and the gold turn sat tenth of sixteen alphabetically — the three kept
were all from a conversation about live music. Better recall meant more candidates,
and more candidates meant the answer was sliced off. On the smaller mixed store the
same question retrieved nothing and never met the slice, which is why it survived
unseen for so long.

Both are the same omission — nothing ranked the candidates against the question — so
ranking them fixes both. `single-session-assistant` went 5/6 → **6/6**, and the arm
66.0% → 72.0%.
</details>

---

## Failure analysis

Not every wrong answer is a retrieval failure. Tracing all 14 `dev50` failures to the
layer that loses the answer:

| stage | n | |
|---|---:|---|
| S0 source | 0 | the fact is not in the conversation |
| **S1 extraction** | **10** | it never became a memory |
| S2 lifecycle | 1 | it became one, then was merged or superseded away |
| S3 eligibility | 0 | filtered before ranking |
| **S4 retrieval** | **0** | eligible, never reached the context |
| S4b composition | 1 | in context and top-ranked, still not used |
| S5 reasoning | 2 | everything supplied and correct, answer still wrong |

**Ten of fourteen die at extraction and none at retrieval.** Every module then got an
oracle ceiling *before* anything was built on it — inject the missing fact, run the
real pipeline, count what changes:

| module | ceiling | measured |
|---|---:|---|
| extraction | 10 | 4 fixed of 9 tried |
| **retrieval** | **0** | not run — no failure is lost here |
| composition | 1 | 3 of 4 runs |
| reasoning | 2 | 0 |

That extraction's oracle fixed only 4 of 9 is the more useful half: there is a second
defect behind the missing facts, so perfect extraction alone does not reach the
ceiling. [Details](results/failure-stages.md).

**Batched extraction loses memories by position.** Same conversation, same batch size,
same prompt, same neighbours — moved to the back of the request it yields a third as
much (4.5 → 1.6 memories/session, paired within session, exact McNemar p < 0.0001). At
batch size 1 zero-yield goes to 0.0% and yield to 12.7, at 12x the requests. Nothing
has been changed on the strength of this yet: every number here was produced at batch
size 15. [Details](results/batch-position-pilot.md).

---

## Engineering

- **529 tests**, CI across ubuntu / windows / macos, 80% line coverage
- **Result versioning** — every evaluation row records its answerer prompt, judge
  prompt and extractor version; the extractor version comes from the *store*, not the
  checkout, because it describes the data being evaluated
- **Ingestion fingerprinting** — resume compares a hash of the prompts, schema, model
  id, batch size and dedup threshold, and names which component moved. Added after a
  store was found to have been written by two extractor generations with nothing able
  to notice ([the incident](docs/ENGINEERING_REPORT.md))
- **Frozen manifests** — reported runs name an explicit question set, because a sample
  size is not an experiment identity
- **Gates exit non-zero** — a gate that prints its verdict and exits 0 is not a gate
- **Structured logging** — one JSON event per request with request id, latency and a
  config fingerprint; no memory content, no credentials
- **Encoder warm-up** at startup: first request 12,486 ms → 181 ms

Full write-up: [System Design Report](docs/REPORT.md) ·
[Design decisions](docs/DECISIONS.md) · [Roadmap](docs/ROADMAP.md) ·
[Engineering Report](docs/ENGINEERING_REPORT.md) · [中文版](docs/REPORT.zh-CN.md)

---

## Development

```bash
uv sync --group dev --extra api --extra llm --extra embed
uv run pytest
uv run lltm --help
```

```
src/llm_long_term_memory/
  api/         FastAPI service, inspector, recorded demo runs
  store/       schema.sql, SQLiteMemoryStore, NumpyFlatIndex
  ingest/      two-stage extraction, dedup, checkpointed pipeline, quality gates
  retrieve/    hybrid retrieval, raw-conversation fallback, reranker (optional)
  evaluation/  benchmark loaders, runners, judge, manifests, reporting
  llm/         quota-aware rate limiter (RPM / TPM / RPD), retrying client
```

<details>
<summary>Benchmark workflow</summary>

```bash
uv run lltm data download --variant s
uv run lltm doctor
uv run lltm ingest run --store-name two-stage-p10
uv run lltm eval freeze dev50 --n 50
uv run lltm eval run two_stage --questions results/manifests/dev50.json
uv run lltm eval compare naive_rag two_stage
```

`--questions` takes a frozen manifest. `--limit` re-samples and is for exploration
only — a stratified subset is scattered across the split, not a prefix of it.
</details>

---

## Limitations

- **No baselines on the held-out set.** `heldout100` was run for this system only, so
  `70.0%` has no unseen point of comparison, and "structured memory ties naive RAG" is
  a `dev50` claim. This is the largest gap in the evidence. Closing it is the first
  registered arm of the next round, on `dev100` and `test100`.
- **Every paired p-value here is one run per arm, and that is too few.** Three runs of
  `heldout100` flip 6 of 100 verdicts. Repeats are now protocol
  ([data-protocol.md](results/data-protocol.md)); nothing published before 2026-08-20
  follows it.
- **`dev50`'s per-type numbers carried no information, and were used anyway.** Each
  cell held 3 to 13 questions.
- **`knowledge-update` is 66.7% unseen and was 100% on `dev50`.** All five failures
  recalled the gold session and the gold evidence; four are downstream of retrieval.
  The store offers a mechanism: supersession keys on `(user_id, subject, predicate)`,
  extraction invents a fresh predicate per fact, and **546 of 622 replacement signals
  sit alone on their key — 96% of them beside sibling predicates on the same subject**.
  55.6% of the store sits on multi-valued keys that can never supersede. Evidenced, but
  not yet shown to cause these five failures.
- **The extractor is known-lossy and was not changed.** Batch size 15 yields 2.6
  memories per session against 12.7 at batch size 1. Every number here sits under that
  ceiling.
- **14.5% of substantive sessions yield no memory at all** (304 of 2,096). Recorded as
  a baseline, not yet explained.
- **Four of five retrieval signals carry zero weight, and turning them on is worse.**
  Measured offline over 150 questions: every added signal degrades ranking, from -5.3pp
  (`importance`) to -17.3pp (`entity`). `recency` is a **no-op** — its 30-day half-life
  against a corpus whose freshest memory is 932 days old sends every score to ~1e-12.
  Keep `semantic` alone. [Details](results/retrieval-weights.md).
- **`stores/two-stage-hydrated.db` mixes two extractor generations** — 4,843 pre-fix
  rows and 2,265 after. Results on it stay labelled diagnostic.
- **Single-writer SQLite**, with a cross-process lock added after two simultaneous
  ingests overwrote each other's quota accounting.
- **Temporal arithmetic is unsolved** and the fallback does not touch it — dates have
  to be computed, not looked up.
- **Not a product yet.** No trusted identity (`user_id` is taken from the request
  body), no hard delete, no restore drill, no cost accounting. Acceptance criteria for
  each are in [the productization plan](docs/PRODUCTIZATION_V2_PLAN.md).

## License

MIT
