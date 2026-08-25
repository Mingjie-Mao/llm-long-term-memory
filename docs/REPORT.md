# LLTM System Design Report

A complete technical description of an LLM long-term memory layer. Written for someone
who knows Python, LLMs and the basics of RAG but has never seen this project: after
reading it you should be able to say what happens to a conversation as it moves through
the system, why each component is built the way it is, and which experiment supports
each decision.

**Every implementation detail here is taken from the current code and the frozen
configuration.** Where the code cannot establish something, this report says *not
established from the current material* rather than filling the gap with a guess.

Two things this report is careful never to conflate: *what the architecture supports*
and *what the shipped configuration actually runs*; and *correlation* and *causation*.
Both distinctions matter a lot in this project.

Companion documents: [`ENGINEERING_REPORT.md`](ENGINEERING_REPORT.md) is the earlier
chronological write-up, [`DECISIONS.md`](DECISIONS.md) holds the numbered design
decisions, and [`REPORT.zh-CN.md`](REPORT.zh-CN.md) is this document in Chinese.

Document version: 2026-08-25.

---

## Tech stack

| Layer | What | Version | Note |
|---|---|---|---|
| Language | Python | `>=3.11` | Uses `StrEnum` (3.11+) and `Literal` type aliases |
| Packaging / build | `uv` + `hatchling` | — | `uv.lock` pins every transitive dependency |
| Validation | `pydantic` / `pydantic-settings` | `>=2.7` / `>=2.3` | Extraction output, config and API models all go through pydantic schemas |
| CLI | `typer` + `rich` | `>=0.12` / `>=13.7` | `lltm` and `llm-long-term-memory` are two entry points into one Typer app |
| Numerics | `numpy` | `>=2.0` | The vector index *is* a numpy matrix |
| HTTP client | `httpx` | `>=0.27` | Talks to the LLM API |
| Logging | `structlog` | `>=24.1` | One JSON event per request |
| Config | `pyyaml` | `>=6.0` | `configs/*.yaml` |
| Timezone | `tzdata` (Windows only) | — | Windows ships no system tz database, so `zoneinfo` cannot resolve `America/Los_Angeles` — and the provider resets its daily quota at Pacific midnight, which is exactly what the rate limiter counts against. Caught by the Windows CI job |

**Optional extras, deliberately split apart:**

| Extra | Contents | Why optional |
|---|---|---|
| `embed` | `sentence-transformers>=3.0` | Pulls torch (~2GB). Dataset and statistics work does not need it |
| `llm` | `google-genai>=1.0` | Only extraction and answering need an API key |
| `api` | `fastapi>=0.115`, `uvicorn[standard]>=0.30` | The CLI and the test suite should not depend on a web framework |
| `rerank` | `sentence-transformers>=3.0` | Cross-encoder reranking is **measured and not enabled**. Its own extra so the dependency is deliberate rather than incidental |
| `mcp` | `mcp>=1.2` | A REST-only deployment should not carry a protocol implementation it never speaks |
| dev | `pytest>=8.2`, `pytest-cov>=5.0`, `ruff>=0.6` | — |

**Storage and retrieval stack:**

| | |
|---|---|
| Relational store | SQLite (`journal_mode=WAL`, `foreign_keys=ON`), one file |
| Full-text search | SQLite FTS5 virtual tables, `tokenize='porter unicode61'`, external-content tables kept in sync by triggers |
| Lexical ranking | FTS5's built-in `bm25()` — **returns negative values**, more negative is better |
| Vector storage | `.npy` (float32 matrix) + `.ids.json`, sitting beside the `.db` |
| Vector search | Exact numpy inner product + `argpartition` top-k. **No FAISS, no ANN** |

**Models — all pinned, never `-latest`:**

| Role | Model id |
|---|---|
| Extractor (Stage A + Stage B) | `gemini-3.1-flash-lite` |
| Answerer (the system under test) | `gemini-3.5-flash-lite` |
| Judge | `gemma-4-31b-it` |
| Embedder (runs locally) | `sentence-transformers/all-MiniLM-L6-v2`, 384-dim |

**Engineering:**

| | |
|---|---|
| Tests | pytest, **529 tests**, 80% line coverage |
| Lint | ruff, `line-length = 100` |
| CI | ubuntu / windows / macos matrix |
| Warnings are errors | `filterwarnings = ["error::EncodingWarning"]` — an encoding defect cannot come back quietly |
| Container | Docker + compose, 2.95GB image (CPU torch) |
| Interfaces | REST (FastAPI), MCP, Memory Inspector — all three are clients of the same service object |

**What is deliberately *not* in this stack** (see section 4): no separate database
server, no vector database, no Elasticsearch, no message queue, no ORM. At the current
scale — single machine, single writer, 18,017 memories at most — each of those adds
operational surface without solving a problem the project has. The moment this becomes
a multi-tenant service, SQLite is the first thing to replace.

---

## 0. Executive summary

**The problem.** An LLM application has to remember what a user said months ago.
Putting the whole transcript in the context window is expensive and grows linearly;
retrieving raw conversation chunks with a vector store cannot express that *facts
change* — after someone moves house, the old address and the new one come back
together, with nothing to say which one still holds.

**The core idea.** Compress conversations into structured facts — compact, updatable,
explainable — **and keep the original conversation forever**. Compression is lossy.
This project does not try to eliminate the loss; it requires only that the loss be
**recoverable**: when structured memory cannot answer, the system goes back to the
original turns.

**The architecture.** Two paths:

- **Write path**: conversation → raw archive → two-stage extraction → structured
  memory → temporal resolution → embedding → index
- **Read path**: query → hybrid retrieval → candidate scoring → context assembly →
  answerer → sufficiency verdict → (if insufficient) raw-archive fallback → answer
  with provenance

**The headline result.** On 100 questions that had never informed any decision, run
once: **70.0% accuracy**, of which 50.0% is answered from structured memory alone and
20.0% is rescued by the raw archive, at a median of 1,468 context tokens. On the
development set the same system scores 72.0%, against 56.0% for putting the whole
transcript in the prompt and 54.0% for ordinary RAG — at 1/75 and 1/9 of their context
respectively.

**The three largest limitations, in order:**

1. **No baselines on the held-out set.** That run covered this system only, so
   "structured memory ties naive RAG" remains a development-set claim.
2. **The extractor is known-lossy and was not changed.** At batch size 15 it yields
   2.6 memories per session; at batch size 1 it yields 12.7. This was established by a
   randomised controlled experiment. Every number here sits under that lowered ceiling.
3. **Four of the five retrieval signals carry zero weight in production.** The
   architecture supports a five-signal weighted fusion; the frozen config sets
   `semantic=1.0` and the other four to `0.0`.

---

## 1. Problem definition and design goals

### 1.1 Option A: full context

**How it works.** On every question, paste the user's entire conversation history into
the prompt and let the model find what it needs.

**What is good about it.** Nothing is lost, the implementation is string concatenation,
and there is no retrieval error.

**What is wrong with it.**
- **Cost grows linearly with history.** Measured here: a median of **109,260 tokens**
  per question.
- **Putting it in is not the same as using it.** It scores **56.0%** on the development
  set — beaten by an approach with 75x less context. Information in a long context gets
  diluted; that is measured here, not cited from elsewhere.
- **There is no notion of fact state.** "I live in Canberra" from three years ago and
  "I moved to Sydney" from last year are both in the prompt, and the model has to
  re-derive which one holds on every single question.

### 1.2 Option B: naive RAG over conversation history

**How it works.**

```
conversation → split into session / turn chunks → embed each chunk → vector store
query → embed query → take the k nearest chunks → paste into prompt → answer
```

**What is good about it.** Context drops to a median of **13,057 tokens**, about an
eighth of full context. The implementation is simple, and what comes back is the
*original wording* — exact URLs, model numbers and figures survive intact. It scores
**54.0%** on the development set.

**What it does not naturally solve.**
- **It retrieves chunks, not facts.** A session may contain twenty things of which one
  is relevant, and the whole chunk is pulled in.
- **It cannot express fact evolution.** Vector similarity has no way to know that "I
  moved to Sydney" should retire "I live in Canberra". Both come back, and the stale
  one may rank higher because its wording is closer to the question.
- **It cannot answer questions requiring cross-session aggregation.** "How many museums
  did I visit in February?" — no single chunk states that number.

### 1.3 Option C: structured long-term memory

**How it works.** Use an LLM to read the conversation into individual typed,
time-bounded facts:

```
source:  "I quit coffee last month, I'm drinking matcha now."
   ↓
memory:  content    = "The user stopped drinking coffee."
         subject    = "user"
         predicate  = "drinks"
         scope      = "profile"
         event_time = 2026-07-15
         update_op  = "replaces"
```

**Why it is worth doing.**
- **Compact**: a fact is tens of tokens; median context drops to **1,455**.
- **Updatable**: a fact has a `(subject, predicate)` key, so a new value can retire the
  old one instead of coexisting with it.
- **Explainable**: every fact traces back to the turn it came from, and facts passed
  over at retrieval time carry a reason.

### 1.4 The fundamental trade-off: compression is lossy

This is the most important section in the report.

Extraction drops things, and what it drops is often exactly the identifier the question
is about. A real case:

```
Original assistant turn, three months ago:
  "...I recommend this Mayo Clinic video:
   How to Sit Properly at a Desk to Avoid Back Pain
   https://www.youtube.com/watch?v=UfOvNlX9Hh0 ..."

Extracted memory:
  "The assistant recommended a Mayo Clinic resource about desk posture."

Today's question:
  "What was the Mayo Clinic YouTube video you recommended?"
```

That memory **is true**, and retrieval correctly ranks it first — it is precisely about
Mayo Clinic. It still cannot answer the question, because the URL was thrown away in
compression. A pure vector store fails here and **stays** failed: re-running, changing
the embedding model or raising top-k changes nothing, because that string is not in the
store at all.

**This project's answer is not "make extraction better". It is "accept that extraction
is lossy, and guarantee the loss is recoverable":**

> **Lossy structured memory + a recoverable raw conversation archive**

The original turns live permanently in the same database, with a full-text index. When
structured memory cannot answer, the system searches them.

#### Two uses of raw text that must not be conflated

Many systems blur these together. This project measured them separately and got
opposite results:

| | **Always-on hydration** | **Conditional fallback** |
|---|---|---|
| When raw text is fetched | On **every** answer, attached to the retrieved memories | **Only** when the answerer declares memory insufficient |
| Context cost | About **3x** | Near zero on average — measured trigger rate 36% |
| Measured effect | **+3.2pp**, 2W-1L, p = 1.000 | **+16.0pp** on the development set (56.0% → 72.0%) |
| Current status | **Implemented, demoted, off by default** | **The shipped product path** |

Always-on hydration bought an effect indistinguishable from zero at a certain cost of
three times the context — which is a slow slide back toward naive RAG. Conditional
fallback spends that money only on the 36% of questions that need it.

> ⚠️ There is a sentence here that is easy to read backwards. This project also
> concludes that "the gain came from the extraction rewrite (+29.0pp), not from
> hydration (+3.2pp)". **That statement is about always-on hydration, not about the raw
> archive.** Stated precisely:
>
> *Two-stage extraction produced the largest upstream improvement, while always-on
> raw-evidence hydration did not justify its token cost. Conditional fallback serves a
> different role: it recovers details lost by compression, only when structured memory
> is insufficient.*
>
> On the development set the archive independently contributes 18.0 percentage points
> (54.0% → 72.0%). It is not decoration.

### 1.5 Design goals

| Goal | Testable form |
|---|---|
| Compact | Context at least an order of magnitude below the full transcript |
| Updatable | A new value on the same `(subject, predicate)` removes the old one from retrieval, without deleting it |
| Recoverable | A detail extraction dropped can still be answered through the raw path |
| Explainable | Every memory traces to its source turn; every **omission** has a stated reason |
| Isolated | Cross-user reads and writes are impossible, even knowing the id |

---

## 2. End-to-end architecture

```
                          ┌──────────────────────┐
                          │      Conversation     │
                          └───────────┬──────────┘
                                      │
        ┌─────────────────────────────┴─────────────────────────────┐
        │                     WRITE PATH                            │
        │                                                           │
        │   turns (raw archive)          Stage A: bare facts        │
        │   sessions                             ↓                  │
        │   turns_fts (BM25 index)       Stage B: key + update_op   │
        │        │                               ↓                  │
        │        │                        memories table            │
        │        │                    (bi-temporal + provenance)     │
        │        │                               ↓                  │
        │        │                     temporal resolution           │
        │        │                               ↓                  │
        │        │                embedding → NumpyFlatIndex (.npy) │
        └────────┼──────────────────────────────┬───────────────────┘
                 │                              │
        ┌────────┼──────────────────────────────┼───────────────────┐
        │        │          READ PATH           │                   │
        │        │                              │                   │
        │  Query ─────────────────────────→ hybrid retrieval        │
        │        │                     (5 signals, weighted fusion)  │
        │        │                              ↓                   │
        │        │                  filtering + top_k truncation     │
        │        │                              ↓                   │
        │        │                      context assembly             │
        │        │                 (v1 flat / v2 session-coherent)   │
        │        │                              ↓                   │
        │        │                    answerer (1 call)              │
        │        │                              ↓                   │
        │        │                 status ∈ {answer,                │
        │        │                  need_source, no_evidence}       │
        │        │                     ┌────────┴────────┐          │
        │        │                  answer          need_source     │
        │        │                     ↓                 ↓          │
        │        └──────────────────→  │        raw-archive search   │
        │                              │        (BM25 over turns)    │
        │                              │                 ↓          │
        │                              │       answerer (2nd call)   │
        │                              └────────┬────────┘          │
        │                                       ↓                   │
        │                       answer + provenance + rejections     │
        └───────────────────────────────────────────────────────────┘
```

These two paths are the skeleton of this report. Section 3 covers the write path,
section 4 the storage layer they share, section 5 the read path.

**One convention that runs through everything:** answering a question costs **one**
answerer call by default. A second call is paid only when the answerer itself declares
that memory was insufficient. Extraction happens offline in batches, two calls per 15
sessions.

---

## 3. Write path — how a conversation becomes memory

### 3.1 The raw conversation archive

**What it solves.** Extraction is lossy (1.4). If the original is not kept, the loss is
permanent and the system cannot distinguish "not in memory" from "never happened".

**Input.** One session: an ordered list of `(role, content, timestamp)`.

**Output.** Rows in two tables, plus an automatically maintained full-text index.

**Schema.**

```sql
sessions(id PK, user_id, started_at, source)
turns(id PK, session_id FK→sessions, turn_index, role, content, ts)
INDEX idx_turns_session ON turns(session_id, turn_index)
```

Note that `turns` has **no `user_id` column**. Ownership is reached by joining
`session_id` to `sessions`. This is not an oversight: it means any query that skips the
join cannot see user data at all. Isolation is enforced by the schema rather than by
every query site remembering a predicate. Archive full-text search is therefore a
three-table join:

```sql
FROM turns_fts f
JOIN turns t     ON t.rowid = f.rowid
JOIN sessions s  ON s.id = t.session_id
WHERE turns_fts MATCH ? AND s.user_id = ?
```
(`store/sqlite.py:378`, `search_turns`)

**Provenance.** Every memory carries four columns pointing back at the source:

```
source_session_id  → sessions.id
source_turn_index  → which turn
source_char_start  → start offset within that turn's text
source_char_end    → end offset
```

So "where did this memory come from" is answerable down to a character range, not just
to a session. On the clean P10 store, **4,843 / 4,843 memories resolve to a source
turn** — that number is the precondition for conditional fallback, whose first level is
exactly "fetch the turns these memories are anchored to".

**Why this is a storage decision and not a retrieval algorithm.** Keeping the raw turns
and *finding* them are independent problems. The first is a schema decision that cannot
be repaired after the fact; the second (source-local lookup, BM25, dense, hybrid) can be
swapped at any time, and this project did swap it once (see 5.8). Conflating them
produces a common mistake: not keeping the raw text *because we already have vector
retrieval*.

| | |
|---|---|
| **Choice** | Full raw text in the same SQLite file, with an FTS5 index |
| **Why** | Same database as the memories, so the fallback path needs no cross-system consistency; the index keeps the archive from being write-only |
| **Alternatives** | Object storage only (unsearchable); summaries only (that *is* extraction, and is not recoverable); do not keep it |
| **Trade-off** | Database size. `train150.db` is 185MB, mostly raw text |
| **Evidence** | The Mayo case answers end to end; the archive contributes 18.0pp independently on the development set |
| **Limitation** | Sessions refused by the content filter keep their raw text but produce no memories; these are counted on their own line (1 on `heldout100`) |
| **Code** | `store/schema.sql`, `store/sqlite.py` |

### 3.2 Two-stage memory extraction

This is the most important component on the write path, and where failure concentrates:
10 of the 14 development-set failures die here.

**What it solves.** Turn natural-language conversation into structured, retrievable,
updatable facts.

**Input.** 15 sessions per batch (`ingest.sessions_per_request: 15`).

**Output.** A set of `Memory` rows.

#### A worked example

```
── input turn ────────────────────────────────────────────
session s_4f2a, turn 3, role=assistant, ts=2026-05-02
"Since you mentioned your desk setup, I recommend this Mayo Clinic
 video: How to Sit Properly at a Desk to Avoid Back Pain
 https://www.youtube.com/watch?v=UfOvNlX9Hh0"

── Stage A output (bare fact strings, 1 LLM call / batch) ──
{ "session_index": 0,
  "fact": "The assistant recommended a Mayo Clinic video about desk posture." }

── Stage B output (keying, 1 LLM call / batch) ─────────────
{ "temporal_key": "assistant_recommendation",
  "update_op":    "coexists",
  "object":       "Mayo Clinic desk posture video" }

── rule-based fill-in (0 calls) ────────────────────────────
type=episodic  entities=[mayo clinic]  importance=0.5
event_time=2026-05-02 (parsed from the turn's ts)
source_session_id=s_4f2a  source_turn_index=3  source_char_start/end=…

── final memory row ────────────────────────────────────────
content     "The assistant recommended a Mayo Clinic video about desk posture."
subject     "assistant"        ← who the fact is ABOUT
source_role "assistant"        ← who SAID it
predicate   "assistant_recommendation"
object      "Mayo Clinic desk posture video"
scope       "recommendation"
update_op   "coexists"
status      "active"
```

Note the URL is gone at Stage A. That is the lossy compression from 1.4, and the entire
reason conditional fallback exists.

#### Why two stages instead of one call

Stage A is only responsible for *which facts in this conversation are worth keeping*.
Stage B is only responsible for *giving a fact a temporal key and saying what it does to
earlier facts*. Two calls per batch instead of one takes the cost from roughly 190
requests to roughly 380.

The reason is that the alternative Stage B **was measured**. It started as regexes
(`_rule_keying`, still runnable in the code), over 148 real memories:

| Stage B implementation | Predicate accuracy | Supersessions produced |
|---|---:|---:|
| Rules (regex) | **43%** | **0** |
| LLM call | passes all four temporal-gate metrics | non-zero |

Regexes cannot key open-domain relations. That was measured, not assumed. And merging
Stage A and Stage B into one call has a different problem: a single call's output budget
would have to carry both "read fifteen conversations" and "design a key for every fact",
and the position-decay experiment below indicates the output budget is exactly this
model's bottleneck.

**Rules are kept where they have the advantage**: `type`, `entities`, `importance` and
date parsing are all rule-based and cost no LLM budget, so Stage B spends its entire
output budget on the part regexes could not do.

#### Why v1 extraction failed

The first version (`chronomem`) scored **26.0%** on the development set — 28 points
below naive RAG. Two causes:

**(1) Rules describing what to extract, which the model did not follow.** The fix was
**worked examples** in the prompt. That change was quantified separately in this
project: adding worked examples moved source fidelity from **33.6% → 36.6%**. And there
is a sharper observation attached to it — *the one instruction that had no example was
the one being ignored*.

**(2) `subject` had become a speaker flag.** In v1 `subject` was derived from a string
prefix and in practice held only `user` / `assistant`. The consequence:

| | v1 | two-stage |
|---|---:|---:|
| memories with `subject='assistant'` | 12 / 2,007 = **0.6%** | 326 / 4,843 = **6.7%** |

And **third-party facts had nowhere to live** — "Andy wore a blue shirt" fits neither
value. The observable consequence: on `single-session-assistant`, every memory variant
scored **0/4** while `full_context` and `naive_rag` both scored 4/4. A gap that uniform
cannot be a ranking problem.

**Fix** (see 3.3): split "who said it" and "who the fact is about" into two columns.

#### Where batch size 15 came from, and its known cost

`sessions_per_request: 15` was originally a quota decision: the free tier allows 500
requests per model per day, and 2,400 sessions at batch 15 is 189 batches / 378
extractor calls, which fits in one day. The quality justification at the time was that
batch size 10 had produced zero cross-session misattributions.

**A randomised controlled experiment later showed this configuration loses information.**
60 sessions (30 that yielded nothing in production, 30 that yielded normally, matched on
turn count ±2 and assistant share ±0.08), batch size fixed at 15, two random orderings
each paired with its own reverse — so a session at position `p` in one arrangement sits
at `14-p` in the other. Model, prompts, schema and decoding identical throughout; the
extractor is driven directly rather than through the pipeline, and **no store is
written**.

Paired within session (51 sessions seen at both ends):

| | Coverage view | Count view |
|---|---|---|
| yielded in front, zero in back | **8** | memories/session, front: **4.5** |
| zero in front, yielded in back | **0** | memories/session, back: **1.6** |
| exact McNemar | **p = 0.0078** | **p < 0.0001** |

Same conversation, same batch size, same prompt, same neighbours — moved to the back of
the request it yields a third as much. **Zero-yield is not a separate failure mode; it
is the tail of a continuous attenuation.**

Batch size itself:

| Batch size | Zero-yield | Memories / session | Corpus extraction requests |
|---|---:|---:|---:|
| **15 (production)** | 11.7% | **2.7** | 400 |
| 5 | 8.3% | 4.8 | 1,000 |
| 1 | **0.0%** | **12.7** | 4,800 |

Not one session in sixty yielded more under a larger batch, twice over (p = 1.7e-18).

**The cost, stated plainly: every number in this project was produced at batch size 15,
that is, on top of an extractor that drops roughly three quarters of the memories per
session.** The configuration was not changed, because changing it means rebuilding every
store (28 more quota days on the free tier) while the experimental conclusion is not yet
frozen. This is a lowered ceiling, not an unknown.

**Why the planned remedy would have been wrong.** The original plan was to re-run the
zero-yield sessions and measure the recovery rate. That design would have produced a
confident wrong answer: pulling those sessions out and re-batching them moves most of
them to *early* positions, so recovery would be high and the conclusion would read
"stochastic dropout, add a retry" when the mechanism was "we moved them". More
fundamentally, retry treats the wrong thing — a session that should yield ten memories
and yields two never triggers a retry, and on these numbers that is most of the loss.

| | |
|---|---|
| **Choice** | Two-stage LLM extraction, batch 15, rules for non-semantic fields |
| **Why** | Rule-based Stage B measured 43% key accuracy and 0 supersessions; batch 15 was the quota constraint at the time |
| **Alternatives** | One combined call (output-budget conflict); pure rules (measured, failed); batch 1 (measured, better, 12x the requests) |
| **Trade-off** | ~3/4 of the per-session yield; one extra LLM call per batch |
| **Evidence** | `results/batch-position-pilot.md`; `docs/DECISIONS.md` D28/D30 |
| **Limitation** | The mechanism behind position decay is undetermined — output-budget exhaustion, enumeration drift, input-position effects and schema-length pressure are all consistent with the data, and this experiment separates none of them |
| **Code** | `ingest/two_stage.py`, `extract_facts.py`, `keying.py`, `structure.py` |

### 3.3 The memory schema

Every field below is from `store/schema.sql`.

| Column | Type | Meaning |
|---|---|---|
| `id` | TEXT PK | Memory id |
| `user_id` | TEXT | Namespace. Required on every read; the enforcement point for isolation |
| `type` | TEXT | `semantic` / `episodic` / `preference` / `procedural` / `profile` |
| `content` | TEXT | The human-readable sentence; also what gets embedded and BM25-indexed |
| `subject` | TEXT | **Who the fact is about** |
| `predicate` | TEXT | Temporal key. Supersede detection matches `(user_id, subject, predicate)` |
| `object` | TEXT | The predicate's value |
| `source_role` | TEXT | **Who said it**: `user` / `assistant` / `system` |
| `scope` | TEXT | Why it is worth keeping: `profile`/`preference`/`plan`/`recommendation`/`commitment`/`shared_context`/`event` |
| `importance` | REAL | [0,1], assigned at write time; the fourth retrieval signal |
| `confidence` | REAL | [0,1], lowered by consolidation |
| `event_time` | TEXT | **Valid-time axis**: when the fact holds in the world |
| `valid_from` / `valid_to` | TEXT | Validity interval; `valid_to IS NULL` means still true |
| `ingested_at` | TEXT | **Transaction-time axis**: when we learned it |
| `replaces_previous` | INTEGER | Set when the user's own wording signals a replacement |
| `update_op` | TEXT | Stage B's verdict: `coexists`/`replaces`/`removes`/`none` |
| `superseded_by` | TEXT FK | Which memory replaced it |
| `status` | TEXT | `active` / `superseded` / `evicted` |
| `strength`, `access_count`, `last_accessed_at`, `strength_updated_at` | | Decay and reinforcement (P5). **Not enabled in the shipped config** (`use_strength=False`) |
| `token_count` | INTEGER | Computed at write time — the packer treats memory selection as a knapsack problem and needs each item's weight cheaply |
| `source_session_id`, `source_turn_index`, `source_char_start`, `source_char_end` | | Provenance anchor |

Indexes:

```sql
idx_mem_user_status  (user_id, status)
idx_mem_sp           (user_id, subject, predicate) WHERE status='active'   -- partial
idx_mem_type         (user_id, type, status)
idx_mem_valid        (user_id, valid_from, valid_to)
```

`idx_mem_sp` is a **partial index**: supersede detection only cares about facts that
still hold, so retired rows are kept out of the index entirely — smaller index, faster
lookup.

#### Why "who said it" and "who it is about" must be two concepts

Three sentences, all spoken by the user:

| Sentence | `source_role` | `subject` |
|---|---|---|
| "I live in Sydney." | user | user |
| "Andy wore a blue shirt." | user | **andy** |
| (assistant) "I recommend the Mayo Clinic video." | **assistant** | assistant |

With a single column the second row has nowhere to go — it is not about the user, and
the speaker is not Andy. That was v1, and the result was that **every memory became a
user-profile entry and third-party facts were lost**.

The pair is also what makes "what did you recommend?" answerable at all: that question
filters on `source_role='assistant'`, which is not a property of the subject.

### 3.4 Bi-temporal representation

Two independent time axes, present since the first migration:

| Axis | Columns | The question it answers |
|---|---|---|
| **Valid time** | `event_time`, `valid_from`, `valid_to` | When was this true **in the world**? |
| **Transaction time** | `ingested_at` | When did **we learn** it? |

**Why both are needed.**

```
2026-03-01  the user says: "I moved to Sydney last month."
```

- `event_time` / `valid_from` = **2026-02** (when the move happened)
- `ingested_at` = **2026-03-01** (when the system learned it)

With one axis, two questions contaminate each other: "Where did I live in February?"
needs valid time; "Did the system know I lived in Sydney before March 1?" needs
transaction time.

Users also state the past retrospectively all the time ("last month…", "last year…"),
so the two axes can even run in opposite order: a fact with an earlier `event_time` may
have a later `ingested_at`. Supersession sorts on `event_time`, **not** `ingested_at` —
otherwise "I moved to Sydney last month" would be retired by a memory recorded earlier
that describes an earlier event.

**Why from day one.** Retrofitting a time axis onto existing memories means rewriting
every historical row, and rebuilding a store costs hours of quota. Carrying the columns
from the start costs almost nothing. This is `DECISIONS.md` D9.

There is no formula here and none is forced.

### 3.5 Fact update and supersession

**What it solves.** "I live in Canberra" and "I moved to Sydney" are both true
statements, but only one holds now. Retrieval has to know which.

**Input.** Every memory under one `(user_id, subject, predicate)` key, superseded ones
included.

**Output.** Updated `status` / `valid_to` / `superseded_by`.

**Mechanism, step by step** (`temporal/resolve.py:195`, `_resolve_key`):

```
1. Load every memory on the key (including superseded). Fewer than 2 → return.

2. Is this key resolvable?
      is_single_valued(predicate)  OR  any memory carries replaces_previous
   Neither → go to the repair branch (below) and return.

3. Keep only memories with an event_time. Fewer than 2 → return.
   (undated ones are counted in skipped_undated, never guessed at)

4. Sort by (event_time, id).

5. Collapse consecutive equal values into runs: when one value is restated
   several times, the earliest member owns the interval and the rest are
   restatements of a value already in force.

6. For each run: close the current run's validity ONLY IF the first member of
   the NEXT run itself carries replaces_previous.
```

**Step 6 is the fix for a real incident.** An earlier implementation retired every
consecutive pair on a key once the key was deemed resolvable. Observed live:

> "averaging $100 per week on groceries" was retired by "spent $75 at Walmart last
> Saturday" — two facts that are both true — because some *third* memory on
> `grocery_spending` had signalled a change, which made the whole chain resolvable.

Stage B's verdict scores 0% false supersede in isolation; the loss was entirely in the
integration. So **the per-fact verdict has to be honoured per fact** and never summarised
into a per-key one.

**The repair branch in step 2 matters just as much.** If a key was once resolved under a
wrong arity call, its facts are sitting `superseded` and invisible to retrieval. So on
discovering the key is not resolvable after all, the code sets them back to active
(`mark_current`) rather than merely abstaining. The reason is in the comment: otherwise
one bad call is permanent in every store already built, and rebuilding a store costs
hours of quota.

**How the two criteria relate.**

```python
SINGLE_VALUED_PREDICATES = frozenset(
    {
        "lives_in",
        "works_as",
        "works_at",
        "uses_framework",
    }
)
```

Four entries. Deliberately conservative: a wrong supersede makes a **still-true fact**
disappear from every later query, while a missed supersede only leaves the store no
worse than a flat list. The default is `coexists` — **asymmetric costs justify an
asymmetric default.**

`DECISIONS.md` D23 records that this list was wrong on two of seven entries, and that
two were removed for chaining unrelated facts together. `update_op` was introduced to
replace the hand-maintained list: ask the model directly what a fact *does* to history,
rather than inferring single-valuedness from the predicate's name.

#### What the current knowledge-update failures expose

`knowledge-update` scored 8/8 = 100% on the development set and **10/15 = 66.7%** on the
held-out set, the largest per-type drop. All five failures recalled the gold session
**and** the gold evidence, so none is a retrieval failure; four are downstream, in how
supplied facts are read and combined.

The store offers a matching symptom. Queried directly on `heldout100.db` for this report:

```sql
SELECT subject, predicate, COUNT(*) c,
       SUM(replaces_previous) rp, SUM(status='superseded') sup
FROM memories WHERE subject<>'' AND predicate<>''
GROUP BY subject, predicate HAVING c>1 AND rp>0 ORDER BY c DESC;
```

| store | keys | memories | |
|---|---:|---:|---|
| keys holding exactly 1 memory | 5,348 | 5,348 | nothing to replace |
| keys holding more than 1 | 1,266 | 7,054 | |
| …of which **resolvable** | 56 | 160 | single-valued predicate, or the key carries a replacement signal |
| …of which **inert** | **1,210** | **6,894** | multiple values on one key that can *never* supersede — **55.6% of the store** |

And the replacement signal itself:

| | heldout100 | train150 |
|---|---:|---:|
| memories carrying `replaces_previous` | 622 | 902 |
| …of which **alone on their key** | **546 (88%)** | **777 (86%)** |
| …of those, whose `(user, subject)` holds **other** predicates | **525 (96%)** | **757 (97%)** |
| memories actually superseded | 42 | 76 |
| signal-to-effect ratio | **14.8 : 1** | **11.9 : 1** |

**The last row of the middle block is the mechanism.** Supersession matches on the exact
`(user_id, subject, predicate)` triple. Extraction invents a fresh predicate string per
fact, so the "this replaces an earlier fact" signal and the fact it should replace land
on **different keys inside the same namespace** — 96–97% of orphaned signals sit beside
sibling predicates on the same subject. The resolver then never fires.

Direct corroboration: within a single namespace and subject, train150 holds **17
singular/plural collisions** — `assistant_recommendation` vs `assistant_recommendations`,
`recipe_recommendation` vs `recipe_recommendations`, `aquarium` vs `aquariums`. One
concept, two keys, invisible to each other.

Where the key does have the right shape the mechanism works: 37 keys on `heldout100` and
71 on `train150` did produce a supersession.

**Stated carefully: this is an evidenced mechanism hypothesis, not a proven cause.** It
explains why supersession fires 42 times against 622 signals, and it matches the shape of
the five failures. It does not yet show that these keys are what made *those* five
questions wrong — that needs an experiment linking individual failures to key shape,
which has not been run (see section 12).

> **A correction.** An earlier revision of this section grouped memories by
> `(subject, predicate)` and reported that 43% of the store sat on two keys. That was
> wrong: every question is its own namespace here, and the resolver keys on
> `(user_id, subject, predicate)`. Grouped correctly, concentration is mild — the largest
> key holds 0.4% — and the real finding is the one above, which is both cleaner and
> stronger. The numbers in this section come from
> `scripts/` -equivalent read-only SQL over `stores/heldout100.db` and `stores/train150.db`.

---

## 4. Storage and indexing

### 4.1 SQLite is not a database server

**The common misunderstanding.** "Database" usually evokes PostgreSQL or MySQL: a
separate process, listening on a port, reached over the network. **SQLite is not that.**

SQLite is **a C library linked into your application process**. There is no server, no
port, no wire protocol. The entire database is **one file on disk**. When Python runs
`conn.execute("SELECT …")`, the SQLite library inside the Python process parses the SQL,
executes it in that same process, and `read()`s / `write()`s that file directly. No
inter-process communication occurs.

In this project the chain is:

```
HTTP client
    │  (network)
    ▼
FastAPI app            ←─ the only server process
    │  (ordinary function call)
    ▼
MemoryService          ←─ composition root; REST / MCP / Inspector share this object
    │  (ordinary function call)
    ▼
SQLiteMemoryStore      ←─ store/sqlite.py
    │  (in-process C library call)
    ▼
sqlite3 library
    │  (filesystem read/write)
    ▼
stores/<name>.db       ←─ an ordinary file
```

**How FastAPI relates to SQLite.** FastAPI is the application backend: HTTP, routing,
validation, serialization. SQLite is its in-process persistence layer. They are not
"application ↔ database service" but "application ↔ a library it calls". This explains a
key constraint: **concurrency here is determined by the process model, not by database
configuration** (see 4.2).

**Where the data physically lives.**

| | Path |
|---|---|
| Main database file | `stores/<store-name>.db`, e.g. `stores/train150.db` (185MB) |
| Vector matrix | `stores/<store-name>-index.npy` |
| Vector id list | `stores/<store-name>-index.ids.json` |
| Ingest checkpoint | `stores/<store-name>-ingest.json` |

**The Docker mapping.** From `docker-compose.yml`:

```yaml
environment:
  LLTM_STORE_DIR: /data/stores
volumes:
  - ./stores:/data/stores        # host directory → container path
  - ./configs:/app/configs:ro    # config mounted read-only
```

`/data/stores/train150.db` inside the container and `./stores/train150.db` on the host
**are the same file**. The compose file's own comment notes that this is done so a
container and a local `lltm` run see the same data during development — and that it is
**exactly what you would not do in production**, because SQLite is single-writer.

**The main tables.**

| Table | Contents |
|---|---|
| `sessions` / `turns` | Raw archive |
| `memories` | Structured memory |
| `memories_fts` / `turns_fts` | FTS5 virtual tables (inverted indexes) |
| `entities` / `memory_entities` | Normalized entities; data for the fifth retrieval signal |
| `evidence` | Which raw memories back a synthesized one |
| `meta` | Store-level metadata (extractor fingerprint, etc.) |

### 4.2 WAL: write-ahead logging

The first line of `schema.sql`:

```sql
PRAGMA journal_mode = WAL;
```

**How the default (rollback journal) works.** At the start of a write transaction SQLite
copies the **original pages about to be modified** into a journal file, then writes
directly into the main `.db`. On a crash the journal restores those pages. Consequence:
during a write the main file is in an inconsistent state, so **readers must be blocked**.

**How WAL works.** The other way round: the main `.db` is **untouched** during the
transaction, and new pages are appended to a `-wal` file. Readers keep reading the main
file plus whatever portion of the WAL is visible to them. **Writers do not block readers
and readers do not block writers.**

The three files:

| File | Role | Lifetime |
|---|---|---|
| `train150.db` | Main database | Permanent |
| `train150.db-wal` | New pages not yet merged back | Cleared at checkpoint |
| `train150.db-shm` | Shared-memory index telling connections which page version is current | Removed when the last connection closes |

A **checkpoint** merges the `-wal` pages into the `.db` and truncates the WAL. It can be
automatic or manual. Seeing `two-stage-p10.db-shm` next to a zero-byte `.db-wal` in
`stores/` is the normal post-checkpoint state.

**Why WAL still does not mean multiple writers.** WAL solves **read/write** concurrency,
not **write/write**. SQLite still permits exactly **one** write transaction at a time; a
second writer gets `SQLITE_BUSY`.

**Hence the cross-process lock.** `locking.py` provides a pid-based advisory lock that
ingestion and evaluation must both take. It is not theoretical caution — it is the
product of a real incident:

> Two shells each started an ingest. Quota accounting recorded 211 requests against
> roughly 320 actually made. Nothing in the store looked wrong — what was corrupted was
> the **accounting**, not the data.

The lock is advisory and pid-based. It stops a second run from another shell, which is
the case that actually occurred. It is not a defence against a hostile process and it is
not a distributed lock.

### 4.3 FTS5: full-text search and inverted indexes

**What an inverted index is.** An ordinary index answers "given a row, what is its
column value". An inverted index goes the other way:

```
forward:  memory_42  → "The user stopped drinking coffee"
inverted: "coffee"   → [memory_42, memory_87, ...]
          "drink"    → [memory_42, memory_11, ...]
```

Searching for "coffee" needs no table scan — the inverted list gives the candidates
directly.

**FTS5 virtual tables.** SQLite's full-text module. "Virtual table" means it looks like a
table and is queried with SQL, but underneath it is an inverted index rather than row
storage. The definition here:

```sql
CREATE VIRTUAL TABLE memories_fts USING fts5(
    content,
    content='memories',          -- external content: do not duplicate the text
    content_rowid='rowid',
    tokenize='porter unicode61'
);
```

Three details:

- **`content='memories'` (external content table)**: FTS5 stores only the inverted
  index and **keeps no second copy of the text**, reaching back to `memories` by
  `rowid`. That saves a full duplicate of the corpus.
- **`tokenize='porter unicode61'`**: `unicode61` tokenizes and case-folds by Unicode
  rules; `porter` is a stemmer, so `drinking` / `drinks` / `drink` collapse to one term.
- **Synchronization is by trigger**: an external-content table does not follow its base
  table automatically, so the schema defines triggers for `INSERT`, `DELETE` and
  `UPDATE OF content` (`memories_ai` / `_ad` / `_au`), with a matching set for
  `turns_fts`.

**BM25 ranking.** FTS5 ships a `bm25()` function. This project calls it directly and
**does not implement BM25 itself**:

```sql
SELECT m.id AS id, bm25(memories_fts) AS score
FROM memories_fts JOIN memories m ON m.rowid = memories_fts.rowid
WHERE memories_fts MATCH ? AND m.user_id = ? AND m.status = 'active'
ORDER BY score LIMIT ?          -- bm25() is negative; more negative is better
```
(`store/sqlite.py:414`)

⚠️ **Implementation detail: FTS5's `bm25()` returns the negation of the standard BM25
score.** "Most relevant" is therefore the **smallest** value, and `ORDER BY score` is
ascending. This sign convention reappears in the downstream normalization (5.3).

**Where each index is used.**

| Index | Used by |
|---|---|
| `memories_fts` | The lexical signal in hybrid retrieval |
| `turns_fts` | Archive-wide raw fallback (`search_turns`) |

### 4.4 Why SQLite + WAL + FTS5 and not something else

Compared against this project's actual scale and deployment shape, not in the abstract.
**The facts**: single machine, single writer, largest store 18,017 memories / 185MB,
concurrency requirement zero (evaluation is serial).

| Option | What it would bring | Why not here |
|---|---|---|
| **SQLite + FTS5** (current) | One file, zero service dependency, BM25 for free, transactions and foreign keys | — |
| **PostgreSQL** | Real multi-writer concurrency, stronger typing and constraints, mature operations | Needs a running service, connection configuration, migration tooling. Concurrency demand here is zero, and the price is that every developer and every CI job has to stand up a database. **If this becomes a multi-tenant production deployment, this is the first thing to replace** |
| **pgvector** | Vector index inside Postgres, SQL and vectors in one transaction | As above, and the vector scale here (18k × 384 float32 ≈ 27MB) makes exact search a millisecond operation that an index structure cannot improve on meaningfully |
| **Elasticsearch** | Stronger full-text search, distribution, rich analyzers | A JVM service, cluster operations, and a second copy of the data. `schema.sql`'s comment puts it directly: FTS5 gives BM25 for free, the alternative was standing up Elasticsearch, and that is not worth a service dependency at this scale |
| **External vector DB** (Pinecone / Weaviate / Qdrant …) | Managed, scalable, rich filtering | A second persistence system, so a memory row and its vector live in different places and need cross-system consistency. Deletion, namespace isolation and `status` filtering here all rely on being in the same transaction |

**The trade-off in one sentence:** the current choice exchanges "cannot support multiple
writers" for "zero operations, one file, copyable whole, checksummable, creatable from
nothing inside CI". For a project where every experiment is frozen, hashed and
reproduced, those last properties are the main benefit. Once it becomes a multi-tenant
service the trade-off inverts.

### 4.5 Vector storage and exact search

**Where embeddings are produced.** Locally, not through an API. Model
`sentence-transformers/all-MiniLM-L6-v2`, dimension **384**
(`configs/fallback.yaml`'s `embedding_dim: 384`; also readable from `encoder.dim`).
Device selection is MPS → CUDA → CPU.

The reason is quota: embedding the corpus is roughly **53M tokens**, and daily request
count is the project's binding constraint. Putting embeddings on the API would make them
compete with extraction for the same pool.

**Where vectors live.** Not in SQLite, but in two sibling files:

```
stores/train150-index.npy         # (n, 384) float32 matrix, np.save
stores/train150-index.ids.json    # length-n list of memory ids, row-aligned
```

**Is L2 normalization applied — confirmed from the code: yes, twice.**

```python
# embed/encoder.py
self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, ...)

# store/vector.py :: add()
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
vectors = vectors / np.maximum(norms, 1e-12)

# store/vector.py :: search()
q = q / max(float(np.linalg.norm(q)), 1e-12)
```

The encoder normalizes, and the index normalizes again in both `add` and `search`. The
redundancy is deliberate; the comment states the reason: *done here rather than at the
encoder so every backend gets the same guarantee* — the index does not trust its callers.

**How exact search works** (`store/vector.py :: search`):

```python
scores = self._vectors @ q  # (n, 384) @ (384,) = (n,)
k = min(limit, len(self._ids))
top = np.argpartition(-scores, k - 1)[:k]  # O(n) selection, no full sort
top = top[np.argsort(-scores[top])]  # sort only those k
```

One matrix-vector product scores the whole store, then `argpartition` does an O(n) top-k
selection and only the selected k are sorted (O(k log k)), avoiding a full sort of n.

**Why not FAISS / HNSW / ANN.** `vector.py`'s module docstring gives two reasons at
different levels:

1. **FAISS's `IndexFlatIP` is a brute-force inner-product scan** — exactly what that `@`
   already does. At this corpus size (<1M vectors) the difference is a constant factor on
   an operation that is not the bottleneck; **the LLM calls are**. The dependency buys
   nothing.
2. **Approximate indexes (HNSW / IVF) are excluded for a different reason**: their recall
   noise is **indistinguishable from a regression in the memory algorithm**. Every
   conclusion in this project rests on ablation comparisons, and a retrieval layer that
   randomly drops candidates would corrupt that table.

| | Exact (current) | Approximate |
|---|---|---|
| Recall | 100%, deterministic | < 100%, parameter-dependent |
| Latency | O(n·d); 18k × 384 is milliseconds | Sub-millisecond |
| Effect on ablations | Clean | Noise confounded with the variable under test |

**Limitation.** The vector files and SQLite are separate artifacts; their consistency is
maintained by the pipeline, not by a transaction. Hence an explicit read-only checker
(`scripts/check_ingest_state.py`) verifies checkpoint, raw archive, SQLite integrity,
memory count, index ids, vector rows and extractor fingerprint together before every
resume. On the current train150 all seven agree (18,017 memories = 18,017 ids = 18,017
vector rows, dimension 384).

---

## 5. Read path — how a question becomes an answer

Tracing one concrete query end to end:

```
user_id = "alice"
query   = "What was the Mayo Clinic YouTube video you recommended?"
```

### 5.1 Query representation

The query text goes down **two** paths, because the two retrieval modes need different
input forms:

| Use | Form |
|---|---|
| Semantic | `encoder.encode_one(query)` → 384-dim float32 vector, L2-normalized |
| Lexical | Raw text → `_fts_match()` → an FTS5 MATCH expression |

Both use the **same encoder and the same model**, so query and memory land in one vector
space — the precondition for "similarity" meaning anything.

**What an embedding is.** A function $f_\theta$ mapping text to a fixed-length real
vector:

$$\mathbf{e} = f_\theta(x) \in \mathbb{R}^{384}$$

$\theta$ is the model's weights (`all-MiniLM-L6-v2`; this project only runs inference,
never trains). Its training objective places semantically similar sentences near each
other, which is why "what does she drink" is close to "stopped drinking coffee" even
without a shared content word — precisely what lexical retrieval cannot do.

### 5.2 Semantic retrieval

**Formula.** Cosine similarity between two vectors:

$$\cos(\mathbf q, \mathbf m) = \frac{\mathbf q^\top \mathbf m}{\lVert \mathbf q\rVert\,\lVert \mathbf m\rVert}$$

| Symbol | Meaning |
|---|---|
| $\mathbf q$ | Query vector, 384-dim |
| $\mathbf m$ | A memory's `content` vector |
| $\mathbf q^\top \mathbf m$ | Inner product, $\sum_{i=1}^{384} q_i m_i$ |
| $\lVert\cdot\rVert$ | L2 norm, $\sqrt{\sum_i v_i^2}$ |

**The implementation is a normalized inner product.** Section 4.5 confirmed from the code
that every vector is L2-normalized on both write and query, so
$\lVert\mathbf q\rVert = \lVert\mathbf m\rVert = 1$, and therefore

$$\lVert \mathbf q\rVert = \lVert \mathbf m\rVert = 1 \;\Longrightarrow\; \cos(\mathbf q,\mathbf m) = \mathbf q^\top \mathbf m$$

The line `scores = self._vectors @ q` **is** cosine similarity; no division by norms is
needed. Range $[-1, 1]$.

**Rescaling to [0,1]** before fusion:

$$S_{\text{sem}} = \mathrm{clip}_{[0,1]}\!\left(\frac{\cos(\mathbf q,\mathbf m) + 1}{2}\right)$$

(`retrieve/hybrid.py`: `semantic=_clip((raw + 1.0) / 2.0)`)

**How the candidate set is formed.** The semantic path searches the **entire index**
(`limit=len(self.index)`), then filters by namespace, drops `evicted`, and (in temporal
mode) keeps only `active` — and truncates to `candidate_limit` (default 50) **after**
filtering. The order matters: truncating first would let filtered-out rows consume slots.

### 5.3 BM25 / lexical retrieval

**What BM25 is.** A classic bag-of-words relevance function. It understands no semantics,
but it is extremely strong on **exact token matches** — URLs, model numbers, proper
nouns, figures, which is exactly what embeddings blur.

$$\mathrm{BM25}(q,d) = \sum_{t \in q} \mathrm{IDF}(t)\cdot \frac{f(t,d)\,(k_1+1)}{f(t,d) + k_1\left(1 - b + b\,\dfrac{|d|}{\overline{dl}}\right)}$$

| Symbol | Meaning | Intuition |
|---|---|---|
| $t$ | A query term | One token after tokenization |
| $f(t,d)$ | **Term frequency** of $t$ in document $d$ | More occurrences means more relevant, with diminishing returns |
| $\mathrm{IDF}(t)$ | **Inverse document frequency** | Rare terms discriminate; "the" contributes almost nothing |
| $\lvert d\rvert$ | Document length in tokens | — |
| $\overline{dl}$ | Average document length | — |
| $k_1$ | Term-frequency saturation | How much better 10 occurrences are than 2; smaller saturates faster |
| $b$ | Length-normalization strength, $b\in[0,1]$ | $b=1$ penalizes long documents fully, $b=0$ not at all |

A common IDF form:
$\mathrm{IDF}(t) = \ln\!\left(\dfrac{N - n_t + 0.5}{n_t + 0.5} + 1\right)$, with $N$ the
document count and $n_t$ the number containing $t$.

**This project calls SQLite FTS5's `bm25()` and does not implement it.** Two
implementation facts matter:

1. **The sign is inverted.** FTS5's `bm25()` returns the negation of the standard score,
   so more negative is more relevant and the SQL sorts ascending.
2. **The parameters are not visible.** $k_1$ and $b$ are fixed inside FTS5; this project
   does not configure them and they do not appear in the code. **The specific $k_1$ / $b$
   values FTS5 uses are not established from the current material.**

**Normalization.** BM25 values are negative and unbounded while cosine is bounded. Adding
them raw would make the lexical term dominate or vanish **according to an implementation
detail instead of a configured weight** — `hybrid.py`'s docstring says explicitly that
this is not cosmetic. So a min-max is applied over **the candidates actually observed for
this query**:

$$S_{\text{lex}} = \frac{\max_j r_j - r_i}{\max_j r_j - \min_j r_j}$$

where $r_i$ is candidate $i$'s raw FTS5 score. The numerator is "max minus current"
because `lower_is_better=True` — the most negative score gets 1.0.

⚠️ This is a **relative** normalization: it measures how well a candidate ranks *within
this result set*, not absolute relevance. If every candidate is poor, the best of a bad
batch still receives 1.0. There is also a special case: when `min ≈ max` (one candidate,
or all tied) everything returns 1.0, with the comment "one unambiguous hit gets full
credit".

### 5.4 Five signals and their fusion

**The general scoring function:**

$$S(m,q) = w_s S_{\text{sem}} + w_l S_{\text{lex}} + w_r S_{\text{rec}} + w_i S_{\text{imp}} + w_e S_{\text{ent}}$$

Each signal, all clipped to $[0,1]$:

| Signal | Formula | Notes |
|---|---|---|
| **semantic** | $(\cos + 1)/2$ | See 5.2 |
| **bm25** (lexical) | min-max normalized FTS5 BM25 | See 5.3 |
| **recency** | $S_{\text{rec}} = \exp\!\left(-\ln 2 \cdot \dfrac{\Delta_{\text{days}}}{H}\right)$ | Exponential half-life. $H$ = `recency_halflife_days`, default **30 days**. $\Delta_{\text{days}}$ measured from `event_time`, falling back to `ingested_at`. At $\Delta = H$ the value is exactly 0.5 |
| **importance** | `memory.importance` directly | The $[0,1]$ value assigned at write time |
| **entity** | $S_{\text{ent}} = \max\limits_{e \in E(m)} \dfrac{\lvert T(e)\cap T(q)\rvert}{\lvert T(e)\rvert}$ | $E(m)$ is the memory's linked entities, $T(\cdot)$ tokenizes (`[a-z0-9]+`, length > 2). **Max**, not mean: one entity matching fully is enough |

An optional strength factor multiplies the result:

$$S_{\text{final}} = S(m,q)\cdot \big(\text{strength}\big)^{[\,\texttt{use\_strength}\,]}$$

`use_strength` defaults to `False` and is **not enabled in the shipped configuration**.

**Ordering and determinism.** The sort key is `(-score, -semantic_raw, memory.id)` — ties
break on the raw cosine, then on id. This guarantees **the same store and the same query
always produce the same context**; a context that varied between runs would be
indistinguishable from the answerer variance it is meant to reduce.

#### ⚠️ Only one signal is active in the shipped configuration

The defaults in `config.py`:

```python
class RetrievalWeights(BaseModel):
    semantic: float = 1.0
    bm25: float = 0.0  # the lexical signal; the field is named bm25
    recency: float = 0.0
    importance: float = 0.0
    entity: float = 0.0
```

`configs/fallback.yaml`'s `retrieval:` block **sets only `top_k: 20` and overrides no
weights.** So the effective scoring function of the frozen configuration is:

$$S(m,q) = 1.0\cdot S_{\text{sem}} + 0\cdot S_{\text{lex}} + 0\cdot S_{\text{rec}} + 0\cdot S_{\text{imp}} + 0\cdot S_{\text{ent}} = S_{\text{sem}}$$

**"The architecture supports it" and "production runs it" are two different claims and
this report keeps them apart.** Stated precisely:

> The system implements a weighted fusion of five signals, and all five are **computed
> and recorded** on every retrieval (they appear in the `signals` field, visible in the
> Inspector and in the API's `explain` mode). But **the frozen configuration gives a
> non-zero weight only to semantic.** The current ranking is therefore equivalent to
> pure semantic search. The other four are **implemented and unmeasured**, not measured
> and rejected.

One consequence worth naming: BM25 still affects the **candidate set** (lexical hits are
unioned in), just not the **ranking**. So the lexical path is not inert — its role is
recall, not ordering.

### 5.5 Candidate filtering and recorded rejections

The funnel for one retrieval:

```
whole-index vector search
   → drop user_id ≠ namespace          (namespace isolation)
   → drop status = 'evicted'           (deleted)
   → in temporal mode keep only 'active'  (superseded facts excluded)
   → truncate to candidate_limit = 50
   ∪  BM25 top 50
   → score five signals, sort
   → (optional) rerank
   → truncate to top_k = 20
```

**Rejections are returned too.** The API's `explain` mode returns a `rejected` list with
a reason per entry:

| Reason | Meaning |
|---|---|
| `superseded` | This fact no longer holds |
| `below_rank` | Still true, but its score did not reach top_k |

The design reason, from the README: **an absence without an explanation is
indistinguishable from a bug.** The same principle produces `dropped_sessions` in context
assembly and `RetrievalTrace` in retrieval.

**Staged recall tracing.** `RetrievalTrace` records the full candidate set **before**
truncation and reranking. A single end-of-pipeline recall number cannot distinguish "a
stage failed to find the evidence" from "a stage discarded evidence it had". If candidate
recall is 100% and post-rerank recall is 90%, the reranker is deleting correct evidence —
a different problem from failing to find it. This instrumentation is what later makes
per-module oracle ceilings computable.

### 5.6 Context assembly

**v1: flat top-k.** Hand the answerer 20 memories in score order.

**The problem: individual relevance is not a coherent reasoning context.** A memory from
March's bike repair can sit between two from June's sculpture class. A question spanning
events receives a shredded timeline.

A five-question probe (3 arms × 3 repeats × 5 questions = 45 answers) tested:

| Arm | Context | Cells where all three runs agreed |
|---|---|---:|
| `flat20` | Production: top_k 20, score order | 3 / 5 |
| `flatN` | Same count as coherent, still score-ordered and scattered | 3 / 5 |
| **`coherent`** | The gold sessions' memories, in event order | **5 / 5** |

`flatN` is the cheap hypothesis — if trimming top_k produced the effect, the fix costs
nothing. **It is worse**: on one question `flatN` scored 1/3 against `flat20`'s 3/3, at a
third of the context. **The variable is not how many memories are supplied; it is the
shape.** And what `coherent` bought was mainly not accuracy but **consistency**: it never
disagreed with itself.

⚠️ Two limitations keep this from being a product claim: `coherent` is an oracle (it uses
gold session ids), and five questions selected for having failed is not a sample.

**v2: session-coherent context.** Rebuild the same shape from retrieval alone:

```
retrieved memories
  → group by source_session_id
  → aggregate each group into one session score
  → rank sessions by that score
  → take the top max_sessions
  → within each: take ALL that session's active memories, in event order
  → optionally keep only window_radius memories either side of the best hit
  → apply a global cap max_total_memories; if a session would breach it,
    skip the WHOLE session
  → lay the sessions out chronologically
```

**Session aggregation.** Four options, selected by `budget.aggregate`:

$$A_{\max}(s) = \max_{m\in M_s} S(m,q) \qquad A_{\text{sum}}(s) = \sum_{m\in M_s} S(m,q)$$

$$A_{\text{mean}}(s) = \frac{1}{\lvert M_s\rvert}\sum_{m\in M_s} S(m,q) \qquad A_{\text{top3}}(s) = \sum_{j=1}^{3} S_{(j)}$$

| Symbol | Meaning |
|---|---|
| $s$ | One source session |
| $M_s$ | The **retrieved** memories that came from $s$ (not all of that session's memories) |
| $S(m,q)$ | The fused score from 5.4 |
| $S_{(j)}$ | The $j$-th highest score in $M_s$ |

`max` rewards a single strong hit; `sum` rewards long sessions regardless of relevance;
`mean` rewards sessions that are relevant throughout; `sum_top3` is the compromise.

**Two design constraints, both with stated reasons:**

- **The window is contiguous in event order, not score order.** From the comment: the
  point of a window is that *what surrounds a fact explains it*, and a window assembled
  by score is just a smaller flat ranking.
- **Whole sessions or nothing.** When the cap is hit the session is **skipped**, not
  truncated, because half a timeline is exactly the shape this is trying to escape.
  Skipped sessions are recorded in `truncated` and `dropped_sessions`.

**Current status, stated precisely.** The `context:` block in `configs/fallback.yaml` is
`max_sessions=3, window_radius=null, max_total_memories=20, aggregate=sum_top3,
session_order=chronological, include_superseded=false`. **These are starting points, not
findings** — the config's own comment says "These values are the starting point, NOT a
result".

The candidate developed on train150 is `mean` aggregation, `window_radius=1`,
`max_total_memories=30`, reaching 95.2% top-3 session recall and 95.2% assembled recall
over 145/150 questions at a median of 140 context tokens. **It has not been written to
any config file**, because the finalizer requires all 7,180 sessions to be terminal before
`configs/v2.yaml` may be produced.

**v2's goal is currently consistency, not accuracy.** The registered decision rule says
`coherent-auto` is adopted if it does **not degrade** accuracy and does not raise context
by more than 50% — it is **not required to improve accuracy**. The effect the probe
observed was consistency, and setting an accuracy bar the design cannot resolve would
repeat the mistake the batch-size pre-registration made. **Whether v2 improves accuracy
is unknown until dev100 runs.**

### 5.7 The answerer

**Input.** A system prompt + the assembled memory context + the user's question.

**How the memory context is formatted.** Not a bare list — memories are **grouped by
`scope` under headings** (`evaluation/runners/memory.py :: render_grouped`):

```
About the user:
- The user lives in Sydney.

User preferences:
- The user prefers turbinado sugar in baking.

Previously recommended by the assistant:
- The assistant recommended a Mayo Clinic video about desk posture.

Other people and things discussed:
- Andy wore a blue shirt to the meeting.
```

The full mapping (`_SCOPE_HEADINGS`): `profile` → "About the user", `preference` → "User
preferences", `plan` → "Current plans", `event` → "Past events", `recommendation` →
"Previously recommended by the assistant", `commitment` → "The assistant agreed to",
`shared_context` → "Other people and things discussed", unscoped → "Other things known
about the user".

**Why group.** A constraint ("the user is allergic to nuts") and a candidate answer ("the
user likes almond cake") look identical in a bare list. With headings the model can tell
that the first is there to *constrain* the reply and the second is something that could
be *recommended*. Pre-P10 memories carry no scope; when nothing has a scope the code
falls back to a plain list rather than guessing or inventing one.

**The system prompt** (`ANSWER_PROMPT_VERSION = "memory-aware-v2"`,
`runners/base.py:41`) has four parts:

1. You answer using long-term memories from the user's earlier conversations.
2. **Those memories are context about the user, not necessarily a direct answer** — use
   them actively when they can personalize, constrain or improve the reply; ignore
   irrelevant ones.
3. If the question asks for a specific fact and no memory contains it, say you do not
   know rather than guessing. **But do not refuse merely because no memory states the
   answer word for word** — if the memories are enough for a useful, personalized reply,
   give it.
4. Answer directly and concisely.

The second clause and the second half of the third are the product of a real fix, below.

**The sufficiency verdict: what `need_source` is.** The first call returns **a structured
verdict, not prose** (`AnswerVerdict`):

```python
status: Literal["answer", "need_source", "no_evidence"]
answer: str  # the reply, when status == "answer"
reason: str  # what is missing, in one sentence
source_query: str  # for need_source: keywords to search the raw conversation with
```

| status | Meaning | What happens next |
|---|---|---|
| `answer` | The memories suffice | Return; **one LLM call total** |
| `need_source` | A memory is on topic, but the specific detail asked for (a URL, an exact figure, a product name) was not preserved | Raw fallback + a **second** call |
| `no_evidence` | Nothing supplied relates to the question | Raw fallback (archive-wide) |

**Why a structured verdict rather than letting the model write prose.** The comment puts
it directly: returning prose alone forces a choice between always attaching raw evidence
(measured: 3x context for no detectable gain) and never recovering it. A verdict lets the
model say *which case it is in*, so the second call is paid only for `need_source`. This
turns the trade-off from 1.4 into a runtime decision.

#### Evidence retrieved but the answer is still wrong

This is a distinct and important observation, and it is established, not suspected. The
strongest evidence is the dev50 failure staging: of 14 failures, **S4 (retrieval) is
zero**, while S4b (in context, top-ranked, still unused) is 1 and S5 (everything supplied
and correct, answer still wrong) is 2. All five `knowledge-update` failures recalled the
gold session **and** the gold evidence.

At least five distinct causes have to be separated, and they belong to different modules:

| # | Class | What it looks like | Owner |
|---|---|---|---|
| 1 | **Context assembly failure** | Two dates ranked 1 and 2, and the model read them as the same day | Assembly |
| 2 | **Temporal reasoning failure** | "How many weeks ago did I receive the chandelier?" — needs date arithmetic | Reasoning layer |
| 3 | **Multi-session aggregation failure** | "How many museums in February?" — no turn states the count | Query decomposition |
| 4 | **Knowledge-update interpretation failure** | Both the old and the new fact arrive; the model cannot tell which holds now | Update semantics |
| 5 | **Stochastic generation** | Same input, one of three runs differs | Sampling |

**Class 5 has been quantified separately.** Running the same configuration over 100
questions three times: 6/100 flip verdict, accuracy range 3 points. And the noise is not
diffuse — **four of the six question types are bit-identical across all three runs**, and
every disagreeing question is `temporal-reasoning` or `multi-session`, i.e. classes 2 and
3. **Lookup is deterministic in practice; arithmetic and aggregation are where sampling
shows.**

**What has already been done** (all implemented and measured):

- **Answer prompt rewrite**: separating "this was never discussed" from "there is usable
  context here". Before the fix, on the battery question the system **retrieved
  correctly** — the reply names the power bank and the charging pad — and then declined
  with "I do not know what phone you use". The old prompt was optimized for factual
  recall and abstention, and the same disposition refuses on advice-shaped questions.
- **Scope grouping**, above.
- **Judge routed by question type**: preference questions have a rubric as their gold,
  not a reference answer, so grading them as one fails a reply that did exactly what the
  rubric asked.
- **Session-coherent context** (the v2 candidate), aimed at class 1.
- **Repeated-run evaluation**, which turned class 5 from unknown into a number.

**Possible future directions — all of these are future work and none is implemented:**

- Routing by question type to different assembly and answering strategies;
- Handing date arithmetic to a deterministic calculator instead of the model;
- An explicit current/superseded resolver that settles "which one holds now" before the
  prompt is built, rather than hoping the model infers it;
- A structured evidence table (facts as a table rather than prose);
- Escalating unstable question types to a stronger model.

### 5.8 Conditional raw-archive fallback

**What it solves.** Recovering the details extraction dropped.

**Input.** `user_id`, `source_query` (supplied by the answerer itself), and the memories
retrieved for this question.

**Output.** A `RawEvidence`: some raw turns, which level produced them, and a reason.

**Two candidate sources.**

| Level | Source | When it applies |
|---|---|---|
| `source_local` | The turns the retrieved memories are anchored to (via `source_session_id` / `source_turn_index`) | Retrieval found the right memory and only the detail is missing |
| `archive_wide` | BM25 over every turn in the namespace (`turns_fts`) | Extraction missed that conversation entirely |
| `none` | Neither has anything | Still answer "I do not know" |

**Mechanism** (`retrieve/fallback.py :: recover`):

```python
ranked = store.search_turns(user_id, query, limit=pool)  # pool = max(15, max_turns*5) = 15
local = store.turns_for_memories(memories)
found_the_conversation = bool(local) and (not ranked or ranked[0].session_id in local_sessions)
```

The deciding question is: **does the best evidence for this query live in a conversation
the retrieved memories point at?**

- Yes → the memories found the right conversation and merely lost a detail inside it →
  use `source_local`, and **rank those local turns by the archive's ordering** before
  truncating.
- No → extraction missed that conversation altogether and the memories point elsewhere,
  however confident they look → use `archive_wide`.

**Why the decision is per session, not per turn.** From the comment: within one
conversation BM25 routinely prefers the *user's question* to the *assistant's answer*,
because a question repeats the query's own words. Deciding at turn granularity would read
that as "the archive beat the memories" and abandon a memory that had in fact found the
right place.

**A real incident, and a convincing wrong diagnosis.** On the clean store the Mayo
question — the one this report opens with — stopped working in both formal arms after a
rebuild that *improved* retrieval.

- **The apparent cause**: the level branch. The first version took the source turns
  whenever retrieval returned anything and reached the archive only when it came back
  empty, with nothing checking the memories were about the question. That **is** a real
  defect, but it was **not this one**.
- **The actual cause was truncation.** Retrieval **had** found the right conversation;
  the gold turn was among the candidates. But `turns_for_memories` returns turns ordered
  by session id, the code kept `[:max_turns]`, and the gold turn sat tenth of sixteen
  alphabetically — the three kept were all from a conversation about live music. **Better
  recall meant more candidates, and more candidates meant the answer was sliced off.**
- On the smaller mixed store the same question retrieved nothing, fell through to the
  archive, and never met the slice — which is why the defect survived unseen for so long.

Both defects are the same omission — **nothing ranked the candidates against the
question** — so ranking them fixed both. `single-session-assistant` went 5/6 → **6/6**,
and the arm went 66.0% → **72.0%**.

**No level licenses invention.** If the archive has nothing either, the correct answer is
still "I do not know". Abstention is a measured strength (100% here against
`full_context`'s 50%), and a fallback that turns misses into confident guesses would trade
it away.

**Current configuration.** `fallback: enabled=true, max_turns=3, max_chars=2400`. Render
format:

```
[session <id> · turn <index> · <role>]
<raw text, subject to the max_chars budget>
```

**What the fallback actually contributes — three numbers side by side:**

| | dev50 | heldout100 |
|---|---:|---:|
| Answered from structured memory alone | 54.0% | 50.0% |
| Rescued by the raw archive | **+18.0pp** | **+20.0pp** |
| **Final accuracy** | **72.0%** | **70.0%** |
| Fallback fired | 32% | 36% |
| …and was right | 56.2% | 55.6% |

**Structured memory on its own ties `naive_rag` exactly (54.0%). The archive is the half
that puts the product ahead of the baseline.** Quoting 72.0% undivided reads as though
the memory layer answered all of them; this report does not do that.

---

## 6. Model roles

Four roles, four independent model ids, all pinned and **never `-latest`** — a floating
alias would silently make a week-1 run incomparable with a week-8 run.

| Role | Model id | Input | Output | Responsibility |
|---|---|---|---|---|
| **Extractor** (Stage A) | `gemini-3.1-flash-lite` | 15 sessions of raw text | Bare fact strings | Read the conversation, say which facts are worth keeping |
| **Keyer** (Stage B) | same model, second call | Stage A's fact strings | `temporal_key` + `update_op` + `object` | Give each fact a temporal key and say what it does to history |
| **Answerer** | `gemini-3.5-flash-lite` | System prompt + memory context + question | `AnswerVerdict` | The system under test. Held constant across every variant |
| **Judge** | `gemma-4-31b-it` | Question + gold + candidate answer | correct / incorrect | Grading |
| (Embedder) | `all-MiniLM-L6-v2` | Text | 384-dim vector | Runs locally, consumes no API quota |

**Why they must be separate — two independent reasons:**

1. **A judge must not grade its own prose.** If answerer and judge are the same model it
   has a systematic preference for its own phrasing and the evaluation stops being
   independent. This is `DECISIONS.md` D4, and judge-versus-human agreement is
   **measured**, not assumed.
2. **The free tier meters requests per model.** The server's own 429 body reports
   `GenerateRequestsPerDayPerProjectPerModel-FreeTier`. Three roles on three different
   models means three separate pools — **that is what stops a large ingest from starving
   evaluation.**

**Why each role got a different size.** The config's comments give measured reasons: the
full flash models allow only 20 requests/day on the free tier, not enough for a single
50-question run, while the flash-lite models are limited per *minute* rather than per day.
Extraction is the heavy consumer (~6.1M tokens for one dev-subset pass) and needs the
250k TPM models to finish in minutes rather than hours. The judge is the opposite: judge
inputs are ~200 tokens, so TPM is irrelevant, while gemma's 1,500 requests/day is exactly
what judging needs.

**Constraints.** `quota: rpm=10, tpm=250000, rpd=500`. These are **discovered at
runtime**, not hard-coded, because the server's reported limits disagree with the
documentation and vary by model (D12).

---

## 7. Mathematical reference

Formulas live in their component sections; this is an index, not a re-derivation.
**Only mathematics the project actually uses is listed.**

| Name | Formula | Section |
|---|---|---|
| Embedding | $\mathbf e = f_\theta(x)\in\mathbb R^{384}$ | 5.1 |
| Cosine / normalized inner product | $\cos(\mathbf q,\mathbf m)=\dfrac{\mathbf q^\top\mathbf m}{\lVert\mathbf q\rVert\lVert\mathbf m\rVert}\;\xrightarrow{\ \lVert\cdot\rVert=1\ }\;\mathbf q^\top\mathbf m$ | 5.2 |
| Semantic rescaling | $S_{\text{sem}}=(\cos+1)/2$ | 5.2 |
| BM25 | $\sum_{t\in q}\mathrm{IDF}(t)\dfrac{f(t,d)(k_1+1)}{f(t,d)+k_1(1-b+b\lvert d\rvert/\overline{dl})}$ | 5.3 |
| Lexical normalization | $S_{\text{lex}}=\dfrac{\max_j r_j-r_i}{\max_j r_j-\min_j r_j}$ | 5.3 |
| Recency | $S_{\text{rec}}=\exp(-\ln2\cdot\Delta_{\text{days}}/H)$, $H=30$ days | 5.4 |
| Entity overlap | $S_{\text{ent}}=\max_{e}\lvert T(e)\cap T(q)\rvert/\lvert T(e)\rvert$ | 5.4 |
| Weighted fusion | $S=w_sS_{\text{sem}}+w_lS_{\text{lex}}+w_rS_{\text{rec}}+w_iS_{\text{imp}}+w_eS_{\text{ent}}$ | 5.4 |
| Session aggregation | $A_{\text{mean}}(s)=\frac{1}{\lvert M_s\rvert}\sum_{m\in M_s}S(m,q)$ and three others | 5.6 |

### Evaluation statistics

**Accuracy**

$$\text{Accuracy}=\frac{\#\{\text{questions the judge marked correct}\}}{N}$$

**Source-session Recall@k**

$$\text{Recall@}k=\frac1N\sum_{i=1}^{N}\mathbf 1\!\left[\,g_i\in R_i^{(k)}\right]$$

$g_i$ is question $i$'s gold session, $R_i^{(k)}$ the top-$k$ sessions (or the source
sessions of the top-$k$ memories), and $\mathbf 1[\cdot]$ the indicator function. This
project measures it at several stages — candidates → ranked → selected → assembled —
because one end-to-end number cannot say which stage lost it.

**Exact McNemar test**

The core of paired experiments: two systems run **the same questions**, and only the
questions where they **disagree** are examined.

|  | B correct | B wrong |
|---|---:|---:|
| **A correct** | $a$ | $b$ |
| **A wrong** | $c$ | $d$ |

$a$ and $d$ — both right, both wrong — **carry no information about the difference** and
are discarded. Two discordant cells remain: $b$ (A right, B wrong) and $c$ (A wrong, B
right). Under the null hypothesis of no difference, each discordant pair independently
falls into either cell with probability $1/2$, so

$$b \sim \text{Binomial}(b+c,\ 0.5)$$

and the two-sided exact $p$-value is that binomial's two-tailed probability.

**Why a paired experiment looks at disagreement rather than two accuracies.** Two systems
at 70% and 72% could mean "the same 70 questions correct, plus 2 more" (strong evidence)
or "B got 15 of A's correct ones wrong and 17 of its wrong ones right" (almost no
evidence). Only the discordant cells separate those. The example here: the fallback arm
is 9W-1L against the baseline, $b+c=10$, $p=0.022$.

⚠️ **This project has explicitly recorded that this $p$-value does not hold.** It is one
run per arm, and repeats measured 6 of 100 questions flipping under identical
configurations; on 50 questions about 3 flips are expected, and **a single flip** turns
9W-1L into 8W-2L, $p=0.109$. The effect size is large and probably real, but the published
significance exceeds what one run per arm can support.

**Fisher exact test**: used for **unpaired** proportion comparisons, e.g. the same
question type on dev50 versus heldout100. Every per-type comparison here has $p \ge 0.12$
— none is distinguishable.

**Risk ratio**: used in the zero-yield analysis,
$RR = p_{\text{late positions}} / p_{\text{early positions}}$. The observational data give
$RR = 2.86$, a batch-clustered bootstrap 95% CI of $[2.15, 4.35]$, and a within-batch
permutation test at $p = 0.00005$ ($N=20{,}000$). ⚠️ That is **observational** — each
session sat at exactly one position. The causal evidence is the later randomised
experiment.

---

## 8. Evaluation protocol

### 8.1 The benchmark: LongMemEval-S

500 questions, each with a "haystack" of dozens of sessions in which only a few contain
the answer. Six question types:

| Type | What it needs |
|---|---|
| `single-session-user` | Something the user said in one conversation |
| `single-session-assistant` | Something the **assistant** said in one conversation |
| `single-session-preference` | Apply the user's preference to give a personalized reply; the gold is a **rubric** |
| `multi-session` | Aggregate across several sessions |
| `temporal-reasoning` | Date arithmetic |
| `knowledge-update` | A fact changed; use the value that holds **now** |

**Why this and not LoCoMo.** It has real multi-session time spans and an explicit type
split; LoCoMo's conversations are synthetic (D1).

### 8.2 Baselines

| Baseline | Role |
|---|---|
| `full_context` | A reference point, **not a ceiling** (D16). At 56.0% on dev50 it demonstrates that pasting everything in does not itself solve the problem |
| `naive_rag` | The real opponent: ordinary vector retrieval over conversation text |

### 8.3 Dataset roles — the most important methodology decision here

**The failure first.** dev50 lasted two weeks before it was unusable. Not because it was
small, but because:

> In ordinary supervised learning the training set absorbs the fitting and validation is
> touched by a handful of model-selection decisions — a narrow channel. Here **the
> fitting is prompt engineering**, and it was done by reading dev50's failures in full:
> the extraction prompt was rewritten from them, the fallback design chosen from them,
> gates and thresholds tuned on them, and six modules cancelled on them. Fifty questions
> were doing training and validation duty simultaneously, through the widest possible
> channel.

**That is the mechanism by which prompt tuning leaks a validation set**: no gradients
required, only somebody reading the same failures repeatedly.

**The frozen protocol** (2026-08-20; verified that all pairwise overlaps are zero and the
union is exactly 500):

| Set | n | Role | May individual failures be read? |
|---|---:|---|---|
| `dev50` | 50 | v1 development — **burned** | Historical only |
| `heldout100` | 100 | v1 final test — **spent**, 70.0% | Spent, so reading costs nothing more |
| `train150` | 150 | **v2 development** | **Yes, without limit** |
| `dev100` | 100 | **v2 validation** | **No** — aggregate metrics only |
| `test100` | 100 | **v2 final test** | **No** — not before the single run |

The asymmetry is the point: `train150` exists to be consumed (error analysis consumes
data fastest, which is why it is the largest), `dev100` yields only aggregates and
pre-declared slices, and `test100` runs once.

### 8.4 Pre-registration

Before each experiment runs, a document fixes the arms, the metrics, the decision rule
and a **registered prediction**. The three files are `results/prereg-*.md`. The purpose
is to make "the result looks bad, let us read it differently" impossible: the decision
rule is applied **in code** after the run and written to a hash-bound decision file, with
no room for post-hoc reinterpretation.

One concrete consequence: heldout100 scored 70.0% on the single registered shot and 71.3%
as a three-run mean. **The pre-registration requires reporting the single shot, so this
project loses 1.3 points by honouring it.** That is the direction in which a
pre-commitment is worth something.

### 8.5 Repeated runs

Three repeats of `heldout100` gave 70 / 71 / 73, range 3; 69 correct in all three, 25
wrong in all three, **6 disagreeing**. The protocol therefore became:

- For a set of 100+, once the spread has been characterised, one run is an acceptable
  headline;
- **Any paired claim on tens of questions needs k runs per arm and a stated agreement
  rate**, or it reports noise.
- This applies **retroactively** — every paired result in this repository predating
  2026-08-20 is a single run.

### 8.6 What is reported, and why

| Metric | Why it must be reported |
|---|---|
| Accuracy | The outcome |
| Per-type accuracy | But each cell holds 3–27 questions, so **descriptive only** |
| Source-session recall (staged) | Locates which layer lost the answer |
| Median context tokens | The product claim is 75x less context than the transcript; a change that spends that margin is a different product |
| Fallback trigger rate + correct-after-fallback | Separates what the archive actually contributes |
| API requests and tokens | Cost |
| Run-to-run agreement | How much of the above is luck |

---

## 9. Experiments and ablations

**Organized by the design question under test, not by date.** Each has a fixed shape:
hypothesis → control → treatment → metric → result → interpretation → decision.
**"Not adopted" is also a result.**

### 9.1 Extraction: v1 single-stage vs two-stage

| | |
|---|---|
| **Hypothesis** | Extraction quality is the bottleneck of a memory system |
| **Control** | `chronomem` (v1 extraction) |
| **Treatment** | Two-stage extraction + schema split + worked examples in the prompt |
| **Metric** | Paired accuracy (31-question pilot) |
| **Result** | **+29.0pp**, 11W-2L, **p = 0.022** |
| **Interpretation** | The largest single gain in the project. Source-session recall is 93.5% in *both* arms, so the gain cannot come from retrieval |
| **Decision** | Adopt. Extraction becomes the default path |

### 9.2 Evidence strategy: memory-only vs always-on hydration vs conditional fallback

| | |
|---|---|
| **Hypothesis** | Attaching raw evidence to the answerer improves accuracy |
| **Control** | Memory only |
| **Treatment 1** | Always-on hydration (`two_stage_hydrated`) |
| **Treatment 2** | Conditional fallback |
| **Metric** | Paired accuracy + median context tokens |
| **Result** | Hydration **+3.2pp**, 2W-1L, p = 1.000, at **3x context**; conditional fallback **+18.0pp** (54.0% → 72.0%), 9W-1L |
| **Interpretation** | Hydration's gain is indistinguishable from zero while its cost is certain; the fallback spends the same money only on the 36% that need it |
| **Decision** | Hydration demoted from default; conditional fallback becomes the product path |

### 9.3 Retrieval: cross-encoder reranking

| | |
|---|---|
| **Hypothesis** | A cross-encoder improves the ranking |
| **Control** | Plain hybrid retrieval |
| **Treatment** | `cross-encoder/ms-marco-MiniLM-L-6-v2`, 50 in, 20 out |
| **Metric** | Per-question answer agreement + source recall |
| **Result** | At k=10 both arms answered all 31 questions **identically** (zero disagreements); reranking **lowered** source recall at **every** k (93.5% → 90.3% at k=20) |
| **Interpretation** | It changes no answers, and it deletes correct evidence |
| **Decision** | **Not adopted.** Moved behind an optional `rerank` extra, off by default, torch removed from default dependencies |

### 9.4 Retrieval: just cut top_k

| | |
|---|---|
| **Hypothesis** | Too many distractors; fewer memories will help |
| **Control** | `flat20` |
| **Treatment** | `flatN` — same count as coherent, still scattered by score |
| **Result** | On one question `flatN` was **1/3** where `flat20` was **3/3** — **worse**, at a third of the context |
| **Interpretation** | The variable is shape, not quantity |
| **Decision** | Not adopted. Testing the cheap hypothesis first is what made building the expensive one defensible |

### 9.5 Raw fallback: BM25 vs dense retrieval

| | |
|---|---|
| **Hypothesis** | BM25 collapses on paraphrases, so the fallback needs dense retrieval |
| **Initial evidence** | A constructed query set gave BM25 an R@1 of **6.9%** |
| **Audit** | **Manual inspection of those constructed queries** showed the construction had deleted the *question itself* — "How long have I been collecting vintage cameras?" became `"long"` |
| **Re-test** | Hand-written paraphrases retrieve the gold turn at **rank 1**, including phrasings containing no distinctive noun from the source |
| **Decision** | Evidence discarded. Dense retrieval **deferred, not rejected**, with the conditions that would reopen it written down (≥50 validated retrieval-solvable cases with a clear Recall@3 gain) |

### 9.6 Packing: a learned utility predictor

| | |
|---|---|
| **Hypothesis** | Learning "how useful is this memory" beats packing by relevance |
| **Metric** | Held-out RMSE |
| **Result** | **0.310** against a mean-baseline of **0.263** — it lost to predicting the mean |
| **Decision** | Not adopted |

### 9.7 Extraction batch size and position effect

Full tables in 3.2. Summary:

| | |
|---|---|
| **Original hypothesis** | Zero-yield is extractor variance |
| **Counter-evidence** | Observational: positions 0–3 zero-yield at 6.4%, positions 4–14 at 18.3%, $RR=2.86$ |
| **Randomised control** | The same session at the front and back of a batch. 4.5 vs 1.6 memories/session, **p < 0.0001** |
| **Batch size** | 15 → 2.7 memories/session; 5 → 4.8; **1 → 12.7, zero-yield 0.0%**; 400 / 1,000 / 4,800 requests |
| **Interpretation** | Zero-yield is the tail of a continuous attenuation, not a separate failure mode |
| **Decision** | **Not changed.** Changing it means rebuilding every store. Recorded as v2's known ceiling and listed as a v3 question |

### 9.8 Context shape (the v2 candidate, not yet validated)

| | |
|---|---|
| **Hypothesis** | Session-coherent context improves **consistency** (not necessarily accuracy) |
| **Arms** | `flat20` (baseline) / `coherent-auto` (rebuilt from retrieval) / `coherent-oracle` (gold session ids, the ceiling) |
| **Free gate** | Session recall@top-3 on train150 must be ≥80% before any validation quota is spent. Currently **95.2%** over 145/150 questions |
| **Status** | **dev100 has not run. The accuracy effect is unknown.** |
| **Registered decision rule** | Adopt if accuracy does not degrade and context does not rise more than 50%; if auto degrades but oracle does not, the bottleneck is session selection; if both degrade, close the direction |

---

## 10. Failure analysis by pipeline stage

The taxonomy first, then incidents as case studies. Its value is that **it determined
engineering priority**.

### 10.1 The taxonomy

| Stage | Meaning | dev50 failures |
|---|---|---:|
| **S0 Source** | The fact is not in the conversation | 0 |
| **S1 Extraction** | It never became a memory | **10** |
| **S2 Lifecycle** | It became one, then was merged or superseded away | 1 |
| **S3 Eligibility** | Filtered before ranking | 0 |
| **S4 Retrieval** | Eligible, never reached the context | **0** |
| **S4b Composition** | In context and top-ranked, still not used | 1 |
| **S5 Reasoning** | Everything supplied and correct, answer still wrong | 2 |
| **Evaluation infrastructure** | The system was fine; the measurement was not | See 10.4 |

**Ten of fourteen die at extraction and none at retrieval.**

### 10.2 Module ceilings, measured before anything was built on them

| Module | Ceiling | Measured |
|---|---:|---|
| Extraction | 10 | 9 attempted, **4 fixed** |
| **Retrieval** | **0** | **Not run** — no failure is lost here |
| Composition | 1 | 3 of 4 runs |
| Reasoning | 2 | 0 |

The extraction oracle injects the missing fact and runs the real pipeline. Fixing only 4
of 9 is the more useful half: **there is a second defect behind the missing facts**, so
perfect extraction alone does not reach the ceiling.

### 10.3 Case studies: stage S1

- **`single-session-assistant` scored 0/4 across every memory variant.** (3.2) The schema
  had turned `subject` into a speaker flag, leaving third-party facts nowhere to live.
  Fixed by splitting `subject` / `source_role` and adding worked examples.
- **Batch-position attenuation.** (3.2 / 9.7) The same session moved to the back of a
  request yields a third as much. **Not fixed; quantified.**
- **Zero-yield.** 14.5% of substantive sessions produced no memory on the clean store
  (304/2,096). The planned remedy — re-running the zero-yield sessions — would have
  produced a **confident wrong answer**, because re-batching them moves them to the front.

### 10.4 Case studies: evaluation infrastructure

These are not system defects but **measurement** defects, and they can drive a wrong
decision just as effectively.

- **A store was written by two extractor generations and nothing could notice.** The
  checkpoint recorded nine progress counters and **no extractor version**, while
  `set_meta("extractor_version", …)` **overwrites** on every run instead of comparing. A
  store ingested to 63% by the pre-P10 extractor was resumed to completion by the P10
  extractor; both runs succeeded and every structural check passed. The two generations'
  `source_role` distributions are 93.3%/6.7% versus 44.8%/54.4%. **The fix was not a
  version-string comparison** — that string is hand-edited and misses the likelier drift
  of a reworded prompt. Resume now compares a fingerprint computed from the inputs
  themselves (both prompt texts, `schema.sql`, model id, sessions per request, dedup
  threshold) and names **which component moved**. The store was rebuilt end to end; the
  old one is kept, unmodified, as the diagnostic record.
- **A gate printed its verdict and exited 0.** That reads correctly to a human and is
  invisible to anything checking a return code. Gates now exit non-zero when closed.
- **A stale artifact does not look like an error.** Reading `gate_open: true` from
  `temporal-gate.json` is not evidence that *this* run's gate passed — if the gate crashed
  before writing, the file on disk is the previous run's verdict. Freshness is now checked
  against the run's own start time.
- **An error message asserted an unverified cause.** `2348/2400 sessions — quota probably
  ran out`: quota had never stopped. It was **explicitly withdrawn** rather than quietly
  corrected, because *an error message that guesses a cause is worse than one that reports
  only what it observed — it directs the next person away from the real problem.* The
  same check also compared the wrong two numbers: the corpus has 2,400 session *entries*
  but 2,348 unique session *IDs*, because one session is evidence for more than one
  question.
- **Two simultaneous ingests overwrote each other's quota accounting.** 211 recorded
  against ~320 made. A cross-process lock was added.
- **A metric was wrong three times — more often than the extractor was.** The fidelity
  metric was wrong twice before the extractor was, and a third time at the worked-examples
  revision.
- **Session ids were not user-scoped**, producing a 64% session-recall figure (the script
  mixed unfinished questions into the denominator and compared scoped internal ids against
  public dataset ids). Corrected it is 95%+; the 64% is explicitly withdrawn.

### 10.5 What this analysis did to engineering priority

**The conclusion is blunt: the ceiling on any retrieval-side improvement is zero
questions, while extraction has 10 and reasoning has 2.**

That is why this project rejected an already-implemented reranker; audited the negative
BM25 evidence by hand instead of adopting dense retrieval; put its effort into the
extraction schema and prompt; and chose **context assembly** (the S4b/S5 side) for v2
rather than continuing to tune retrieval.

---

## 11. Engineering and reproducibility

These mechanisms are collected here rather than interrupting the architecture. Together
they answer one question: **how do you know a number was produced by the system it claims
to describe.**

| Mechanism | What it prevents | Implementation |
|---|---|---|
| **Ingestion fingerprint** | Resuming with changed code | Hash of Stage A/B prompt texts, `schema.sql`, model id, sessions per request, dedup threshold; on mismatch it names **which component moved** |
| **Result versioning** | A result row of unknown provenance | Every evaluation row records answerer prompt, judge prompt and extractor version — the extractor version comes from the **store**, not the checkout, because it describes the data being evaluated |
| **Frozen manifests** | "n=50" as an experiment identity | Reported runs must name an explicit question set; `--limit` is exploration only |
| **Content-addressed freezes** | Quietly changing a config afterwards | The freeze hashes code, configuration, data and pre-registered rules together; a post-ingest freeze is accepted only if all of those hashes are unchanged |
| **One-shot ledger** | Running the final test twice | Once status is `complete`, the runner refuses to execute again |
| **Gates exit non-zero** | A gate that prints "closed" and exits 0 | All gates return non-zero when closed |
| **Artifact freshness** | Reading the previous run's verdict | Checked against this run's start time |
| **Cross-process lock** | Two ingests overwriting each other | `locking.py`, pid-based advisory lock |
| **Quota-interruption safety** | An in-flight batch counted as "zero-yield" | Quota/network failures write no terminal state for the in-flight batch and it is excluded from analysis; content-policy refusals keep their raw text and get an explicit terminal state; an automated replay test proves resume produces no duplicates |
| **Read-only state check** | Resuming a store of unknown consistency | `scripts/check_ingest_state.py` verifies checkpoint, raw archive, SQLite integrity, memory count, index ids, vector rows and extractor fingerprint together |
| **Tests and CI** | — | **529 tests**, ubuntu / windows / macos, 80% line coverage (88–100% on critical experiment paths) |
| **Structured logging** | — | One JSON event per request: request id, latency, config fingerprint. **No memory content, no credentials** |

**Three external interfaces share one service object**, so they cannot drift from the
behaviour that was measured:

| Interface | Shape |
|---|---|
| REST | FastAPI, 9 endpoint groups; every read requires `user_id`; **cross-namespace reads return 404, not 403** — 403 would confirm the id exists |
| MCP | `search_memory` / `remember` / `search_conversations` / `get_timeline` / `forget`; every tool takes an explicit `user_id`, with no ambient session identity |
| Memory Inspector | Web page showing why each memory was selected, passed over or superseded, with shareable URLs; the four demos are **recorded real runs** fingerprinted against store and prompt versions, and say `stale` when either moves |

**Docker**: 2.95GB, mostly PyTorch. The default PyTorch install pulls the CUDA wheel —
24.4GB of GPU runtime for a container that will never see a GPU — so the Dockerfile
installs the CPU wheel and deletes the orphaned CUDA packages. The image bakes in no
datasets, models, credentials or database; the store arrives on a mounted volume.

---

## 12. Current v2 status and next steps

### 12.1 Established by experiment

| Conclusion | Evidence |
|---|---|
| Structured memory is a viable compression | 75x less context, higher accuracy on the development set |
| The gain came mainly from the extraction rewrite | +29.0pp, 11W-2L, p = 0.022; identical source recall in both arms |
| Always-on hydration does not justify its token cost | +3.2pp, 2W-1L, p = 1.000, 3x context |
| Conditional fallback contributes substantially | +18.0pp on dev, +20.0pp held out |
| The headline generalises | Held out 70.0% (three-run mean 71.3), -0.7pp against dev |
| Cross-encoder reranking does not help here | Identical answers at k=10, and lower source recall |
| Batched extraction loses memories by position | Randomised control, p < 0.0001 |
| Single-run noise is about 3 points | Three repeats, 6/100 flip |
| The noise concentrates in two question types | Four of six types are bit-identical across three runs |

### 12.2 The current candidate (not yet validated)

**Session-coherent context.** On train150 (145/150 questions): top-3 session recall
**95.2%**, assembled recall **95.2%**, median context **140 tokens**, 1 question
truncated — well above the registered 80% gate.

⚠️ **These are proxy metrics. Finding the right source session is not the same as
answering the question. Whether v2 improves accuracy is unknown until dev100 runs.** The
candidate configuration (`mean` / `radius=1` / `cap=30`) **has not been written to any
config file**; the finalizer requires all 7,180 sessions to be terminal before
`configs/v2.yaml` may exist. Currently **6,983 / 7,180**.

### 12.3 Open questions — supported by data, without conclusions

| Question | Known | Not known |
|---|---|---|
| **Batch-size attenuation** | Batch 1 yields 4.7x batch 15; causally established | The mechanism — output-budget exhaustion, enumeration drift, input-position effects, schema-length pressure; this experiment separates none of them |
| **Wrong answers after correct retrieval** | S4=0, S4b=1, S5=2; all five knowledge-update failures recalled the gold | How the five causes in 5.7 divide up |
| **The predicate namespace** | 55.6% of the store sits on multi-valued keys that can never supersede; 88% of replacement signals are alone on their key, and 96% of those have sibling predicates on the same subject | An evidenced mechanism, but **not yet shown to cause** the observed failures |
| **Exact-detail fidelity** | The recurring failure shape is "keeps the gist, drops the identifier" (the Mayo URL, `Garmin Forerunner`, `2-3 eggs`) | Whether identity-bearing spans should survive extraction verbatim |
| ~~Retrieval weights~~ **closed 2026-08-25** | Measured offline: every added signal degrades ranking (-5.3pp to -17.3pp); `recency` is a no-op because its 30-day half-life meets a corpus 932-1,727 days old | Whether a *tuned* hybrid could help. No weight grid or interaction search was run |
| **Fallback depth** | `max_turns=3` has never been swept; one known case has its answer at BM25 rank 6 | How deep is right |
| **No baselines on the held-out set** | `full_context` / `naive_rag` were never run on heldout100 | Whether "structured memory ties naive RAG" holds on unseen data |

### 12.4 Future hypotheses — none of these is implemented or validated

Marked explicitly so they cannot be read as current capability:

- **A deterministic temporal calculator**: take date arithmetic away from the model.
  Aimed at part of S5.
- **Routing by question type** to different assembly and answering strategies.
- **An explicit current/superseded resolver** that settles which fact holds before the
  prompt is built, instead of expecting the model to infer it from a pile of facts.
- **A structured evidence table**: facts as a table rather than prose.
- **Escalation to a stronger answerer** for unstable question types.
- **A graph representation** making relations between memories explicit.

### 12.5 Execution order

| Step | Action | Completion evidence | Quota |
|---:|---|---|---|
| 1 | Finish train150's last 197 sessions | 7,180/7,180 in both checkpoint and store | ~70 calls |
| 2 | `finalize_train150.py` | Final zero-yield audit, seven fixed grid artifacts, selection record, `configs/v2.yaml` | **None** |
| 3 | Freeze and ingest dev100 | Pre- and post-ingest hashes, the latter a strict extension of the former | ~1,400 calls |
| 4 | dev100, five arms x three repeats | Scores after all fifteen combinations finish; writes `dev100-decision.json` | ~2,050 calls |
| 5 | Freeze and ingest test100 | Two hashes, also binding dev100's aggregate and decision | ~1,400 calls |
| 6 | Spend test100 **once** | One ledger and one table containing every retained arm | ~550 calls |
| 7 | Product hardening | See `docs/PRODUCTIZATION_V2_PLAN.md` | — |

Step 2 is the watershed. If it records STOP — top-3 or assembled recall below 80%, or
median context above 1.5x flat20 — the registered response is to move the work to session
selection and **spend no dev100 quota at all**.

### 12.6 Productization status: this is still a prototype

| Item | Current state |
|---|---|
| Trusted identity and tenant isolation | `user_id` is taken from the request body, not a trusted token |
| Hard deletion and export | Only a soft-deleted memory status, which is not the same as the data being gone |
| Backup and restore | Backup files exist; no restore drill has been performed |
| Cost and quota monitoring | Free tier, no verified price schedule, so no dollar amounts are recorded |
| Concurrency and idempotency | Single-writer SQLite plus an advisory process lock; multi-process writes are not supported |
| Privacy and security | Log redaction, key management and retention are undefined |
| Operations | No health alerting, staged rollout or rollback |

### 12.7 What is not claimed

- No comparison against any third-party system. None has been run under this protocol.
- No baseline comparison on the held-out set.
- No support for multi-process writes on single-machine SQLite.
- A soft-deleted memory status is not described as the user's data being erased.
- Backups are not called reliable without a restore drill.
- No dollar figures are guessed from a free tier or a stale price list.
- A `user_id` in the request body is not described as authentication.
- "The architecture supports five signals" is not written as "five signals rank in
  production".

---

## Appendix A: configuration reference

From `configs/fallback.yaml`, the current frozen configuration.

```yaml
models:
  extractor:  gemini-3.1-flash-lite      # Stage A + Stage B
  answerer:   gemini-3.5-flash-lite      # the system under test
  judge:      gemma-4-31b-it             # its own pool; never grades its own prose
  embedder:   sentence-transformers/all-MiniLM-L6-v2
  embedding_dim: 384

quota:        rpm: 10   tpm: 250000   rpd: 500      # discovered at runtime

retrieval:
  top_k: 20
  # weights not overridden → config.py defaults: semantic 1.0, other four 0.0
  # candidate_limit defaults to 50, recency_halflife_days to 30

ingest:
  sessions_per_request: 15    # known-lossy, see 3.2
  two_stage: true
  checkpoint_every: 5
  dedupe_similarity_threshold: 0.92

temporal_resolution: true

fallback:
  enabled: true
  max_turns: 3
  max_chars: 2400

context:                      # read only by the two_stage_coherent variant
  max_sessions: 3             # ⚠️ starting point, not a result
  window_radius: null
  max_total_memories: 20
  aggregate: sum_top3
  session_order: chronological
  include_superseded: false
```

## Appendix B: code index

| Component | File |
|---|---|
| Schema, indexes, FTS5 triggers | `src/llm_long_term_memory/store/schema.sql` |
| SQLite store, BM25 queries, archive search | `store/sqlite.py` |
| Exact vector index | `store/vector.py` |
| Local encoder | `embed/encoder.py` |
| Two-stage extraction | `ingest/two_stage.py`, `extract_facts.py`, `keying.py`, `structure.py` |
| Provenance anchors | `ingest/provenance.py` |
| Ingestion pipeline and checkpointing | `ingest/pipeline.py` |
| Extractor fingerprint | `ingest/fingerprint.py` |
| Zero-yield audit | `ingest/zero_yield.py`, `zero_yield_report.py` |
| Temporal resolution and supersession | `temporal/resolve.py` |
| Hybrid retrieval and the five signals | `retrieve/hybrid.py` |
| Session-coherent context | `retrieve/coherent.py` |
| Conditional raw fallback | `retrieve/fallback.py` |
| Cross-encoder reranker (off by default) | `retrieve/rerank.py` |
| Answerer prompt and verdict | `evaluation/runners/base.py`, `runners/memory.py` |
| Baselines | `evaluation/runners/full_context.py`, `naive_rag.py` |
| Freezes and lineage validation | `evaluation/reproducibility.py`, `scripts/freeze_v2.py` |
| Cross-process lock | `locking.py` |
| Config definitions and default weights | `config.py` |

## Appendix C: experiment record index

| Content | File |
|---|---|
| Numbered design decisions | `docs/DECISIONS.md` |
| Earlier chronological engineering report | `docs/ENGINEERING_REPORT.md` |
| Dataset role protocol | `results/data-protocol.md` |
| Randomised batch-size / position control | `results/batch-position-pilot.md` |
| Three repeats measuring run noise | `results/heldout-variance.md` |
| Failure staging and module ceilings | `results/failure-stages.md` |
| Five-question context-shape probe | `results/context-arms.md` |
| Rerank Pareto | `results/rerank-pareto.md` |
| Raw-retrieval diagnostic | `results/raw-recall-diagnostic.md` |
| The three pre-registrations | `results/prereg-batch-size.md`, `prereg-context-shape.md`, `prereg-v2-final.md` |
| v2 runbook | `results/v2-runbook.md` |
| v2 progress log | `results/v2-progress.md` |
| Productization acceptance criteria | `docs/PRODUCTIZATION_V2_PLAN.md` |
