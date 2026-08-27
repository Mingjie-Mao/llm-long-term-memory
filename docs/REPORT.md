# LLTM System Design Report

A complete technical description of an LLM long-term memory layer. Written for someone
who knows Python, LLMs and the basics of RAG but has never seen this project: after
reading it you should be able to say what happens to a conversation as it moves through
the system, why each component is built the way it is, and which experiment supports
each decision.

**Every implementation detail here comes from the current code and the frozen
configuration.** Where the code cannot establish something, this report says *not
established from the current material*. Two distinctions run through the whole
document: *what the architecture supports* is not *what production runs*, and
*correlation* is not *causation*.

The body (chapters 0–8) follows one line: **what is used → why → what the result was →
what is still wrong**. Detail that would interrupt that line lives in the appendices:
the full schema (A), mathematical definitions (B), configuration (C), the engineering
incident log (D), a code map (E), and an experiment index (F).

Companion documents: [`ENGINEERING_REPORT.md`](ENGINEERING_REPORT.md) is the earlier
chronological write-up, [`DECISIONS.md`](DECISIONS.md) holds the numbered design
decisions, and [`REPORT.zh-CN.md`](REPORT.zh-CN.md) is this document in Chinese.

---

## Tech stack

| Layer | What | Note |
|---|---|---|
| Language | Python `>=3.11` | Uses `StrEnum` and `Literal` type aliases |
| Packaging / build | `uv` + `hatchling` | `uv.lock` pins every transitive dependency |
| Validation | `pydantic>=2.7` / `pydantic-settings>=2.3` | Extraction output, config and API models all go through schemas |
| CLI | `typer>=0.12` + `rich>=13.7` | `lltm` and `llm-long-term-memory` point at one app |
| Numerics | `numpy>=2.0` | The vector index *is* a numpy matrix |
| HTTP | `httpx>=0.27` | Talks to the LLM API |
| Logging | `structlog>=24.1` | One JSON event per request |
| Config | `pyyaml>=6.0` | `configs/*.yaml` |
| Timezone | `tzdata` (Windows only) | Windows ships no system tz database, so `zoneinfo` cannot resolve `America/Los_Angeles` — and the provider resets quota at Pacific midnight |

Optional extras are deliberately split: `embed` (sentence-transformers, pulls ~2GB of
torch), `llm` (google-genai), `api` (fastapi + uvicorn), `rerank` (measured and not
enabled; its own extra so the dependency is deliberate), `mcp`. Dev group: pytest,
pytest-cov, ruff.

| Storage and retrieval | |
|---|---|
| Relational store | SQLite (`journal_mode=WAL`, `foreign_keys=ON`), one file |
| Full-text search | SQLite FTS5, `porter unicode61` tokenizer, external-content tables kept in sync by triggers |
| Lexical ranking | FTS5's built-in `bm25()` — returns negative values, more negative is better |
| Vectors | `.npy` float32 matrix + `.ids.json` beside the `.db`; exact numpy inner product |

| Models — all pinned, never `-latest` | |
|---|---|
| Extractor (Stage A + B) | `gemini-3.1-flash-lite` |
| Answerer (system under test) | `gemini-3.5-flash-lite` |
| Judge | `gemma-4-31b-it` |
| Embedder (local) | `all-MiniLM-L6-v2`, 384-dim |

Engineering: **532 tests**, 80% line coverage, ruff (line-length 100), a
ubuntu/windows/macos CI matrix, `filterwarnings = ["error::EncodingWarning"]`, and a
2.95GB Docker image (CPU torch).

**What this stack does not contain**: a separate database server, a vector database,
Elasticsearch, a message queue, an ORM. Reasons in §4.5.

---

## 0. Executive summary

**The problem.** An LLM application has to remember what a user said months ago. Putting
the whole transcript in the context window is expensive and grows linearly; retrieving
raw conversation chunks with a vector store cannot express that *facts change* — after
someone moves house, the old address and the new one come back together, with nothing to
say which one still holds.

**The approach.** Compress conversations into structured facts — compact, updatable,
explainable — **and keep the original conversation forever**. Compression is lossy. This
project does not try to eliminate the loss; it requires only that the loss be
**recoverable**: when structured memory cannot answer, the system goes back to the
original turns.

**The architecture.** Two paths:

- **Write**: conversation → raw archive → two-stage extraction → structured memory →
  temporal resolution → embedding → index
- **Read**: query → hybrid retrieval → candidate filtering → context assembly →
  answerer → sufficiency verdict → (if insufficient) raw fallback → answer with
  provenance

**The headline result.** On 100 questions that informed no decision, run once:
**70.0%** accuracy, of which 50.0% is answered from structured memory alone and 20.0% is
recovered by the raw archive, at a median of 1,468 context tokens. On the development set
the same system scores 72.0%, against 56.0% for the full transcript and 54.0% for
ordinary RAG — at 1/75 and 1/9 of their context.

**The three largest limitations:**

1. **No baselines on the held-out set.** That run covered this system only, so
   "structured memory ties naive RAG" remains a development-set claim.
2. **The extractor is known-lossy and was not changed.** Batch size 15 yields 2.6
   memories per session against 12.7 at batch size 1, established by a randomised
   control. Every number sits under that lowered ceiling.
3. **Four of five retrieval signals carry zero weight in production** — and enabling
   them measured worse.

---

## 1. Problem definition and design goals

### 1.1 Option A: full context

Paste the user's entire history into the prompt on every question. Nothing is lost, the
implementation is string concatenation, and there is no retrieval error.

It costs on three axes. **Cost grows linearly with history** — a measured median of
**109,260 tokens** per question here. **Putting it in is not using it** — it scores
**56.0%** on the development set, beaten by an approach with 75x less context;
information in a long context gets diluted, which is measured here rather than cited.
**There is no notion of fact state** — "I live in Canberra" from three years ago and "I
moved to Sydney" from last year are both in the prompt, and the model re-derives which
holds on every question.

### 1.2 Option B: naive RAG over conversation history

```
conversation → split into session / turn chunks → embed each → vector store
query → embed query → take the k nearest chunks → paste into prompt → answer
```

Context drops to a median of **13,057 tokens** (about an eighth of full context), the
implementation is simple, and what comes back is the *original wording* — exact URLs,
model numbers and figures survive. It scores **54.0%**.

Three things it does not naturally solve. **It retrieves chunks, not facts**: a session
may hold twenty things of which one is relevant, and the whole chunk is pulled in.
**It cannot express fact evolution**: vector similarity has no way to know that "I moved
to Sydney" should retire "I live in Canberra", and the stale one may rank higher because
its wording is closer to the question. **It cannot aggregate across sessions**: "how many
museums did I visit in February?" — no single chunk states that number.

### 1.3 Option C: structured long-term memory

Use an LLM to read the conversation into individual typed, time-bounded facts:

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

**Compact**: a fact is tens of tokens, and the median context drops to **1,455**.
**Updatable**: a fact has a `(subject, predicate)` key, so a new value can retire the old
one instead of coexisting with it. **Explainable**: every fact traces back to the turn it
came from, and facts passed over at retrieval time carry a reason.

### 1.4 The fundamental trade-off: compression is lossy

Extraction drops things, and what it drops is often exactly the identifier the question
is about.

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

**The answer is not "make extraction better". It is "accept that it is lossy, and
guarantee the loss is recoverable":**

> **Lossy structured memory + a recoverable raw conversation archive**

The original turns live permanently in the same database with a full-text index. When
structured memory cannot answer, the system searches them.

#### Two uses of raw text that must not be conflated

| | **Always-on hydration** | **Conditional fallback** |
|---|---|---|
| When raw text is fetched | On **every** answer | **Only** when the answerer declares memory insufficient |
| Context cost | About **3x** | Near zero on average — measured trigger rate 36% |
| Measured effect | **+3.2pp**, 2W-1L, p = 1.000 | **+18.0pp** on the development set (54.0% → 72.0%) |
| Current status | Implemented, **demoted, off by default** | **The shipped product path** |

Full comparison in §6.3.

> ⚠️ One sentence here is easy to read backwards. This project also concludes that "the
> gain came from the extraction rewrite (+29.0pp), not from hydration (+3.2pp)".
> **That is about always-on hydration, not about the raw archive.** Stated precisely:
>
> *Two-stage extraction produced the largest upstream improvement, while always-on
> raw-evidence hydration did not justify its token cost. Conditional fallback serves a
> different role: it recovers details lost by compression, only when structured memory is
> insufficient.*
>
> On the development set the archive independently contributes 18.0 percentage points.
> It is not decoration.

### 1.5 Design goals

| Goal | Testable form |
|---|---|
| Compact | Context at least an order of magnitude below the full transcript |
| Updatable | A new value on the same `(subject, predicate)` removes the old one from retrieval, without deleting it |
| Recoverable | A detail extraction dropped can still be answered through the raw path |
| Explainable | Every memory traces to its source turn; every **omission** has a stated reason |
| Isolated | Cross-user reads and writes are impossible, even knowing the id |

---

## 2. System architecture

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

**Write path** (§3): raw archive → two-stage extraction → temporal resolution → storage
and index. It runs offline in batches, two LLM calls per 15 sessions.

**Read path** (§4 retrieval, §5 answering): retrieval → context assembly → answer →
fallback when needed. A question costs **one** answerer call by default; a second is paid
only when the answerer itself declares memory insufficient.

---

## 3. Memory ingestion and update

### 3.1 The raw archive

**What it solves.** Extraction is lossy (§1.4). If the original is not kept, the loss is
permanent and the system cannot distinguish "not in memory" from "never happened".

**Input** is one session (an ordered list of `(role, content, timestamp)`); **output** is
rows in two tables plus an automatically maintained full-text index:

```sql
sessions(id PK, user_id, started_at, source)
turns(id PK, session_id FK→sessions, turn_index, role, content, ts)
INDEX idx_turns_session ON turns(session_id, turn_index)
```

`turns` has **no `user_id` column**. Ownership is reached by joining `session_id` to
`sessions`, so any query that skips the join cannot see user data at all — **isolation is
enforced by the schema rather than by every query site remembering a predicate**. Archive
full-text search is therefore a three-table join (`store/sqlite.py:378`).

**Provenance.** Every memory carries `source_session_id`, `source_turn_index`,
`source_char_start` and `source_char_end`, so "where did this come from" is answerable
down to a character range. On the clean P10 store **4,843 / 4,843 memories resolve to a
source turn** — the precondition for conditional fallback, whose first level is exactly
"fetch the turns these memories are anchored to".

**Why this is a storage decision, not a retrieval algorithm.** Keeping the raw turns and
*finding* them are independent problems. The first is a schema decision that cannot be
repaired after the fact; the second (source-local lookup, BM25, dense, hybrid) can be
swapped at any time, and this project did swap it once (§5.5). Conflating them produces a
common mistake: not keeping the raw text *because we already have vector retrieval*.

**Limitations.** Database size — `train150.db` is 185MB, mostly raw text. Sessions refused
by the content filter keep their raw text but produce no memories, and are counted on
their own line (1 on `heldout100`).

### 3.2 Two-stage extraction

**What it solves.** Turn natural-language conversation into structured, retrievable,
updatable facts. Input is 15 sessions per batch
(`ingest.sessions_per_request: 15`); output is a set of `Memory` rows.

#### A worked example

```
── input turn ────────────────────────────────────────────
session s_4f2a, turn 3, role=assistant, ts=2026-05-02
"Since you mentioned your desk setup, I recommend this Mayo Clinic
 video: How to Sit Properly at a Desk to Avoid Back Pain
 https://www.youtube.com/watch?v=UfOvNlX9Hh0"

── Stage A (bare fact strings, 1 LLM call / batch) ─────────
{ "session_index": 0,
  "fact": "The assistant recommended a Mayo Clinic video about desk posture." }

── Stage B (keying, 1 LLM call / batch) ────────────────────
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

The URL is gone at Stage A. That is the lossy compression from §1.4, and the entire
reason conditional fallback exists.

#### Why two stages

Stage A is only responsible for *which facts are worth keeping*; Stage B only for *giving
a fact a temporal key and saying what it does to earlier facts*. Two calls per batch takes
the cost from roughly 190 requests to roughly 380.

The reason is that the alternative Stage B **was measured**: a regex implementation
(`_rule_keying`, still runnable) scored **43% predicate accuracy** over 148 real memories
and produced **zero supersessions**. Regexes cannot key open-domain relations. And merging
the two stages into one call has a different problem: a single call's output budget would
carry both "read fifteen conversations" and "design a key for every fact", and the
position-decay experiment (§6.4) indicates the output budget is this model's bottleneck.

**Rules are kept where they have the advantage**: `type`, `entities`, `importance` and
date parsing cost no LLM budget, so Stage B spends its whole output budget on the part
regexes could not do.

#### Current limitation

Batch size 15 was a quota decision: 500 requests per model per day, and 2,400 sessions at
batch 15 is 189 batches / 378 extractor calls, which fits in one day.

**A later randomised control showed this configuration loses information**: the same
conversation moved to the back of a request yields a third as much, and batch size 1
yields 4.7x batch 15 with zero-yield falling from 11.7% to 0.0%. Full experiment in
**§6.4**.

The configuration was not changed, because changing it means rebuilding every store (28
more quota days on the free tier) while the experimental conclusion is not yet frozen.
**Every number in this project sits on top of that known-lossy extractor — a lowered
ceiling, not an unknown.**

### 3.3 The memory schema

Key columns (all 20-plus in **Appendix A**):

| Column | Meaning |
|---|---|
| `user_id` | Namespace; the enforcement point for isolation |
| `content` | The human-readable sentence; what gets embedded and BM25-indexed |
| `subject` | **Who the fact is about** |
| `predicate` | Temporal key; supersede detection matches `(user_id, subject, predicate)` |
| `object` | The predicate's value |
| `source_role` | **Who said it**: `user` / `assistant` / `system` |
| `scope` | `profile`/`preference`/`plan`/`recommendation`/`commitment`/`shared_context`/`event` |
| `event_time` / `valid_from` / `valid_to` | Valid-time axis |
| `ingested_at` | Transaction-time axis |
| `replaces_previous` / `update_op` | Replacement signal and Stage B's verdict |
| `status` / `superseded_by` | `active` / `superseded` / `evicted` |
| `source_session_id` / `source_turn_index` / `source_char_start` / `source_char_end` | Provenance anchor |

#### Why "who said it" and "who it is about" must be two columns

Three sentences, all spoken by the user:

| Sentence | `source_role` | `subject` |
|---|---|---|
| "I live in Sydney." | user | user |
| "Andy wore a blue shirt." | user | **andy** |
| (assistant) "I recommend the Mayo Clinic video." | **assistant** | assistant |

With a single column the second row has nowhere to go — it is not about the user, and the
speaker is not Andy. The pair is also what makes "what did you recommend?" answerable:
that question filters on `source_role='assistant'`, which is not a property of the subject.

### 3.4 Bi-temporal representation

Two independent time axes, present since the first migration:

| Axis | Columns | The question it answers |
|---|---|---|
| **Valid time** | `event_time`, `valid_from`, `valid_to` | When was this true **in the world**? |
| **Transaction time** | `ingested_at` | When did **we learn** it? |

```
2026-03-01  the user says: "I moved to Sydney last month."
    event_time / valid_from = 2026-02      (when the move happened)
    ingested_at             = 2026-03-01   (when the system learned it)
```

With one axis, two questions contaminate each other: "Where did I live in February?"
needs valid time; "Did the system know before March 1?" needs transaction time. Users also
state the past retrospectively, so the axes can run in opposite order. **Supersession
sorts on `event_time`, not `ingested_at`** — otherwise "I moved to Sydney last month"
would be retired by a memory recorded earlier that describes an earlier event.

Why from day one (`DECISIONS.md` D9): retrofitting a time axis means rewriting every
historical row, and rebuilding a store costs hours of quota; carrying the columns from the
start costs almost nothing.

### 3.5 Supersession

**What it solves.** "I live in Canberra" and "I moved to Sydney" are both true statements,
but only one holds now, and retrieval has to know which.

**Mechanism** (`temporal/resolve.py:195`):

```
1. Load every memory on the key (including superseded). Fewer than 2 → return.
2. Resolvable? is_single_valued(predicate) OR any memory carries replaces_previous.
   Neither → go to the repair branch and return.
3. Keep only memories with an event_time. Fewer than 2 → return.
   (undated ones counted in skipped_undated, never guessed at)
4. Sort by (event_time, id).
5. Collapse consecutive equal values into runs: the earliest member owns the interval.
6. For each run: close it ONLY IF the first member of the NEXT run itself carries
   replaces_previous.
```

**Step 6's per-fact verdict must not be summarised into a per-key one** — an earlier
implementation retired every consecutive pair once a key was deemed resolvable, and
retired two statements that were both true (incident log in **Appendix D**). Stage B's
verdict scores 0% false supersede in isolation; the loss was entirely in the integration.

**Step 2's repair branch** matters as much: if a key was once resolved under a wrong call,
its facts are sitting `superseded` and invisible to retrieval. On discovering the key is
not resolvable after all, the code sets them **back to active** rather than merely
abstaining — otherwise one bad call is permanent in every store already built.

**How the two criteria relate.**

```python
SINGLE_VALUED_PREDICATES = frozenset({"lives_in", "works_as", "works_at", "uses_framework"})
```

Four entries, deliberately conservative: a wrong supersede makes a **still-true fact**
disappear from every later query, while a missed one only leaves the store no worse than a
flat list. **Asymmetric costs justify an asymmetric default**, so the default is
`coexists`. `update_op` was introduced to replace the hand-maintained list — ask the model
what a fact *does* to history rather than inferring it from the predicate's name.

#### Current limitation: replacement signals fire with nothing to replace

Supersession matches the exact `(user_id, subject, predicate)` triple. Most replacement
signals never reach a target:

| | heldout100 | train150 |
|---|---:|---:|
| memories carrying `replaces_previous` | 622 | 902 |
| …**alone on their key** (nothing to replace) | **546 (88%)** | **777 (86%)** |
| …whose `(user, subject)` holds **other** predicates | **525 (96%)** | **757 (97%)** |
| memories actually superseded | 42 | 76 |
| signal-to-effect ratio | **14.8 : 1** | **11.9 : 1** |

**55.6%** (heldout100) / **55.1%** (train150) of the store sits on multi-valued keys that
can never supersede. What causes the 12–15x gap has been narrowed by elimination rather
than established.

**Predicate naming is not the cause.** An earlier revision of this section inferred one:
extraction invents a fresh predicate per fact, so signal and target land on different keys,
and normalizing names would reunite them. Tested offline, canonicalizing every predicate
merges 17 keys in train150 and **reunites 2 of 777 orphaned signals (0.3%)**, and 0 of 546
on heldout100 ([diagnostics](../results/offline-diagnostics-2026-08-25.md)). **The
hypothesis is withdrawn.**

Sampling the orphans shows why. They sit on well-formed, specific predicates whose
siblings are simply the user's other attributes:

```
signal predicate : age
content          : "The user turned 32 years old on July 15th, 2023."
siblings (48)    : art_studio, asylum_status, audiobook_app, books_read, …
```

`age`, `home_city`, `rent_budget` and `job_tenure` are not spelling variants of anything.
No normalization merges them, because they are different attributes. The 17
singular/plural collisions (`assistant_recommendation` vs `assistant_recommendations`,
`aquarium` vs `aquariums`) are real but account for almost none of the gap.

**The corrected reading is that `replaces_previous` is set on facts with no predecessor in
the store.** Two explanations remain, and this evidence does not separate them: the earlier
value was never extracted — in which case this is a second face of extraction loss rather
than an independent defect in the temporal layer — or the extractor sets the flag on
update-shaped wording ("turned", "renewed", "moved") regardless of whether anything earlier
exists. Separating them requires reading the raw archive for an earlier statement of the
same attribute, which needs semantic judgement rather than SQL and has not been run.

The gap still matches the shape of the five `knowledge-update` failures (§6.7), but no
experiment links individual failures to key shape.

---

## 4. Storage and retrieval

### 4.1 Storage layout

SQLite is a **C library linked into the application process**, not a database server:
no service process, no port, no wire protocol. The entire database is one file, and the
library inside the Python process reads and writes it directly. Concurrency here is
therefore determined by the process model, not by database configuration. The chain is
`FastAPI → MemoryService → SQLiteMemoryStore → sqlite3 → stores/<name>.db`, with no
inter-process communication anywhere in it.

| Artifact | Path |
|---|---|
| Main database | `stores/<name>.db` (185MB for train150) |
| Vector matrix | `stores/<name>-index.npy` (float32) |
| Vector id list | `stores/<name>-index.ids.json` (row-aligned) |
| Ingest checkpoint | `stores/<name>-ingest.json` |

Under Docker, `./stores` is bind-mounted to `/data/stores`, so the file is the same inside
and outside. The compose file's own comment notes this is for development — and that it is
**exactly what you would not do in production**, because SQLite is single-writer.

Main tables: `sessions` / `turns` (raw text), `memories`, `memories_fts` / `turns_fts`
(FTS5 inverted indexes), `entities` / `memory_entities` (data for the entity signal),
`evidence` (which raw memories back a synthesized one), `meta` (store-level metadata
including the extractor fingerprint).

**WAL.** `schema.sql` opens with `PRAGMA journal_mode = WAL`. During a write the main
`.db` is untouched and new pages are appended to a `-wal` file, so readers are not
blocked; `-shm` is the shared-memory index telling connections which page version is
current; a checkpoint merges `-wal` back. **WAL solves read/write concurrency, not
write/write** — SQLite still permits one write transaction at a time. Ingestion and
evaluation therefore take the cross-process advisory lock in `locking.py` (its origin is
in Appendix D).

**Indexes.**

```sql
idx_mem_user_status  (user_id, status)
idx_mem_sp           (user_id, subject, predicate) WHERE status='active'   -- partial
idx_mem_type         (user_id, type, status)
idx_mem_valid        (user_id, valid_from, valid_to)
```

`idx_mem_sp` is a **partial index**: supersede detection only cares about facts that still
hold, so retired rows stay out of it.

### 4.2 Lexical retrieval

An FTS5 virtual table is an inverted index (`"coffee" → [memory_42, memory_87, …]`), so a
search needs no table scan. The definition here:

```sql
CREATE VIRTUAL TABLE memories_fts USING fts5(
    content, content='memories', content_rowid='rowid', tokenize='porter unicode61');
```

Three details: `content='memories'` is an **external content table**, so FTS5 stores only
the index and reaches back by `rowid` rather than duplicating the text; `porter unicode61`
tokenizes and case-folds by Unicode rules and stems, collapsing
`drinking`/`drinks`/`drink`; an external-content table does not follow its base table
automatically, so triggers on `INSERT`, `DELETE` and `UPDATE OF content` maintain it.
`turns_fts` has a matching set.

Ranking uses FTS5's built-in `bm25()`; **this project does not implement BM25 itself**:

```sql
SELECT m.id, bm25(memories_fts) AS score
FROM memories_fts JOIN memories m ON m.rowid = memories_fts.rowid
WHERE memories_fts MATCH ? AND m.user_id = ? AND m.status = 'active'
ORDER BY score LIMIT ?          -- bm25() is negative; more negative is better
```

⚠️ **FTS5's `bm25()` returns the negation of the standard score**, so "most relevant" is
the smallest value and the sort is ascending. Formula and variables in **Appendix B**;
$k_1$ and $b$ are fixed inside FTS5 and not configured here, so their **specific values
are not established from the current material**.

`memories_fts` serves the lexical signal in hybrid retrieval; `turns_fts` serves the
archive-wide raw fallback (§5.5).

### 4.3 Semantic retrieval

Embeddings run **locally**, model `all-MiniLM-L6-v2`, dimension **384**. The reason is
quota: embedding the corpus is roughly 53M tokens, and daily request count is the binding
constraint, so an API embedder would compete with extraction for the same pool.

**Normalization happens twice** (confirmed in code): the encoder passes
`normalize_embeddings=True`, and the index normalizes again in both `add()` and
`search()`. The comment states the reason — *done here rather than at the encoder so every
backend gets the same guarantee* — the index does not trust its callers. Because
$\lVert\mathbf q\rVert=\lVert\mathbf m\rVert=1$, the line `scores = self._vectors @ q`
**is** cosine similarity.

**Exact search** (`store/vector.py`):

```python
scores = self._vectors @ q  # (n, 384) @ (384,) → (n,)
k = min(limit, len(self._ids))
top = np.argpartition(-scores, k - 1)[:k]  # O(n) selection, no full sort
top = top[np.argsort(-scores[top])]  # sort only those k
```

One matrix-vector product scores the whole store; `argpartition` then does an O(n) top-k
selection, avoiding a full sort of n.

**The vector files and SQLite are separate artifacts**, kept consistent by the pipeline
rather than a transaction. Hence a read-only checker
(`scripts/check_ingest_state.py`) verifies checkpoint, raw archive, SQLite integrity,
memory count, index ids, vector rows and extractor fingerprint together before every
resume. On train150 all seven agree (18,519 = 18,519 = 18,519, dimension 384).

### 4.4 Retrieval scoring

The fusion function (per-signal definitions and variables in **Appendix B**):

$$S(m,q) = w_s S_{\text{sem}} + w_l S_{\text{lex}} + w_r S_{\text{rec}} + w_i S_{\text{imp}} + w_e S_{\text{ent}}$$

| Signal | Definition in brief |
|---|---|
| semantic | $(\cos+1)/2$, moving $[-1,1]$ into $[0,1]$ |
| bm25 (lexical) | min-max over **the candidates actually observed for this query**, `lower_is_better=True` |
| recency | Exponential half-life, $H$ = `recency_halflife_days`, default 30 days |
| importance | The $[0,1]$ value assigned at write time |
| entity | **Max**, not mean: one entity matching fully is enough |

The BM25 normalization is **relative** — it measures how well a candidate ranks *within
this result set*, so the best of a uniformly poor set still receives 1.0. `hybrid.py`'s
comment says this is not cosmetic: FTS5's values are negative and unbounded, and adding
them raw would let the lexical term dominate or vanish **according to an implementation
detail instead of a configured weight**.

The sort key is `(-score, -semantic_raw, memory.id)`, so the same store and query always
produce the same context; a context that varied between runs would be indistinguishable
from the answerer variance it is meant to reduce.

#### ⚠️ Only one signal is active in production

`config.py` defaults to `semantic 1.0` with `bm25 / recency / importance / entity` all at
`0.0`, and `configs/fallback.yaml`'s `retrieval:` block sets only `top_k: 20` without
overriding any weight. So

$$S(m,q) = 1.0\cdot S_{\text{sem}} + 0 + 0 + 0 + 0 = S_{\text{sem}}$$

**"The architecture supports it" and "production runs it" are different claims.** All five
signals are **computed and recorded** on every retrieval (visible in `signals`, in the
Inspector and in the API's `explain` mode), but only semantic carries weight. The other
four have now been measured in §6.5 — **enabling them is worse in every case**.

One consequence: BM25 still affects the **candidate set** (lexical hits are unioned in),
just not the **ranking**. The lexical path's role is recall, not ordering.

### 4.5 Why this stack and not another

Compared against the actual scale: single machine, single writer, largest store 18,519
memories / 185MB, concurrency requirement zero.

| Option | Why not here |
|---|---|
| **PostgreSQL** | Needs a running service, connection configuration, migration tooling. Concurrency demand is zero, and the price is that every developer and CI job stands up a database. **If this becomes a multi-tenant deployment, this is the first thing to replace** |
| **pgvector** | As above; and 18k × 384 float32 ≈ 27MB makes exact search a millisecond operation an index cannot meaningfully improve |
| **Elasticsearch** | A JVM service, cluster operations, and a second copy of the data. `schema.sql` puts it directly: FTS5 gives BM25 for free, and standing up ES is not worth a service dependency at this scale |
| **External vector DB** | A second persistence system, so a memory row and its vector live apart and need cross-system consistency. Deletion, namespace isolation and `status` filtering all rely on being in one transaction |
| **FAISS / HNSW / IVF** | `IndexFlatIP` is the same brute-force scan as that `@`, and the bottleneck is the LLM calls. Approximate indexes are excluded for a different reason: **their recall noise is indistinguishable from a regression in the memory algorithm**, which would corrupt the ablation table |

**The trade-off in one sentence**: this exchanges "cannot support multiple writers" for
"zero operations, one file, copyable whole, checksummable, creatable from nothing inside
CI". For a project where every experiment is frozen, hashed and reproduced, those last
properties are the main benefit. It inverts the moment this becomes a multi-tenant service.

---

## 5. Answering pipeline

Tracing one query: `user_id="alice"`,
`query="What was the Mayo Clinic YouTube video you recommended?"`

### 5.1 Candidate filtering

The query goes down two paths: `encoder.encode_one(query)` produces a normalized 384-dim
vector, and `_fts_match()` produces an FTS5 MATCH expression. Both use the same encoder and
model, so query and memory land in one vector space — the precondition for "similarity"
meaning anything.

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

Order matters: the semantic path searches the entire index and truncates **after**
filtering — truncating first would let filtered-out rows consume slots.

**Rejections are returned too.** The API's `explain` mode returns a `rejected` list with a
reason per entry: `superseded` (no longer true) or `below_rank` (still true, lost on
score). The design reason is that **an absence without an explanation is indistinguishable
from a bug**. The same principle produces `dropped_sessions` in context assembly and
`RetrievalTrace` in retrieval.

**Staged recall tracing.** `RetrievalTrace` records the candidate set **before** truncation
and reranking. A single end-to-end recall number cannot distinguish "a stage failed to find
the evidence" from "a stage discarded evidence it had": candidate recall at 100% and
post-rerank recall at 90% means the reranker is deleting correct evidence, which is a
different problem from failing to find it. This instrumentation is what makes the
per-module oracle ceilings in §6.7 computable.

### 5.2 Context assembly

**v1 is flat top-k**: 20 memories in score order. The problem is that **individual
relevance is not a coherent reasoning context** — a memory from March's bike repair can
sit between two from June's sculpture class, so a question spanning events receives a
shredded timeline.

**v2 is session-coherent context**, rebuilt from retrieval alone without gold labels:

```
retrieved memories → group by source_session_id → aggregate into a session score
  → rank sessions → take the top max_sessions
  → within each: ALL that session's active memories, in event order
  → optionally keep only window_radius memories either side of the best hit
  → apply a global cap; if a session would breach it, skip the WHOLE session
  → lay the sessions out chronologically
```

Session aggregation supports `max` / `sum` / `mean` / `sum_top3` (formulas in Appendix B).
Two constraints have stated reasons: **the window is contiguous in event order, not score
order** — "what surrounds a fact explains it, and a window assembled by score is just a
smaller flat ranking"; and **whole sessions or nothing** — the cap skips a session rather
than truncating it, because half a timeline is the shape this is trying to escape, and
skipped sessions are recorded in `dropped_sessions`.

**Current configuration.** `configs/v2.yaml` uses `mean` aggregation, `window_radius=1`,
`max_total_memories=30`, `max_sessions=3`, chronological order. It was chosen by a fixed
rule from a seven-arm grid; the selection is in §6.6.

### 5.3 How memories are presented to the answerer

Not a bare list — memories are **grouped by `scope` under headings**
(`runners/memory.py :: render_grouped`):

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

**Why group.** A constraint ("the user is allergic to nuts") and a candidate answer ("the
user likes almond cake") look identical in a bare list. With headings the model can tell
that the first is there to *constrain* the reply and the second is something that could be
*recommended*. Pre-P10 memories carry no scope; when nothing has one the code falls back to
a plain list rather than guessing. Full heading map in Appendix A.

### 5.4 The sufficiency verdict

The system prompt (`memory-aware-v2`) has four parts: the memories are context about the
user and not necessarily a direct answer, so use them actively; if the question asks for a
specific fact and no memory contains it, say you do not know; **but do not refuse merely
because no memory states the answer word for word**; answer directly and concisely. The
second clause and the second half of the third are the product of a real fix (§6.7).

The first call returns **a structured verdict, not prose**:

```python
status: Literal["answer", "need_source", "no_evidence"]
answer: str  # the reply, when status == "answer"
reason: str  # what is missing, in one sentence
source_query: str  # for need_source: keywords to search the raw conversation with
```

| status | Meaning | What happens next |
|---|---|---|
| `answer` | The memories suffice | Return; **one LLM call total** |
| `need_source` | A memory is on topic, but the specific detail asked for was not preserved | Raw fallback + a **second** call |
| `no_evidence` | Nothing supplied relates to the question | Raw fallback (archive-wide) |

**Why a verdict rather than prose.** Returning prose alone forces a choice between always
attaching raw evidence (3x context for no detectable gain) and never recovering it. A
verdict lets the model say *which case it is in*, so the second call is paid only for
`need_source`. This turns the trade-off from §1.4 into a runtime decision.

### 5.5 Conditional raw-archive fallback

One of the most distinctive parts of the system.

**Input**: `user_id`, `source_query` (supplied by the answerer), and the retrieved
memories. **Output**: a `RawEvidence` — some raw turns, which level produced them, and a
reason.

| Level | Source | When it applies |
|---|---|---|
| `source_local` | The turns the retrieved memories are anchored to | Retrieval found the right memory; only the detail is missing |
| `archive_wide` | BM25 over every turn in the namespace | Extraction missed that conversation entirely |
| `none` | Neither has anything | Still answer "I do not know" |

**Mechanism** (`retrieve/fallback.py :: recover`):

```python
ranked = store.search_turns(user_id, query, limit=pool)  # pool = max(15, max_turns*5) = 15
local = store.turns_for_memories(memories)
found_the_conversation = bool(local) and (not ranked or ranked[0].session_id in local_sessions)
```

The deciding question is **whether the best evidence for this query lives in a conversation
the retrieved memories point at**. Yes → the memories found the right conversation and only
lost a detail, so use `source_local` and **rank those local turns by the archive's ordering**
before truncating. No → extraction missed that conversation, so use `archive_wide`.

**Why the decision is per session, not per turn**: within one conversation BM25 routinely
prefers the *user's question* to the *assistant's answer*, because a question repeats the
query's own words. Deciding at turn granularity would read that as "the archive beat the
memories" and abandon a memory that had found the right place.

**No level licenses invention.** If the archive has nothing either, the correct answer is
still "I do not know". Abstention is a measured strength (100% here against
`full_context`'s 50%), and a fallback that turns misses into confident guesses would trade
it away.

Current configuration: `max_turns=3`, `max_chars=2400`, rendered as
`[session <id> · turn <index> · <role>]` plus the raw text.

**What the fallback actually contributes:**

| | dev50 | heldout100 |
|---|---:|---:|
| Answered from structured memory alone | 54.0% | 50.0% |
| Recovered by the raw archive | **+18.0pp** | **+20.0pp** |
| **Final accuracy** | **72.0%** | **70.0%** |
| Fallback fired | 32% | 36% |
| …and was right | 56.2% | 55.6% |

**Structured memory on its own ties `naive_rag` exactly (54.0%). The archive is the half
that puts the product ahead of the baseline.** Quoting 72.0% undivided reads as though the
memory layer answered all of them; this report does not do that.

**Limitation**: `max_turns` has never been swept, and one known case (`omelette`) has its
answer at BM25 rank 6, so depth is a separate open question. This component was rewritten
after an incident recorded in Appendix D.

---

## 6. Evaluation and experiments

Every experiment in the project is collected here, organized by the design question under
test rather than by date. Each has a fixed shape: hypothesis → control → treatment →
metric → result → interpretation → decision. **"Not adopted" is also a result.**

### 6.1 Evaluation protocol

**Benchmark: LongMemEval-S.** 500 questions, each with a haystack of dozens of sessions
in which only a few contain the answer. Six types: `single-session-user`,
`single-session-assistant`, `single-session-preference` (whose gold is a **rubric**, not a
reference answer), `multi-session`, `temporal-reasoning`, `knowledge-update`. Chosen over
LoCoMo because it has real multi-session time spans and an explicit type split, while
LoCoMo's conversations are synthetic (D1).

**Baselines, and why they are all internal.** `full_context` is a reference point, **not
a ceiling** (D16) — at 56.0% on dev50 it shows that pasting everything in does not itself
solve the problem. `naive_rag` is the real opponent. Every arm shares the same store, the
same answerer, the same judge and the same questions, so the only variable is the memory
layer.

Two other comparison targets were considered and rejected.

**A third-party memory system** (running someone else's implementation on this corpus)
would allow a relative claim, and it is deliberately not attempted. `DECISIONS.md` D2 puts
the reason directly: *cross-system memory numbers are only comparable under an identical
judge model and judge prompt, and published results do not share either — the public
dispute over competing LoCoMo claims is precisely this failure. A claim that cannot be
defended under questioning is worse than no claim.* Doing it properly means re-ingesting
this corpus through their pipeline and forcing this project's answerer and judge onto
their output. That is a separate project's worth of work, and the systems make different
assumptions — some carry their own answering model, some keep no raw text — so the
confounders would outnumber the finding.

**Published LongMemEval numbers** are cheaper to cite and weaker still: a different
answerer, a different judge, a different prompt. A comparison against them measures the
model as much as the memory layer. They may be mentioned in a final report as context for
what range this benchmark lives in; they cannot support a claim of being better.

The consequence is worth stating plainly: **this project will never produce a leaderboard
position.** What it produces instead is a paired internal comparison where one variable
moves, which is the stronger experiment for a design question even though it is the weaker
marketing claim.

**Dataset roles — the most important methodology decision here.** dev50 lasted two weeks,
and not because it was small:

> In ordinary supervised learning the training set absorbs the fitting and validation is
> touched by a handful of model-selection decisions — a narrow channel. Here **the fitting
> is prompt engineering**, and it was done by reading dev50's failures in full: the
> extraction prompt was rewritten from them, the fallback design chosen from them, gates
> and thresholds tuned on them, and six modules cancelled on them. Fifty questions were
> doing training and validation duty simultaneously, through the widest possible channel.

**That is the mechanism by which prompt tuning leaks a validation set**: no gradients
required, only somebody reading the same failures repeatedly.

The frozen protocol (2026-08-20; all pairwise overlaps zero, union exactly 500):

| Set | n | Role | May individual failures be read? |
|---|---:|---|---|
| `dev50` | 50 | v1 development — **burned** | Historical only |
| `heldout100` | 100 | v1 final test — **spent**, 70.0% | Spent, so reading costs nothing more |
| `train150` | 150 | **v2 development** | **Yes, without limit** |
| `dev100` | 100 | **v2 validation** | **No** — aggregate metrics only |
| `test100` | 100 | **v2 final test** | **No** — not before the single run |

**Pre-registration.** Before each experiment runs, a document fixes the arms, metrics,
decision rule and a registered prediction (`results/prereg-*.md`). The decision rule is
applied **in code** after the run and written to a hash-bound file, with **no room for
post-hoc reinterpretation**. One concrete consequence: heldout100 scored 70.0% on the
single registered shot and 71.3% as a three-run mean, and the pre-registration requires
reporting the single shot — **so this project loses 1.3 points by honouring it**.

**Repeated runs.** Three repeats of heldout100 gave 70 / 71 / 73, range 3; 69 correct in
all three, 25 wrong in all three, **6 disagreeing**. The protocol therefore became: for a
set of 100+, one run is an acceptable headline once the spread is characterised; **any
paired claim on tens of questions needs k runs per arm and a stated agreement rate**. This
applies **retroactively** — every paired result predating 2026-08-20 is a single run.

**What is reported**: accuracy; per-type accuracy (3–27 questions per cell, so
**descriptive only**); staged source-session recall; median context tokens; fallback
trigger rate and correct-after-fallback rate; API requests and tokens; run-to-run
agreement.

### 6.2 Extraction: v1 single-stage vs two-stage

| | |
|---|---|
| **Hypothesis** | Extraction quality is the bottleneck of a memory system |
| **Control / treatment** | `chronomem` (v1) / two-stage + schema split + worked examples |
| **Metric** | Paired accuracy (31-question pilot) |
| **Result** | **+29.0pp**, 11W-2L, **p = 0.022** |
| **Interpretation** | The largest single gain in the project. Source-session recall is 93.5% in both arms, so the gain cannot come from retrieval |
| **Decision** | Adopt |

v1 scored only **26.0%**, 28 points below naive RAG, for two reasons. **Rules describing
what to extract, which the model did not follow** — switching to worked examples moved
source fidelity from 33.6% to 36.6%, and *the one instruction that had no example was the
one being ignored*. **`subject` had degenerated into a speaker flag** — only 0.6% of v1
memories carried `subject='assistant'` against 6.7% for two-stage — leaving third-party
facts nowhere to live. The observable consequence was `single-session-assistant` scoring
0/4 for every memory variant while both baselines scored 4/4; a gap that uniform cannot be
a ranking problem.

### 6.3 Evidence strategy: memory-only / always-on hydration / conditional fallback

| | |
|---|---|
| **Hypothesis** | Attaching raw evidence to the answerer improves accuracy |
| **Control** | Memory only |
| **Treatment 1 / 2** | Always-on hydration (`two_stage_hydrated`) / conditional fallback |
| **Metric** | Paired accuracy + median context tokens |
| **Result** | Hydration **+3.2pp**, 2W-1L, p = 1.000, at **3x context**; fallback **+18.0pp** (54.0% → 72.0%), 9W-1L |
| **Interpretation** | Hydration's gain is indistinguishable from zero while its cost is certain; the fallback spends the same money only on the 36% that need it |
| **Decision** | Hydration demoted from default; conditional fallback becomes the product path |

The fallback arm differs from the baseline in three `fallback` settings and nothing else,
checked before the run rather than asserted after. Exact paired McNemar `p = 0.022`.

> ⚠️ **That p-value does not hold.** It is one run per arm, on the same 50 questions that
> exposed the defect whose fix produced it (66.0% → 72.0%). At the measured rate of 6 flips
> per 100, **a single flip** on 50 questions turns 9W-1L into 8W-2L, `p = 0.109`. The
> effect size is large and probably real; the published significance exceeds what one run
> per arm can support.

### 6.4 Extraction batch size and position effect

| | |
|---|---|
| **Original hypothesis** | Zero-yield is extractor variance |
| **Counter-evidence (observational)** | Positions 0–3 zero-yield at 6.4%, positions 4–14 at 18.3%, $RR=2.86$, within-batch permutation $p=0.00005$ |
| **Randomised control** | 60 sessions (30 zero-yield in production, 30 normal, matched on turn count ±2 and assistant share ±0.08), batch fixed at 15, two random orderings each paired with its own reverse |

Paired within session (51 sessions seen at both ends):

| | Coverage view | Count view |
|---|---|---|
| yielded in front, zero in back | **8** | memories/session, front: **4.5** |
| zero in front, yielded in back | **0** | memories/session, back: **1.6** |
| exact McNemar | **p = 0.0078** | **p < 0.0001** |

Batch size itself:

| Batch size | Zero-yield | Memories / session | Corpus extraction requests |
|---|---:|---:|---:|
| **15 (production)** | 11.7% | **2.7** | 400 |
| 5 | 8.3% | 4.8 | 1,000 |
| 1 | **0.0%** | **12.7** | 4,800 |

Not one session in sixty yielded more under a larger batch, twice over (p = 1.7e-18).

**Interpretation**: zero-yield is the tail of a continuous attenuation, not a separate
failure mode. **The mechanism is undetermined** — output-budget exhaustion, enumeration
drift, input-position effects and schema-length pressure are all consistent with the data,
and this experiment separates none of them.

**Decision: not changed.** Changing it means rebuilding every store (28 more quota days).
Recorded as v2's known ceiling and listed as a v3 question (§8.3).

**The planned remedy would have been wrong.** The original plan was to re-run the
zero-yield sessions and measure recovery. That design produces a confident wrong answer:
re-batching them moves most of them to *early* positions, so recovery would be high and the
conclusion would read "stochastic dropout, add a retry" when the mechanism was "we moved
them". More fundamentally, retry treats the wrong thing — a session that should yield ten
memories and yields two never triggers one, and on these numbers that is most of the loss.

**Final zero-yield audit over the complete train150** (7,180 sessions; 1,253 zero-yield,
267 below the minimum turn count, 986 analysed in detail):

| Category | Count | What it supports |
|---|---:|---|
| Content-policy refusals | **0** | — |
| **Identical input, different outcome** | **0** | **No direct evidence of model randomness** |
| Source-format problems | 5 | Found by date / role / empty-text checks |
| Annotated evidence misses | 13 | The annotation proves there was something to keep |
| Same text, different date, different outcome | 175 | The date is model input too, so these are near-repeats |
| Short sessions, possibly no durable fact | 198 | A reasonable candidate, not passed off as proven |
| **Still undetermined** | **862** | **Left unknown rather than filled in by guessing** |

### 6.5 Retrieval experiments

**Cross-encoder reranking.** At k=10 the reranked and plain arms answered all 31 questions
**identically** — zero disagreements — while reranking **lowered** source recall at every k
(93.5% → 90.3% at k=20). It changes no answers and deletes correct evidence. **Not
adopted**; moved behind an optional `rerank` extra with torch out of the default
dependencies.

**Just cutting top_k.** `flatN` (same count as coherent, still scattered by score) scored
**1/3** on a question where `flat20` scored **3/3** — worse, at a third of the context. The
variable is shape, not quantity. Testing the cheap hypothesis first is what made building
the expensive one defensible.

**BM25 vs dense for the raw fallback.** A constructed query set gave BM25 an R@1 of
**6.9%**. **Manual inspection of those queries** showed the construction had deleted the
*question itself* — "How long have I been collecting vintage cameras?" became `"long"`.
The evidence was discarded; hand-written paraphrases retrieve the gold turn at **rank 1**,
including phrasings containing no distinctive noun from the source. Dense retrieval is
**deferred, not rejected**, with the conditions that would reopen it written down (≥50
validated retrieval-solvable cases with a clear Recall@3 gain).

**Retrieval weights (2026-08-25, offline on train150, zero quota).** Seven configurations:

| Weights | Top-3 (mean) | Assembled | Median ctx | vs baseline |
|---|---:|---:|---:|---:|
| **semantic 1.0 only (shipped)** | **95.3%** | **95.3%** | 140.5 | — |
| + recency 0.3 | 95.3% | 95.3% | 140.5 | **bit-identical** |
| + importance 0.3 | 90.0% | 90.0% | 134.0 | -5.3 |
| + bm25 0.5 | 89.3% | 89.3% | 138.5 | -6.0 |
| all five | 87.3% | 87.3% | 151.0 | -8.0 |
| + bm25 1.0 | 86.7% | 86.7% | 136.0 | -8.6 |
| + entity 0.5 | 78.0% | 78.0% | 153.0 | **-17.3** |

**Every added signal degrades ranking.** The recency row being bit-identical is not
robustness — the signal is identically zero across this corpus:

```
memory age    min 932 days    median 1,189    max 1,727
S_recency     min 4.7e-18     median 1.2e-12  max 4.5e-10
memories with S_recency > 0.01 : 0 of 18,519
```

The half-life is 30 days and the freshest memory is 932 days old, so
$2^{-932/30}\approx10^{-10}$. **This is a configuration defect, not a signal defect**, and
it means `recency` has never participated in any retrieval this project has run (the weight
was 0 anyway, so no published result is wrong because of it). The other three genuinely
hurt: min-max scaling lifts a weak signal to full range where it reshuffles a ranking that
was already correct, and `entity` is a coarse max over entity-token overlap, which costs 17
points at weight 0.5.

**Decision: keep semantic-only; this question is closed.** It does **not** establish that a
tuned hybrid is impossible — no weight grid, no interaction search, and the endpoint is a
proxy rather than accuracy. It establishes that these four are not free upside sitting on
the table. Details in [`results/retrieval-weights.md`](../results/retrieval-weights.md).

### 6.6 Context construction

**Five-question probe** (3 arms × 3 repeats × 5 questions = 45 answers):

| Arm | Context | Cells where all three runs agreed |
|---|---|---:|
| `flat20` | Production: top_k 20, score order | 3 / 5 |
| `flatN` | Same count as coherent, still scattered | 3 / 5 |
| **`coherent`** | The gold sessions' memories, in event order | **5 / 5** |

What `coherent` bought was mainly not accuracy but **consistency** — it never disagreed
with itself. Two limitations keep this from being a product claim: `coherent` is an oracle
(it uses gold session ids), and five questions selected for having failed is not a sample.

**The seven-arm fixed grid on train150** (complete 150 questions; the grid was fixed before
the remaining questions arrived):

| Configuration | Aggregate | Radius | Cap | Top-3 | Assembled | Median ctx | Truncated |
|---|---|---:|---:|---:|---:|---:|---:|
| max-whole-cap20 | max | — | 20 | 94.7% | 91.3% | 177.0 | 20 |
| mean-whole-cap20 | mean | — | 20 | 95.3% | 93.3% | 153.0 | 12 |
| mean-r1-cap20 | mean | 1 | 20 | 95.3% | 93.3% | 140.0 | 8 |
| **mean-r1-cap30** | **mean** | **1** | **30** | **95.3%** | **95.3%** | **140.5** | **1** |
| mean-r1-cap40 | mean | 1 | 40 | 95.3% | 95.3% | 140.5 | 0 |
| mean-r2-cap30 | mean | 2 | 30 | 95.3% | 95.3% | 151.5 | 1 |
| max-r1-cap30 | max | 1 | 30 | 94.7% | 94.7% | 157.5 | 3 |

All three gates pass: Top-3 95.3% ≥ 80%, assembled 95.3% ≥ 80%, context 140.5 / 354.5 =
**0.40x**, far below the 1.5x limit. **The choice is made entirely by the rule written in
advance**: cap30, cap40 and r2-cap30 tie at 95.3/95.3 → "smaller hard cap" eliminates
cap40 → "smaller window radius" eliminates r2 → **mean-r1-cap30**.

Worth noting: `mean-r1-cap40` truncates **zero** questions rather than one, which is the
row a human would have picked. The rule chose cap30 because it was written before the data
arrived. **That is the pre-registration doing its job on a decision small enough to be
tempting.**

Session recall by aggregation:

| Aggregate | @1 | @2 | @3 | @5 | Assembled |
|---|---:|---:|---:|---:|---:|
| max | 86.7% | 91.3% | 94.7% | 97.3% | 94.7% |
| **mean** | 84.7% | 93.3% | **95.3%** | 96.0% | **95.3%** |
| sum_top3 | 50.0% | 66.0% | 75.3% | 88.7% | **74.7%** |
| sum | 42.7% | 65.3% | 75.3% | 88.7% | 74.7% |

`sum_top3` was the **starting value** in the config file, and its assembled recall of 74.7%
is **below the 80% gate** — shipping the starting point as a result would have failed its
own registered gate. The ceiling is **98.0%** (the gold session appears somewhere in the
top-20 memory input for 147 of 150), of which assembly delivers 95.3% at a median rank of
**1**.

**v2 is not yet validated.** The registered rule adopts `coherent-auto` if it does **not
degrade** accuracy and does not raise context by more than 50% — it is **not required to
improve accuracy**, because the effect being chased is consistency. **Whether v2 improves
accuracy is unknown until dev100 runs.**

### 6.7 Failure analysis

**The 14 dev50 failures, by the layer that loses the answer:**

| Stage | n | Meaning |
|---|---:|---|
| S0 Source | 0 | The fact is not in the conversation |
| **S1 Extraction** | **10** | It never became a memory |
| S2 Lifecycle | 1 | It became one, then was merged or superseded away |
| S3 Eligibility | 0 | Filtered before ranking |
| **S4 Retrieval** | **0** | Eligible, never reached the context |
| S4b Composition | 1 | In context and top-ranked, still not used |
| S5 Reasoning | 2 | Everything supplied and correct, answer still wrong |

**Ten of fourteen die at extraction and none at retrieval.**

**Module oracle ceilings, measured before anything was built on them:**

| Module | Ceiling | Measured |
|---|---:|---|
| Extraction | 10 | 9 attempted, **4 fixed** |
| **Retrieval** | **0** | **Not run** — no failure is lost here |
| Composition | 1 | 3 of 4 runs |
| Reasoning | 2 | 0 |

The extraction oracle fixing only 4 of 9 is the more useful half: **there is a second
defect behind the missing facts**, so perfect extraction alone does not reach the ceiling.

**This analysis set engineering priority.** The ceiling on any retrieval-side improvement
is zero questions, while extraction has 10 and reasoning 2. That is why this project
rejected an already-implemented reranker, audited the negative BM25 evidence by hand,
put its effort into the extraction schema and prompt, and chose **context assembly** for v2
rather than continuing to tune retrieval.

#### Evidence retrieved, answer still wrong

Established, not suspected: all five `knowledge-update` failures recalled the gold session
**and** the gold evidence. At least five causes have to be separated, and they belong to
different modules:

| # | Class | What it looks like | Owner |
|---|---|---|---|
| 1 | Context assembly failure | Two dates ranked 1 and 2, read as the same day | Assembly |
| 2 | Temporal reasoning failure | "How many weeks ago did I receive the chandelier?" | Reasoning layer |
| 3 | Multi-session aggregation failure | "How many museums in February?" — no turn states the count | Query decomposition |
| 4 | Knowledge-update interpretation failure | Both old and new facts arrive; which holds now is not resolved | Update semantics |
| 5 | Stochastic generation | Same input, one of three runs differs | Sampling |

**Class 5 was quantified separately**: 6 of 100 flip, accuracy range 3 points. The noise is
not diffuse — **four of the six question types are bit-identical across three runs**, and
every disagreeing question is `temporal-reasoning` or `multi-session`, i.e. classes 2 and 3.
**Lookup is deterministic in practice; arithmetic and aggregation are where sampling shows.**

**Per-type generalisation** (dev50 → heldout100, Fisher exact):

| Type | dev50 | heldout100 | Fisher p |
|---|---:|---:|---:|
| knowledge-update | 8/8 = 100% | **10/15 = 66.7%** | 0.122 |
| single-session-user | 7/7 = 100% | 12/14 = 85.7% | 0.533 |
| single-session-assistant | 6/6 = 100% | 9/11 = 81.8% | 0.515 |
| multi-session | 7/13 = 53.8% | 19/27 = 70.4% | 0.480 |
| single-session-preference | 2/3 = 66.7% | 4/6 = 66.7% | 1.000 |
| **temporal-reasoning** | 6/13 = 46.2% | **16/27 = 59.3%** | 0.509 |

Every dev50 cell holds 3–13 questions. Its three 100% categories all fell, its two worst
both rose, and none of it is distinguishable from noise — **dev50's per-type numbers never
carried information**, which nobody said out loud while they were being used to choose what
to build.

**Fixes already made**: the answer prompt rewrite (separating "never discussed" from
"usable context here" — before it, the battery question retrieved correctly and then
declined), scope grouping, a judge routed by question type (preference gold is a rubric,
not a reference answer), session-coherent context, and repeated-run evaluation. Future
directions in §8.3.

---

## 7. Engineering and reproducibility

These mechanisms answer one question: **how do you know a number was produced by the
system it claims to describe.**

| Mechanism | What it prevents | Implementation |
|---|---|---|
| **Ingestion fingerprint** | Resuming with changed code | Hash of Stage A/B prompt texts, `schema.sql`, model id, sessions per request, dedup threshold; on mismatch it names **which component moved** |
| **Result versioning** | A result row of unknown provenance | Every row records answerer prompt, judge prompt and extractor version — the extractor version comes from the **store**, not the checkout, because it describes the data being evaluated |
| **Frozen manifests** | "n=50" as an experiment identity | Reported runs must name a frozen question set; `--limit` is exploration only |
| **Content-addressed freezes** | Quietly changing a config afterwards | Code, configuration, data and pre-registered rules are hashed together; a post-ingest freeze is accepted only if all of them are unchanged. Fields whose names end in `_for_reference_only` are recorded and not compared |
| **One-shot ledger** | Running the final test twice | Once status is `complete`, the runner refuses to execute again |
| **Gates exit non-zero** | A gate that prints "closed" and exits 0 | All gates return non-zero when closed |
| **Artifact freshness** | Reading the previous run's verdict | Checked against this run's start time |
| **Cross-process lock** | Two ingests overwriting each other | `locking.py`, pid-based advisory lock |
| **Quota-interruption safety** | An in-flight batch counted as zero-yield | Quota/network failures write no terminal state for the in-flight batch and it is excluded from analysis; content-policy refusals keep their raw text and get an explicit terminal state; an automated replay test proves resume produces no duplicates |
| **Read-only state check** | Resuming a store of unknown consistency | `scripts/check_ingest_state.py` verifies checkpoint, raw archive, SQLite integrity, memory count, index ids, vector rows and extractor fingerprint together |
| **Tests and CI** | — | **532 tests**, ubuntu / windows / macos, 80% line coverage (88–100% on critical experiment paths) |
| **Structured logging** | — | One JSON event per request: request id, latency, config fingerprint. **No memory content, no credentials** |

**None of these was designed in advance.** Each came from a specific failure — a store
written by two extractor generations with nothing able to notice, a gate that printed
"closed" and exited 0, an error message that guessed a cause, two processes overwriting
each other's quota accounting. The full incident log is **Appendix D**.

**Three external interfaces share one service object**, so they cannot drift from the
behaviour that was measured:

| Interface | Shape |
|---|---|
| REST | FastAPI, 9 endpoint groups; every read requires `user_id`; **cross-namespace reads return 404, not 403** — 403 would confirm the id exists |
| MCP | `search_memory` / `remember` / `search_conversations` / `get_timeline` / `forget`; every tool takes an explicit `user_id`, with no ambient session identity |
| Memory Inspector | Web page showing why each memory was selected, passed over or superseded, with shareable URLs; the four demos are **recorded real runs** fingerprinted against store and prompt versions, and say `stale` when either moves |

**Docker**: 2.95GB, mostly PyTorch. The default install pulls the CUDA wheel — 24.4GB of
GPU runtime for a container that will never see a GPU — so the Dockerfile installs the CPU
wheel and deletes the orphaned packages. The image bakes in no datasets, models,
credentials or database; the store arrives on a mounted volume.

---

## 8. Limitations and next steps

### 8.1 Current limitations

**Extraction loss.** Batch size 15 yields 2.6 memories per session against 12.7 at batch
size 1 (§6.4, randomised control). Every number in this project sits under that lowered
ceiling. It is a known quantity, not an unknown.

**Supersession fires 42 times on 622 signals.** 88% of replacement signals sit alone on
their key. Predicate normalization was proposed as the cause and **tested offline: it
reunites 0.3% of them**, so that hypothesis is withdrawn (§3.5). The remaining
explanations are that the earlier value was never extracted — making this a second face of
extraction loss — or that the flag is set on update-shaped wording regardless. Not yet
separated.

**Reasoning after successful retrieval.** S4 retrieval failures are zero, while S4b
composition is 1 and S5 reasoning is 2; all five `knowledge-update` failures recalled the
gold (§6.7). How the five causes divide up is unknown.

**Final baseline evidence is missing.** `full_context` and `naive_rag` were never run on
the held-out set, so "structured memory ties naive RAG" has no unseen evidence. This is the
largest gap in the evidence chain.

**Storage and auth are still a prototype.** `user_id` comes from the request body rather
than a trusted token; deletion is a soft status change rather than an erase; SQLite is
single-writer with an advisory lock, so multi-process writes are not supported; backups
exist but no restore drill has been run; the free tier has no verified price schedule, so
no dollar figures are recorded; log redaction, key management, retention, alerting and
rollback are undefined. Acceptance criteria are in
[`PRODUCTIZATION_V2_PLAN.md`](PRODUCTIZATION_V2_PLAN.md).

**Other recorded limitations**: a prompt change (rewriting `ANSWER_SYSTEM` on 2026-08-14)
broke comparability with every number measured before it — a deliberate trade, recorded
rather than smoothed over; `stores/two-stage-hydrated.db` mixes two extractor generations
(4,843 pre-fix rows and 2,265 after), so results on it stay labelled diagnostic; temporal
arithmetic is unsolved and the fallback does not touch it; `fallback.max_turns` has never
been swept.

### 8.2 What remains of the v2 evaluation

**Done.** All 7,180 train150 sessions terminal, 18,519 memories, the seven-arm fixed grid,
the final zero-yield audit, and `configs/v2.yaml` (`mean` / radius 1 / cap 30) selected by
the fixed rule.

**dev100.** A first ingest reached 37% (1,789 / 4,791 sessions, 408 extractor calls) and
was **discarded** — the data was sound, but the lineage was not: its pre-ingest freeze had
been captured against an uncommitted tree, and committing that work moved `git HEAD`, which
the freeze compared. The calls are written off rather than freezing after the fact, and the
enforcement defect is fixed (§7). Re-ingest is about 1,400 calls, followed by **5 arms x 3
repeats** (about 2,050 answerer calls): `flat20` / `coherent-auto` / `coherent-oracle` are
decision arms, while `naive_rag` / `memory-only` are **reported baselines** registered on
2026-08-25 and unable to move the product choice.

**test100.** Freeze and ingest (about 1,400 calls) plus a **single** run (about 690). When
that finishes the experimental phase is over, and it gives the first answer to "is 70%
actually better than naive RAG, on data nothing was fitted to?"

**Three sentences that must appear in the final report**, fixed now rather than after the
numbers: the result sits on a batch-15 extractor and is a lowered ceiling; single-run noise
is 3 points, so no arm gap below that counts as a result; `test100` is one-shot, with no
repeats and no majority vote, so its p-values are descriptive.

### 8.3 Future work

**None of the following is implemented or validated.** Marked explicitly so it cannot be
read as current capability.

| Direction | Basis | Why not now |
|---|---|---|
| **Batch size 1 / adaptive batching** | 4.7x the yield, causally established (§6.4). A tail-only second pass was measured as an alternative: it costs 1.73x instead of 12x but recovers only **22%** of the loss, because position is the small half and batch crowding is 78% | Rebuilds every store, 10x the quota — and there is no cheap substitute |
| ~~Predicate normalization~~ **closed 2026-08-25** | Simulated offline: a stemming merge reunites 2 of 777 orphaned signals | Withdrawn. The open question moved to whether the missing predecessors were ever extracted (§3.5) |
| **Deterministic temporal calculator** | The S5 class needs date arithmetic | Needs its boundary located on train150 first |
| **Routing by question type** | Noise concentrates in temporal and multi-session | As above |
| **Explicit current/superseded resolver** | Settle which fact holds before the prompt is built | Depends on predicate normalization landing first |
| **Structured evidence table** | Facts as a table rather than prose | Unvalidated |
| **Stronger answerer** | Escalate unstable question types | Belongs after test100, as its own ablation |
| ~~`recency` half-life~~ **closed 2026-08-25** | Retested at 400 / 600 / 1200 days with non-zero weight: Top-3 and assembled recall do not move, @2 drops | Configured correctly it is still useless here — the gold session is not preferentially recent |
| **Production storage** | SQLite is single-writer (§4.5) | Only **after** the experimental conclusion is fixed, or experiment and refactor get mixed |

### 8.4 What is not claimed

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

# Appendices

## Appendix A — Full memory schema

All from `store/schema.sql`.

| Column | Type | Meaning |
|---|---|---|
| `id` | TEXT PK | Memory id |
| `user_id` | TEXT | Namespace; the enforcement point for isolation |
| `type` | TEXT | `semantic` / `episodic` / `preference` / `procedural` / `profile` |
| `content` | TEXT | The human-readable sentence; embedded and BM25-indexed |
| `subject` | TEXT | Who the fact is about |
| `predicate` | TEXT | Temporal key; supersede detection matches `(user_id, subject, predicate)` |
| `object` | TEXT | The predicate's value |
| `source_role` | TEXT | Who said it: `user` / `assistant` / `system` |
| `scope` | TEXT | `profile`/`preference`/`plan`/`recommendation`/`commitment`/`shared_context`/`event` |
| `importance` | REAL | [0,1], assigned at write time; the fourth retrieval signal |
| `confidence` | REAL | [0,1], lowered by consolidation |
| `event_time` | TEXT | Valid-time axis: when the fact holds in the world |
| `valid_from` / `valid_to` | TEXT | Validity interval; `valid_to IS NULL` means still true |
| `ingested_at` | TEXT | Transaction-time axis: when we learned it |
| `replaces_previous` | INTEGER | The user's own wording signalled a replacement |
| `update_op` | TEXT | Stage B's verdict: `coexists`/`replaces`/`removes`/`none` |
| `superseded_by` | TEXT FK | Which memory replaced it |
| `status` | TEXT | `active` / `superseded` / `evicted` |
| `strength` | REAL | Decay and reinforcement (P5); **not enabled** (`use_strength=False`) |
| `access_count` | INTEGER | As above |
| `last_accessed_at` | TEXT | As above |
| `strength_updated_at` | TEXT | The point `strength` was last brought forward to — separate from access, or applying decay twice at one moment would compound the same interval |
| `token_count` | INTEGER | Computed at write time; the packer treats selection as a knapsack problem and needs each item's weight cheaply |
| `source_session_id` | TEXT FK | Provenance: source session |
| `source_turn_index` | INTEGER | Provenance: which turn |
| `source_char_start` | INTEGER | Provenance: start offset in that turn |
| `source_char_end` | INTEGER | Provenance: end offset |

**Other tables**: `sessions`, `turns`, `memories_fts`, `turns_fts`, `entities`,
`memory_entities`, `evidence`, `meta`.

**Full `scope` → heading map** (`_SCOPE_HEADINGS`):

| scope | Heading |
|---|---|
| `profile` | About the user |
| `preference` | User preferences |
| `plan` | Current plans |
| `event` | Past events |
| `recommendation` | Previously recommended by the assistant |
| `commitment` | The assistant agreed to |
| `shared_context` | Other people and things discussed |
| (unscoped) | Other things known about the user |

**Single-valued predicates**:
`SINGLE_VALUED_PREDICATES = {lives_in, works_as, works_at, uses_framework}`. Four entries,
deliberately conservative; D23 records that this list was wrong on two of seven, and that
two entries were removed for chaining unrelated facts together.

## Appendix B — Mathematical definitions

**Embedding**: $\mathbf e = f_\theta(x)\in\mathbb R^{384}$. $\theta$ is the
`all-MiniLM-L6-v2` weights; this project only runs inference.

**Cosine / normalized inner product**

$$\cos(\mathbf q, \mathbf m) = \frac{\mathbf q^\top \mathbf m}{\lVert \mathbf q\rVert\,\lVert \mathbf m\rVert}
\;\xrightarrow{\ \lVert\cdot\rVert=1\ }\;\mathbf q^\top \mathbf m$$

$\mathbf q$ is the query vector, $\mathbf m$ a memory's `content` vector,
$\mathbf q^\top\mathbf m$ the inner product $\sum_{i=1}^{384}q_im_i$, and
$\lVert\cdot\rVert$ the L2 norm. Every vector is normalized on write and on query (§4.3,
confirmed in code), so the inner product *is* cosine. Range $[-1,1]$, rescaled before
fusion: $S_{\text{sem}}=\mathrm{clip}_{[0,1]}((\cos+1)/2)$.

**BM25**

$$\mathrm{BM25}(q,d) = \sum_{t \in q} \mathrm{IDF}(t)\cdot \frac{f(t,d)\,(k_1+1)}{f(t,d) + k_1\left(1 - b + b\,\dfrac{|d|}{\overline{dl}}\right)}$$

| Symbol | Meaning | Intuition |
|---|---|---|
| $t$ | A query term | One token after tokenization |
| $f(t,d)$ | Term frequency of $t$ in document $d$ | More occurrences is more relevant, with diminishing returns |
| $\mathrm{IDF}(t)$ | Inverse document frequency | Rare terms discriminate |
| $\lvert d\rvert$ | Document length in tokens | — |
| $\overline{dl}$ | Average document length | — |
| $k_1$ | Term-frequency saturation | Smaller saturates faster |
| $b$ | Length normalization $\in[0,1]$ | $b=1$ penalizes long documents fully, $b=0$ not at all |

A common IDF form: $\mathrm{IDF}(t)=\ln\!\left(\frac{N-n_t+0.5}{n_t+0.5}+1\right)$.
This project calls FTS5's built-in `bm25()`, whose return value is the **negation** of the
standard score; $k_1$ and $b$ are fixed inside FTS5 and **not established from the current
material**.

**Lexical normalization** (min-max over the candidates observed for this query,
`lower_is_better=True`):

$$S_{\text{lex}} = \frac{\max_j r_j - r_i}{\max_j r_j - \min_j r_j}$$

$r_i$ is candidate $i$'s raw FTS5 score. When $\min\approx\max$ (one candidate, or all
tied) everything returns 1.0 — "one unambiguous hit gets full credit".

**Recency**: $S_{\text{rec}}=\exp\!\left(-\ln 2\cdot\dfrac{\Delta_{\text{days}}}{H}\right)$,
$H$ = `recency_halflife_days` (default 30), $\Delta$ measured from `event_time` falling
back to `ingested_at`. At $\Delta=H$ the value is exactly 0.5.

**Entity overlap**:
$S_{\text{ent}}=\max\limits_{e\in E(m)}\dfrac{\lvert T(e)\cap T(q)\rvert}{\lvert T(e)\rvert}$,
where $E(m)$ is the memory's linked entities and $T(\cdot)$ tokenizes (`[a-z0-9]+`, length
> 2). Max, not mean.

**Weighted fusion**:
$S(m,q)=w_sS_{\text{sem}}+w_lS_{\text{lex}}+w_rS_{\text{rec}}+w_iS_{\text{imp}}+w_eS_{\text{ent}}$,
optionally multiplied by `strength` (off by default).

**Session aggregation**

$$A_{\max}(s)=\max_{m\in M_s}S(m,q)\qquad A_{\text{sum}}(s)=\sum_{m\in M_s}S(m,q)$$
$$A_{\text{mean}}(s)=\frac{1}{\lvert M_s\rvert}\sum_{m\in M_s}S(m,q)\qquad A_{\text{top3}}(s)=\sum_{j=1}^{3}S_{(j)}$$

$M_s$ is the **retrieved** memories from session $s$ (not all of that session's memories),
and $S_{(j)}$ the $j$-th highest score among them.

**Accuracy**: $\text{Accuracy}=\dfrac{\#\{\text{questions the judge marked correct}\}}{N}$

**Source-session Recall@k**:
$\text{Recall@}k=\dfrac1N\sum_{i=1}^{N}\mathbf 1[g_i\in R_i^{(k)}]$, with $g_i$ the gold
session and $R_i^{(k)}$ the top-$k$ sessions. Measured at several stages (candidates →
ranked → selected → assembled), because one end-to-end number cannot say which stage lost
it.

**Exact McNemar test.** Two systems run the same questions:

|  | B correct | B wrong |
|---|---:|---:|
| **A correct** | $a$ | $b$ |
| **A wrong** | $c$ | $d$ |

$a$ and $d$ carry no information about the difference and are discarded. Under the null,
each discordant pair independently falls into either cell with probability $1/2$:

$$b \sim \text{Binomial}(b+c,\ 0.5)$$

and the two-sided exact $p$-value is that binomial's two-tailed probability. **Why
disagreement rather than two accuracies**: 70% and 72% could mean "the same 70 correct plus
2 more" (strong evidence) or "B got 15 of A's correct ones wrong and 17 of its wrong ones
right" (almost none). Only the discordant cells separate those.

**Fisher exact test**: for **unpaired** proportion comparisons, e.g. one question type on
dev50 versus heldout100. Every per-type comparison here has $p \ge 0.12$.

**Risk ratio**: $RR = p_{\text{late}} / p_{\text{early}}$, used in the zero-yield analysis.
Observationally $RR = 2.86$, batch-clustered bootstrap 95% CI $[2.15, 4.35]$, within-batch
permutation $p = 0.00005$ ($N=20{,}000$). ⚠️ That is **observational** — each session sat at
one position; the causal evidence is the randomised control in §6.4.

## Appendix C — Configuration

`configs/fallback.yaml` (the frozen v1 configuration) and `configs/v2.yaml` (the train150
candidate) differ only in the `context` block.

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
  sessions_per_request: 15    # known-lossy, see §3.2 / §6.4
  two_stage: true
  checkpoint_every: 5
  dedupe_similarity_threshold: 0.92

temporal_resolution: true

fallback:
  enabled: true
  max_turns: 3
  max_chars: 2400

context:                      # read only by the two_stage_coherent variant
  max_sessions: 3
  window_radius: 1            # v2: chosen by the fixed rule in §6.6
  max_total_memories: 30      # v2
  aggregate: mean             # v2
  session_order: chronological
  include_superseded: false
```

**Why the model roles must be separate**: a judge must not grade prose from the model that
wrote it (D4, and judge-versus-human agreement is **measured**); and the free tier meters
requests per model (the 429 body reports `…PerProjectPerModel-FreeTier`), so three roles on
three models means three pools — which is what stops a large ingest from starving
evaluation. The sizes have measured reasons too: the full flash models allow 20
requests/day, not enough for one 50-question run; flash-lite is limited per minute;
extraction needs 250k TPM to finish a ~6.1M-token dev pass in minutes; judge inputs are
~200 tokens so TPM is irrelevant, while gemma's 1,500 requests/day is what judging needs.

## Appendix D — Engineering incidents

None of the mechanisms in §7 was designed in advance. Each came from one of these.

**A store was written by two extractor generations and nothing could notice.** The
checkpoint recorded nine progress counters and **no extractor version**, while
`set_meta("extractor_version", …)` **overwrites** on every run instead of comparing. A
store ingested to 63% by the pre-P10 extractor was resumed to completion by the P10
extractor; both runs succeeded and every structural check passed. The two generations'
`source_role` distributions are 93.3%/6.7% versus 44.8%/54.4%. **The fix was not a
version-string comparison** — that string is hand-edited and misses the likelier drift of a
reworded prompt. Resume now compares a fingerprint computed from the inputs and names
**which component moved**. The store was rebuilt; the old one is kept unmodified as a
diagnostic record, and its 60.0% is permanently labelled "diagnostic only, excluded from
formal benchmark claims".

**A gate printed its verdict and exited 0.** That reads correctly to a human and is
invisible to anything checking a return code, so an unattended sequence would have
continued into the run the gate exists to prevent. Gates now exit non-zero when closed.

**A stale artifact does not look like an error.** Reading `gate_open: true` from
`temporal-gate.json` is not evidence that *this* run's gate passed — if the gate crashed
before writing, the file on disk is the previous run's verdict. Freshness is now checked
against the run's own start time.

**An error message asserted an unverified cause.** `2348/2400 sessions — quota probably ran
out`: quota had never stopped. It was **explicitly withdrawn** rather than quietly
corrected, because *an error message that guesses a cause is worse than one that reports
only what it observed — it directs the next person away from the real problem*. The same
check also compared the wrong two numbers: the corpus has 2,400 session *entries* but 2,348
unique session *IDs*, because one session is evidence for more than one question.

**Two simultaneous ingests overwrote each other's quota accounting.** 211 requests recorded
against roughly 320 made. Nothing in the store looked wrong — what was corrupted was the
**accounting**, not the data. A cross-process pid advisory lock was added.

**A metric was wrong three times — more often than the extractor was.** The fidelity metric
was wrong twice before the extractor was, and a third time at the worked-examples revision
(D28, D30). A wrong metric that produces a right conclusion is luck.

**Session ids were not user-scoped.** Session recall came out at roughly 64%, disagreeing
with every other measurement. Two errors compounded: the script mixed unfinished questions
into the denominator, and it compared **scoped internal ids** against **public dataset
ids**. Corrected it is 95%+, and the 64% is explicitly withdrawn. The legacy store was
migrated **without re-extracting** its 9,588 memories, with a SQLite backup retained.

**The clean store briefly broke the Mayo case, and the obvious diagnosis was wrong.** After
a rebuild that *improved* retrieval, the Mayo question failed in both formal arms. The
tempting explanation was the level branch — the first version took source turns whenever
retrieval returned anything, with nothing checking the memories were about the question.
That **is** a real defect, but **not this one**. The actual cause was truncation:
retrieval **had** found the right conversation, but `turns_for_memories` returns turns
**ordered by session id**, the code kept `[:max_turns]`, and the gold turn sat tenth of
sixteen alphabetically — the three kept were all from a conversation about live music.
**Better recall meant more candidates, and more candidates meant the answer was sliced
off.** On the smaller mixed store the same question retrieved nothing and never met the
slice, which is why the defect survived unseen. Both are the same omission — nothing ranked
the candidates against the question — so ranking them fixed both.
`single-session-assistant` went 5/6 → **6/6**, and the arm 66.0% → **72.0%**.

**A per-key verdict retired two statements that were both true.** An earlier resolver
retired every consecutive pair once a key was deemed resolvable. Observed live: *"averaging
$100 per week on groceries"* was retired by *"spent $75 at Walmart last Saturday"*, because
some **third** memory on `grocery_spending` had signalled a change. Stage B's verdict scores
0% false supersede in isolation; the loss was entirely in the integration, so the per-fact
verdict must be honoured per fact (§3.5 step 6).

**A freeze enforced a field named "for reference only".**
`git_head_for_reference_only` was compared leaf by leaf, so **any commit** — even one
touching only `docs/` — invalidated a pre-ingest freeze, and a freeze captured against an
uncommitted tree could only be honoured by never committing the work it described. This
directly caused the 37% dev100 store to be discarded (§8.2). The fix: fields whose names
end in `_for_reference_only` are recorded and not compared, while every file that affects a
result stays hashed individually.

## Appendix E — Code map

| Component | File |
|---|---|
| Schema, indexes, FTS5 triggers | `src/llm_long_term_memory/store/schema.sql` |
| SQLite store, BM25 queries, archive search | `store/sqlite.py` |
| Exact vector index | `store/vector.py` |
| User-scoped session ids | `store/session_keys.py` |
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
| Context selection | `evaluation/context_selection.py` |
| Freezes and lineage validation | `evaluation/reproducibility.py`, `scripts/freeze_v2.py` |
| Validation and the decision rule | `evaluation/validation.py`, `scripts/run_validation.py` |
| Ingest state checks | `evaluation/ingest_state.py`, `scripts/check_ingest_state.py` |
| Cross-process lock | `locking.py` |
| Config definitions and default weights | `config.py` |

## Appendix F — Experiment index

| Content | File |
|---|---|
| 30 numbered design decisions | `docs/DECISIONS.md` |
| Chronological engineering report | `docs/ENGINEERING_REPORT.md` |
| Productization acceptance criteria | `docs/PRODUCTIZATION_V2_PLAN.md` |
| Dataset role protocol | `results/data-protocol.md` |
| Randomised batch-size / position control | `results/batch-position-pilot.md` |
| Three repeats measuring run noise | `results/heldout-variance.md` |
| Failure staging and module ceilings | `results/failure-stages.md` |
| Five-question context-shape probe | `results/context-arms.md` |
| Retrieval weight sweep | `results/retrieval-weights.md` |
| Three offline diagnostics, all negative | `results/offline-diagnostics-2026-08-25.md` |
| Rerank Pareto | `results/rerank-pareto.md` |
| Raw-retrieval diagnostic | `results/raw-recall-diagnostic.md` |
| The three pre-registrations | `results/prereg-batch-size.md`, `prereg-context-shape.md`, `prereg-v2-final.md` |
| v2 runbook | `results/v2-runbook.md` |
| v2 progress log | `results/v2-progress.md` |
| train150 final artifacts | `results/analysis/train150-*.final.*`, `train150-context-grid-final/` |
