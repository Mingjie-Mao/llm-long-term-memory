# Engineering Report — Building a Persistent Long-Term Memory Layer for LLM Applications

**Project:** `llm-long-term-memory` (internal name ChronoMem)
**Report date:** 2026-08-17
**Status:** productised prototype; **held-out result measured 2026-08-19 — 70.0% on 100
unseen questions against dev50's 72.0%** (section 11).

> ⚠️ **This document is a historical record and is no longer maintained.** It describes
> the system as of 2026-08-17 and is superseded by [`REPORT.md`](REPORT.md), which is
> organized by data flow rather than chronology and carries the current numbers. Kept
> because its incident narratives — particularly the mixed-extractor-store investigation
> in section 10 — are the primary account of how several safeguards came to exist.
>
> Figures here that have since moved: the test count (385 → 532), the memory count, and
> the status line above. Most importantly, **the answerer's run-to-run variance is no
> longer unquantified** — three repeats of `heldout100` flip 6 of 100 verdicts with an
> accuracy range of 3 points ([heldout-variance.md](../results/heldout-variance.md)), so
> the "unstated error bar" this report warned about has since been stated.

Every number below was recomputed from committed artifacts in `results/raw/` at the
time of writing, and each is labelled with the file it comes from. Where a figure
appeared in an earlier draft and did not survive re-checking, the report says so
rather than quietly dropping it.

---

## 1. Executive summary

A long-term memory layer for LLM agents: it turns conversations into typed, time-
bounded facts, resolves those facts when they change, retrieves them under a token
budget, and — when the compression turns out to have dropped what a question needs —
recovers the original conversation turn instead of failing.

Four findings shaped the product, all of them measured:

1. **Structured memory is a viable compression.** On a 31-question pilot,
   `two_stage` at k=10 answered 51.6% using a median of **242 context tokens**
   against `naive_rag`'s 51.6% on **13,057** — ~54x less context, with no detectable
   accuracy difference (6W-6L, p = 1.000). *(recomputed from `results/raw/*.jsonl`)*
2. **The gain came from extraction, not from attaching raw evidence.** An initial
   draft credited evidence hydration. A causal ablation reversed that: the
   extraction rewrite is worth 11W-2L (p = 0.022), hydration 2W-1L (p = 1.000) at 3x
   the context. *(`results/a2-pilot.md`)*
3. **Two plausible upgrades were built, measured, and not shipped.** Cross-encoder
   reranking changed *zero* answers at k=10 while lowering source recall
   (`results/rerank-pareto.md`); a utility predictor for budget packing lost to
   predicting the mean (`results/p6-pilot.md`).
4. **The evidence for dense raw retrieval dissolved under audit.** A constructed
   query set suggested BM25 collapses on paraphrases (6.9% R@1); manual inspection
   showed the construction had deleted the questions, not just their vocabulary.
   Hand-written paraphrases retrieve at rank 1. *(`results/raw-recall-diagnostic.md`)*

A live regression on the 7 questions that motivated the most recent fixes moved them
from **0/7 to 6/7, identical across three runs** — including the golden case, where
the answerer declared structured memory insufficient, the archive returned the
original assistant turn, and the answer carried the exact URL extraction had dropped
(`results/live-regression-v2.md`). Those 7 questions were selected because they
failed, so this is a regression signal, not a measurement of the system.

Shipped: a REST API, Docker image (2.95GB), structured request logging, an MCP
server over the same service layer, and a Memory Inspector that shows why each
memory was selected, passed over, or superseded — with shareable URLs. 385 tests,
three-platform CI.

**Not claimed:** any comparison against a third-party system, and no *baseline* on
held-out data — `heldout100` was run for this system only, so `full_context` and
`naive_rag` have no unseen numbers. The dev-50 set was used for prompt iteration and
gate tuning, so every dev-50 figure measures the fit of those choices as much as the
system.

**The formal dev-50 result now exists** (section 10.6). The store it was measured
on was rebuilt end to end under one extractor generation after the first attempt
turned out to mix two; that earlier 60.0% stays in section 10 as a diagnostic and
is carried into no comparison table. On the clean store, over the frozen dev50
manifest: **56.0%** from structured memory alone and **72.0%** with the conditional
raw-conversation fallback, the two arms differing in nothing else (9W-1L, exact
McNemar p = 0.022). Section 10.6 explains why that p-value should be read as
adaptive rather than as evidence of generalisation.

---

## 2. Problem and product goals

A vector store with top-k retrieval answers *"what did the user say?"* It does not
answer *"what is currently true?"*. Given

```
Jan   I use TensorFlow.
Mar   I'm learning PyTorch.
Aug   I've switched completely to PyTorch.
```

naive RAG surfaces all three and lets the model guess. The product goals follow:

| Goal | Why it is not free |
|---|---|
| Answer from a fraction of the history | Context is the dominant cost of long-term memory |
| Track facts that change | Superseded facts must stop being retrieved without being deleted |
| Explain every answer | "Which memory produced this?" is unanswerable in a plain vector store |
| Never lose a detail permanently | LLM extraction is lossy compression |
| Apply memory, not just recall it | Retrieving a preference and ignoring it is a product failure |

The last two are the ones that took longest to understand, and section 8 covers how
each was diagnosed.

---

## 3. System architecture

![System overview: structured memory and recoverable source evidence](figures/overview.svg)

The figure has been redrawn against the current working-tree implementation. See the
[architecture atlas](ARCHITECTURE.md) for all six figures and entry-point differences;
the research narrative below retains its historical context.

**The raw archive is a storage decision, not a retrieval algorithm.** This
distinction matters and was initially blurred in our own notes: keeping original
turns is what makes lossy extraction *recoverable*; how those turns are found —
source-local lookup, BM25, dense, hybrid — is an independent choice that can change
without touching the storage model. Section 6 covers what is stored; section 7
covers how it is searched today and what would change that.

| Layer | Choice | Rationale |
|---|---|---|
| Store | SQLite (WAL) + FTS5 | One file, no service dependency; BM25 for free |
| Vectors | Exact flat inner-product over numpy | 4.8k memories; an ANN index would add a dependency to save microseconds |
| Embeddings | `all-MiniLM-L6-v2`, local | Corpus-wide embedding is ~53M tokens; the API quota is the binding constraint |
| LLM | Gemini, three independently configured roles | Extractor, answerer and judge have different requirements |
| Service | FastAPI, one composition root | The MCP server and Inspector are clients of the same service object |

---

## 4. Memory ingestion and update pipeline

### 4.1 Two-stage extraction

Stage A reads sessions and emits fact lines; Stage B assigns each a temporal key and
an update operation. Two requests per batch rather than one.

The split is not stylistic. A rule-based Stage B was free and scored **43% predicate
accuracy with zero supersessions** over 148 memories; the LLM version passes the
temporal gate on all four metrics. That doubled ingestion from ~190 to ~380 requests
and was worth it, because a wrong temporal key makes the whole timeline layer a
no-op — two spellings of the same relation never collide, so nothing is ever
detected as superseded.

### 4.2 `source_role` is not `subject`

The single most consequential schema fix in the project. The original code derived
the subject from a string prefix:

```python
subject = "assistant" if fact.lower().startswith("the assistant") else "user"
```

That made `subject` a two-valued speaker flag. *"Andy wore a blue shirt"*, said by
the user, was stored as a fact about the user — and **no amount of prompt tuning
could have fixed it, because the schema had nowhere else to put it**.

Who said something and who it is about are different questions:

| | `source_role` | `subject` |
|---|---|---|
| "I stopped drinking coffee" | user | user |
| "I recommend Mod Podge" | assistant | assistant |
| "Andy wore a blue shirt" (user speaking) | user | andy |

Both now exist, alongside `scope` (profile / preference / plan / recommendation /
commitment / event / shared_context). Stage A emits all three, because Stage B never
sees the conversation — speaker identity is unrecoverable after Stage A.

**Migration recovers the past exactly.** The old extractor set `subject='assistant'`
precisely when the fact began "The assistant", so backfilling `source_role` from it
is a derivation, not a guess. Verified on the live store: 326 assistant memories
recovered, 4,517 user, zero mismatches, 4,843 rows preserved. `scope` stays NULL on
pre-P10 rows — it is genuinely unknown, and a made-up value would be
indistinguishable from a real one at query time.

*Status: code and tests complete; the store has not yet been re-ingested under the
new extractor, so the current 4,843 memories still carry the old framing.*

### 4.3 Temporal resolution

Every fact carries a validity window. For each `(subject, predicate)` key the
resolver sorts by event time and rewrites the whole chain.

Rebuilding rather than pairwise comparison is load-bearing. Sessions arrive in
arbitrary order, so "the newest memory supersedes the current head" lets a
late-arriving January fact become current, and a fact landing between two existing
ones could never rewire a link that already points past it. Rebuilding is idempotent,
order-independent, and costs no LLM calls.

Two refinements came out of writing the tests:

- **Restatements are not moves.** "I live in Canberra" in March and again in June is
  one interval owned by March, so "when did you move?" answers correctly.
- **Arity is the default rule, not the whole rule.** `uses_tool` is multi-valued —
  using PyTorch does not stop you using NumPy — so extraction also emits
  `replaces_previous` when the user's own wording signals a replacement.

The first real ingest disproved part of the arity list immediately: `lives_in`
produced `Tokyo → South Bay → Las Vegas`, which is right, while `scheduled` produced
`layover in London → cooking class → the 9:15 train`, which is not a sequence of
competing values at all. `scheduled` and `has_goal` were removed. That mistake cost
seconds because resolution reads stored data and calls no model.

---

## 5. Retrieval architecture

Five signals, each normalized to [0,1] before weighting: semantic, BM25, recency,
importance, entity overlap. Setting a weight to zero disables that signal, which is
how ablation rows are produced.

**Four of them are set to zero in the shipped configuration**, and that was not a
decision anyone recorded. `RetrievalWeights` defaults to `semantic: 1.0` with the
rest at `0.0` — the ablation setting — and neither `baselines.yaml` nor
`fallback.yaml` overrides it, so every result in this report was measured on
semantic similarity alone. The other four are computed on every query and
multiplied by zero: the top hit for the question that exposed this scored
`bm25=0.58`, which contributed nothing to its rank.

That makes "hybrid retrieval" a description of the machinery rather than of the
product, and it is one reason a question like *"how many pages was the other
novel?"* is hard here — cosine similarity does not distinguish four memories that
each contain a different 416. Turning the weights on is a configuration change
with an ablation attached, not a feature to build; it is deliberately not bundled
with any other change.

Normalization is not cosmetic. FTS5's `bm25()` is negative and unbounded while
cosine similarity is bounded; adding the raw values would let the lexical term
dominate or vanish according to an implementation detail rather than a configured
weight.

Retrieval is **namespace-isolated**. In the benchmark the namespace is the question
id; in the product it is the user. Ingesting fifty simulated users under one id
produced a `lives_in` chain running Toronto → Greenville → Seattle → Tokyo →
Shanghai → Hyderabad → Las Vegas and let one question retrieve another's evidence.
The API enforces the same boundary, and cross-namespace reads return **404 rather
than 403** — 403 would confirm that an id exists.

**Staged recall instrumentation.** Each answer records whether the evidence session
survived at each stage: candidates → ranked → selected → hydrated. One
end-of-pipeline recall number cannot distinguish "never found it" from "found it and
then dropped it", and those call for opposite fixes. Section 7.3 is a direct
consequence of having this.

---

## 6. The raw conversation archive

### 6.1 Why it exists

LLM extraction is lossy compression, and the losses are not random: the gist
survives and the artifact disappears. From `results/assistant-gap.md`, a real case —
the source turn reads

> *"…the Mayo Clinic: 'How to Sit Properly at a Desk to Avoid Back Pain',
> https://www.youtube.com/watch?v=UfOvNlX9Hh0"*

and structured memory holds nothing about Mayo Clinic at all. If the original turn
were discarded, the URL would be unrecoverable. Because it is kept, it can be
recovered.

This is the same reasoning MemMachine gives for preserving raw conversational
episodes rather than relying on LLM extraction as the sole representation
([arXiv:2604.04853](https://arxiv.org/abs/2604.04853)). We reached it from a failure
trace rather than from the paper, and the architectures agree on the principle while
differing in emphasis: MemMachine minimises routine extraction, whereas this system
extracts aggressively and treats the archive as the recovery path.

**The archive is not an excuse for lossless extraction.** If extraction improves to
the point of losing nothing, structured memory becomes the conversation again and the
compression claim evaporates. The design accepts loss and requires that it be
*recoverable*.

### 6.2 Conditional, not always-on

Attaching raw evidence to every answer was measured. `two_stage_hydrated` tripled
median context (439 → 1,318 tokens) for no detectable accuracy gain — a slow slide
back to naive RAG. So recovery is conditional and the answerer decides:

```
first call → AnswerVerdict{status: answer | need_source | no_evidence}
  answer       → done, one LLM call
  need_source  → source-local: the turns those memories came from
  no_evidence  → archive-wide: BM25 over every turn in the namespace
              → nothing found: decline, and stay declined
```

Level 1 is preferred when it applies: a memory that names the right session is
stronger evidence than a keyword match over everything, and costs no search.

The final branch is a guard rail with a test of its own. Abstention is a measured
strength — 100% on the frozen v1 rows against `full_context`'s 50% (`results/table.md`)
— and a fallback that answers from whatever BM25 returned would trade it away.

---

## 7. Experiments and engineering decisions

Each below is stated as problem → experiment → result → decision. Paired comparisons
use an exact McNemar test over disagreements only; comparing headline accuracies
cannot separate a real improvement from re-running the same configuration, which on
this setup moved a variant between **48.0% and 54.0%** unchanged (`results/table.md`).

### 7.1 Does structured memory lose too much to be useful?

**Problem.** The frozen v1 system scored 26.0% against naive RAG's 54.0% on the
stratified 50 (`results/table.md`). Source-session retrieval usually succeeded, so
the loss was in the representation.

**Experiment.** Rewrite extraction (two-stage), and evaluate on the 31 fully
ingested namespaces (`results/manifests/dev31-pilot.json`).

**Result** *(recomputed from `results/raw/*.jsonl`, n=31)*:

| Variant | Accuracy | Median ctx tokens | Source-session recall |
|---|---:|---:|---:|
| `full_context` | 64.5% | 109,605 | — |
| `naive_rag` | 51.6% | 13,057 | — |
| `two_stage` k=10 | 51.6% | **242** | 87.1% |
| `two_stage` k=20 | 48.4% | 439 | 93.5% |
| `chronomem` (v1) | 19.4% | 465 | — |

**Decision.** Ship `two_stage`. The defensible claim is **equal accuracy at ~54x
less context** (6W-6L vs `naive_rag`, p = 1.000) — not better accuracy.

**Limits.** n=31, unstratified (an ingest-order prefix), one run each. The v1 row
reads 19.4% here and 26.0% on the stratified 50; they are different question sets and
must not be compared. p = 1.000 means *no difference was detected*, not that the
systems are equivalent.

### 7.2 Which change actually produced the gain?

**Problem.** An earlier draft of `results/a2-pilot.md` attributed the improvement to
evidence hydration. That was an assumption: two things changed at once.

**Experiment.** Evaluate `two_stage` (no hydration) on the same 31 questions.

**Result:**

| Step | Δ | Paired |
|---|---:|---|
| `chronomem` → `two_stage` (extraction rewrite) | **+29.0pp** | 11W-2L, **p = 0.022** |
| `two_stage` → `two_stage_hydrated` (hydration) | +3.2pp | 2W-1L, p = 1.000 |

Source-session recall is **93.5% for both**, which confirms the attribution:
hydration runs after retrieval and cannot change what is recalled, so the
80.0% → 93.5% improvement belongs to extraction too.

**Decision.** Ship `two_stage`; demote hydration from an always-on stage to the
conditional fallback of section 6. Hydration is *not shown to be useless* — 3
disagreements show nothing in either direction — but it is not entitled to 3x
context by default.

### 7.3 Does cross-encoder reranking help?

**Problem.** The v1 failure was mis-ranking, and a cross-encoder is the standard fix:
it scores query and memory jointly, which a bi-encoder cannot.

**Experiment.** Sweep `top_k ∈ {20,10,5} × rerank ∈ {off,on}`, 31 questions
(`results/rerank-pareto.md`).

**Result:**

| top_k | rerank | Accuracy | Median ctx | Source recall |
|---:|---|---:|---:|---:|
| 20 | off | 48.4% | 439 | **93.5%** |
| 20 | on | 48.4% | 420 | 90.3% |
| 10 | off | 51.6% | 242 | **87.1%** |
| 10 | on | 51.6% | 240 | 83.9% |
| 5 | off | 48.4% | 146 | **87.1%** |
| 5 | on | 48.4% | 143 | 83.9% |

**At k=10 the two arms answered all 31 questions identically** — zero disagreements,
which is stronger than "no detectable difference". Reranking lowered source recall at
every k.

**Decision.** Not enabled. The implementation and `configs/rerank.yaml` stay in the
repo so the result is reproducible rather than asserted, and the dependency moved to
its own `rerank` extra so running the product does not download a reranker model
nobody asked for.

**Related work.** [arXiv:2606.04194](https://arxiv.org/abs/2606.04194) reports an
off-the-shelf cross-encoder over a fused top-10 degrading LoCoMo Hit@1 by 6.9 pp.
That is a different corpus, retriever and metric, and their own paper limits the
conclusion to the reranker configuration tested — so this is **directionally
consistent, not a replication**.

### 7.4 Does the raw fallback need dense retrieval?

**Problem.** The fallback searches with BM25. Dense retrieval would mean embedding
every turn, a second index, more ingest cost and a model in the serving path.

**Experiment.** Measure *evidence recall* rather than end-to-end accuracy — accuracy
folds retrieval, answering and judging into one number and cannot say which moved.
Gold turns come from LongMemEval's own `has_answer` flags. Three query types, 31
questions (`results/raw-recall-diagnostic.md`).

**First result:**

| query type | n | R@1 | R@3 | R@5 | MRR |
|---|---:|---:|---:|---:|---:|
| keyword (upper bound) | 31 | 93.5% | 96.8% | 96.8% | 0.946 |
| natural | 31 | 54.8% | 74.2% | 74.2% | 0.640 |
| disjoint (constructed) | 29 | 6.9% | 6.9% | 6.9% | 0.069 |

6.9% looks like decisive evidence that BM25 collapses on paraphrases.

**The audit that reversed it.** The `disjoint` queries were built by deleting every
content word the question shared with its gold turn. Spot-checking them — the step
meant to validate the construction — showed the deletion removes the *question*:

```
"How long have I been collecting vintage cameras?"   →  "long"
"What health issue did I initially think was a cold?" →  "health issue think"
```

No retriever answers `"long"`. That row measures degraded input, not lexical
brittleness, and **cannot be cited as evidence for embeddings**.

**Second result — hand-written paraphrases.** Five phrasings of the Mayo Clinic
question, gold turn `answer_sharegpt_81riySf_0:1`: **all rank 1**, including
phrasings containing no "Mayo", "Clinic" or "YouTube". Genuine paraphrases keep topic
words (posture, ergonomics, sitting, back pain) and drop only the proper noun, which
BM25 never needed.

**Third result — what `natural` actually misses.** All 8 misses are
`single-session-preference` (the gold is a rubric, not a fact), `temporal-reasoning`
(the answer is computed from dates), or `multi-session` (the answer aggregates across
turns). None has a single gold turn for any retriever to rank first.

**Decision.** **Dense raw retrieval deferred** — not rejected. No demonstrated
product-level lexical-recall gap currently justifies the complexity. Two conditions
would reopen it: a model-generated paraphrase set over 30–50 cases, or a corpus of
terser turns where topic words do not co-occur so reliably.

**Related work.** [arXiv:2606.04194](https://arxiv.org/abs/2606.04194) finds
lexical–dense fusion worth +11.2 pp Hit@1 over BM25 on LoCoMo. Our decision is about
a different corpus and a different retrieval stage, and does not contradict it —
their result is one reason the deferral is written as *deferred*.

### 7.5 Does a learned utility predictor beat relevance for packing?

**Problem.** Budget packing selects memories by relevance. A learned utility signal
might select better.

**Experiment.** Leave-one-out influence labels over 10 questions, ridge regression,
5-fold cross-validation grouped by question (`results/p6-pilot.md`).

**Result.** Held-out RMSE 0.310 against a mean-baseline 0.263 — the predictor does
not beat predicting the mean. A budget sweep using it was deliberately *not* run:
applying a failed predictor to its own calibration questions would produce an
optimistic, uninformative comparison.

**Decision.** Not shipped. Relevance packing stays the baseline.

---

## 8. Failure analysis

Not every wrong answer is a retrieval failure. Eight classes, each with a real case
and a different owner:

| Code | Class | Real case | Owner |
|---|---|---|---|
| **S** | Storage / schema policy | Third-party facts ("Andy wore a blue shirt") had no representable subject | Schema |
| **E** | Extraction / representation | "17 vintage cameras" kept, "three months" dropped | Extractor prompt |
| **R** | Retrieval | Correct session never enters the candidate set | Ranking |
| **H** | Hydration / evidence recovery | Memory anchors to the right turn but the span is missing | Fallback |
| **A** | Answerer / application | Retrieved the user's power bank, replied "I do not know what phone you use" | Answerer prompt |
| **T** | Temporal reasoning | "How many weeks ago did I receive the chandelier?" | Reasoning layer |
| **M** | Multi-session aggregation | "How many museums in February?" | Query decomposition |
| **J** | Judge / evaluation | A correct cookie recommendation marked wrong for not restating the rubric | Judge prompt |

### 8.1 S — the assistant gap

`single-session-assistant` scored **0/4 for every memory variant** while
`full_context` and `naive_rag` scored 4/4 (`results/assistant-gap.md`). A gap that
uniform is a policy defect, not a ranking problem.

Tracing all four: the raw evidence was present in every case (`source_session_id`
resolves to a turn for 4,843/4,843 memories), but three answers lived in **assistant**
turns and the fourth described a third party. The extraction prompt already contained
an instruction to record assistant facts — so this was not a missing rule but **a rule
the model did not reliably follow**: 326 of 4,843 memories (6.7%) carry
`subject='assistant'`, against 12 of 2,007 (0.6%) in v1. The two-stage rewrite
improved compliance 12x and still fell short.

Fixed in the schema (§4.2) and the prompt, which now carries worked examples for an
assistant recommendation and a third-party subject — the one instruction that had no
example was the one being ignored, in a project that had already measured worked
examples moving source fidelity 33.6% → 36.6%.

### 8.2 A and J — recall is not application

All three `single-session-preference` questions scored 0/3 for *every* system,
`full_context` included (`results/preference-audit.md`). That ruled out memory as the
cause. Hand-auditing the three found two distinct defects:

**J — the judge.** LongMemEval's preference "gold" is a rubric describing a
well-personalized reply, not a reference answer. Grading it as one failed a reply
that had done exactly what the rubric asked (it recommended turbinado sugar), with
the reason *"the reference answer describes the user's preferences, whereas the
candidate answer provides actual suggestions"*. LongMemEval's own evaluation uses a
separate preference prompt; ours now routes by question type, with abstention
outranking everything and unknown categories falling back to the reference judge.

**A — the answerer.** On the battery question the system **retrieved the right
memories** — the reply names the power bank and the charging pad — and then declined:
*"I do not know what phone you use"*. The old prompt optimised for factual recall and
abstention, which is why abstention is 100%, and that same disposition refuses on
advice-shaped questions.

The fix draws the distinction that matters: *missing fact* versus *available
context*. It still declines when something was never discussed, and personalizes when
the memories support a useful reply. Memories are now grouped by `scope` under
headings ("User preferences", "Current plans", "Previously recommended by the
assistant") so a constraint does not read like a candidate answer.

*Status: **verified live 2026-08-14**. Both fixes confirmed on the questions that
motivated them — the cookie answer now passes the rubric judge, and the battery
answer applies the user's power bank instead of declining. See
`results/live-regression-v2.md`.*

### 8.3 T and M — not retrieval problems at all

"How many weeks ago did I receive the chandelier?" needs date arithmetic over
retrieved evidence. "How many museums did I visit in February?" needs aggregation
across turns — no turn states the count. A better retriever returns the same
evidence and the answer is still absent.

These are recorded in the regression corpus as `reasoning_required: true` and
**excluded from retriever comparisons**, because scoring a retriever on a question
with no single gold turn measures the wrong thing.

---

## 9. Productization

### 9.1 Service

`src/llm_long_term_memory/api/`, three layers, no domain logic in handlers:

```
POST   /v1/messages            ingest a turn, return the memories it created
POST   /v1/memories/search     retrieve, with signals, provenance and rejections
POST   /v1/raw/search          the fallback layer, queryable directly
GET    /v1/memories            browse a namespace; filter by status/type/scope/role
GET    /v1/memories/{id}       one memory with its source turn
GET    /v1/timeline            the supersession chain for a (subject, predicate)
DELETE /v1/memories/{id}       forget (marks evicted; never a hard delete)
GET    /healthz  GET /v1/config
```

`service.py` holds the domain and no HTTP, so the MCP server and the Inspector are
clients of the same object rather than second implementations.

**`rejected` is the differentiating field.** Search returns why a candidate did *not*
come back — `superseded` (the fact was true and no longer is; returning it would be
wrong) or `below_rank` (still true, lost on score; returning it would have been
affordable). Absence without explanation is indistinguishable from a retrieval bug.

### 9.2 Operations

- **Docker**: non-root, no datasets/models/credentials baked in, store on a mounted
  volume, healthcheck hitting the real `/healthz` so an unreadable volume reports
  unhealthy instead of accepting traffic.
- **Structured logging**: one JSON event per request with request id, route, status,
  latency and a config fingerprint; `x-request-id` returned so a bug report can name
  a log line. Asserted to contain no memory content and no credentials. Cost in
  currency is deliberately absent — a USD figure invented from a list price would
  look authoritative and be wrong.
- **Encoder warm-up at startup**: the lazily-loaded embedder moved a 12.5s model load
  onto whichever user arrived first. Warming it cut the first request from
  **12,486ms to 181ms** and made logged p95 a property of the system rather than of
  process age.
- **Cross-platform**: 65 file I/O sites given explicit `encoding="utf-8"`, CI across
  ubuntu/windows/macos, and `filterwarnings = ["error::EncodingWarning"]` so the
  defect fails a test on any platform rather than only where the locale differs.
- **Result versioning**: every evaluation row records `answer_prompt_version`,
  `judge_prompt_version` and `extractor_version`. The extractor version is read from
  the *store*, not the checkout, because it describes the data being evaluated.
  Defaults are `None` — rows written before stamping genuinely do not know.

### 9.3 Memory Inspector

A single self-contained page at `GET /`, a client of the documented endpoints only.
It groups a query's retrieval into **Selected memories** (by scope, with signal bars
and an expandable source turn with the extracted span highlighted), **Not selected —
no longer true**, **Not selected — ranked below the cut**, and the **raw archive
preview**.

State is split by who needs to see it: the URL carries `namespace` and `q` so a view
survives a reload and pastes into another browser; localStorage carries the chat
transcript; memories and raw turns are in neither, because a browser copy goes stale
against the store. Fixed demo routes (`?demo=mayo`, `?demo=timeline`,
`?demo=collectibles`) rewrite themselves into plain shareable links.

> **Superseded 2026-08-27.** The demo routes and the recorded-run replay were
> removed: each hard-coded an evaluation namespace and the no-parameter case
> defaulted into one, so the inspector opened onto a synthetic persona's private-
> looking history with nothing marking it as a fixture. See the dated note in
> [ROADMAP.md](ROADMAP.md) for what replaced them, and for the `autocomplete`
> defect on the query box found in the same pass.

**The Inspector became a debugging surface within minutes of existing.** It rendered
a `collectibles` key as a supersession chain — G.I. Joe *replaced by* stamps
*replaced by* a Mickey Mantle card — which looked convincingly like a data defect
until the store showed all eight rows `active` with no `superseded_by`. The bug was
in the renderer: it drew an arrow between every pair of entries in a key, and most
predicates are multi-valued. Two further bugs surfaced the same way: copy claiming
"structured memory answered it" when nothing had evaluated sufficiency, and a
`const history = [...]` that shadowed `window.history` and blanked the panel with a
TypeError while the chat pane kept rendering from localStorage — which made a
frontend bug look like a backend failure.

---

## 10. Evaluation integrity: the mixed-extractor store incident

*2026-08-15. This section documents a defect that invalidated a planned formal
result, the diagnostic run made in its place, and the rebuild that replaces it.
Nothing in it is a benchmark claim.*

### 10.1 Incident

The ingestion checkpoint records nine progress counters — `done_sessions`,
`memories_written`, `duplicates_dropped` and so on — and **no extractor or schema
version**. `store.set_meta("extractor_version", …)` overwrites the store's version
marker on every run rather than comparing it. Resume therefore has no way to notice
that the code writing into a store is not the code that wrote the rest of it.

That is what happened. A store ingested to 63% by the pre-P10 extractor was resumed
to completion by the P10 extractor. Both runs succeeded, the checkpoint was
consistent, and the resulting store passed every structural check.

The two generations are not a version drift. They behave differently in the exact
dimension P10 changed:

| | rows | `scope` | `source_role` split |
|---|---:|---|---|
| written 2026-08-14 (pre-P10) | 4,843 | all `NULL` | user 93.3% / assistant 6.7% |
| written 2026-08-15 (P10) | 2,265 | populated | user 44.8% / assistant 54.4% / system 19 rows |

The pre-P10 extractor filed 93.3% of everything as `user`, which is the schema
defect P10 fixed: `subject` was derived from a string prefix and so acted as a
two-valued speaker flag with nowhere to put a third-party fact.

**The 68% / 32% split is a proportion of memory rows — 4,843 and 2,265 of 7,108 —
not of sessions.** Session-level attribution is not recoverable from the store,
because a session ingested by one generation can be superseded or deduplicated
against memories written by the other.

The store is complete and internally consistent: 2,348 sessions, 50 namespaces,
24,590 turns, 7,108 memories. It is simply not the product of one system version,
which is the property a formal result needs.

### 10.2 Diagnostic A2 — `diagnostic only, excluded from formal benchmark claims`

Run on the mixed store rather than discarded, because it costs nothing from the
extractor's quota pool (the answerer and judge draw on separate pools) and because
a measurement of a known-heterogeneous system is still worth having on record.

`results/raw/two_stage_hydrated.a2-mixed-store.jsonl`

| | |
|---|---|
| questions | 50 / 50, matching the frozen `dev50` manifest exactly (0 missing, 0 extra) |
| accuracy | 60.0% (30/50) |
| source-session recall | 94.0% |
| median context | 1,346 tokens |
| p95 answer latency | 42.2 s |
| answer prompt | `memory-aware-v2` |
| judge prompt | `lme-type-aware-v2` |
| store fingerprint | `two-stage-p10-v2@7108`, single-valued across all 50 rows |

**The fingerprint proves less than its name suggests.** A single value across 50
rows establishes that every answer came from one store *state* — which is what it
was built for, and which it did. It cannot establish that the store is internally
homogeneous, because it is derived from the store's `meta` marker, and that marker
records only the most recent writer. For this run the honest identifier is the
filename, `a2-mixed-store`, not the fingerprint.

This number does not belong beside the baselines as a headline. It is not carried
into the README, and the formal `dev50` result remains unrun.

**Category split (mixed-store diagnostic observation only).**
`single-session-assistant` scored 1/6. That is the category the P10 `source_role`
fix targets, and 68% of the rows in this store predate the fix — so the result is
consistent with the fix not being exercised, and is equally consistent with several
other explanations. **It cannot be used to judge whether P10 works.** The clean
rebuild is what will answer that.

**Paired against the earlier pilot** (the 31 pilot questions are a strict subset of
`dev50`): 16/31 → 18/31, two questions flipped wrong→right, none the other way,
exact McNemar p = 0.500. Completing the last 37% of the ingest produced **no
detectable change** on those questions.

### 10.3 Three evaluation-integrity failures found in one run

**A completion invariant compared the wrong two numbers.** The preflight checked
`COUNT(DISTINCT session_id)` against 2,400 and stopped at 2,348. Both numbers are
correct and they measure different things: the corpus has 2,400 session *entries*
but 2,348 unique session *IDs* — 51 IDs recur, for 52 extra copies, because
LongMemEval-S uses one session as evidence for more than one question. The pipeline
counts entries; the store deduplicates by ID.

**The failure message asserted a cause that had not been checked, and is
withdrawn.** It read `2348/2400 sessions — quota probably ran out`. The ingest had
run all 69 of 69 batches and reported 2,400 sessions with no quota stop. Nothing
about quota was verified before that sentence was written. It was wrong, and it is
retracted here rather than quietly corrected: an error message that guesses a cause
is worse than one that reports only what it observed, because it directs the next
person away from the real problem.

**Ingestion resume has no guard on what produced the store.** This is the same
defect class that was fixed for evaluation resume the same day — an artifact
resumed across a change in one of its inputs — caught there before it could do
damage, and already realised here.

The fix is not a version-string comparison. `extractor_version` is edited by hand,
so it catches a deliberate generation change and misses the likelier drift: a
prompt reworded, a schema column added, or a batch size changed with the version
left alone, which produces the same two-systems-in-one-store with nothing to notice
it by. Resume now compares a fingerprint computed from the inputs themselves —
Stage A and Stage B prompt text, `schema.sql`, the model id, sessions per request,
the dedup threshold — and names which component moved rather than reporting that a
hash differs:

```
this store was written by a different ingestion setup:
  model: m -> m2
  sessions_per_request: 2 -> 4
```

`--fresh` remains the way to change extractor, because it replaces the data instead
of relabelling it. The guard cannot detect stores that are *already* mixed: their
label was overwritten before it existed. That is what the row-level homogeneity
check below is for.

Two further findings from the same session, both about automation rather than data:

**A gate that prints its verdict is not a gate.** `ingest temporal-gate` printed
`Gate closed` and exited 0. That reads correctly to a human and is invisible to
anything checking a return code, so an unattended sequence would have continued
into the run the gate exists to prevent. `ingest fidelity` had no threshold at all.

**A gate artifact must be proved fresh.** Reading `gate_open: true` from
`temporal-gate.json` is not evidence that this run's gate passed — if the gate
crashed before writing, the file on disk is the previous run's verdict. Freshness is
now checked against the run's own start time. This is the quiet failure mode in
automated evaluation pipelines: stale artifacts do not look like errors.

### 10.4 Resolution

1. **The mixed store is kept**, unmodified, as the diagnostic record. Its A2 result
   keeps the `a2-mixed-store` label permanently.
2. **A clean store is being built from zero** under a separate name
   (`two-stage-p10`), so nothing depends on deleting the evidence. At the time of
   writing it is mid-rebuild; the extractor's daily quota makes this a two-day job
   (~378 requests against a 500/day pool already partly spent).
3. **A homogeneity gate runs before the formal A2**, because `meta` is a claim and
   not evidence — it records the last writer, which is how the mixed store came to
   describe itself as `two-stage-p10-v2`. The fingerprint and the rows are both
   checked, and they do different jobs. The fingerprint establishes that the code
   matches what is running now; for a store first written before the guard existed
   it is stamped on trust, since the resume that stamps it cannot inspect rows
   already on disk. **The row checks are the evidence**: `scope IS NULL` is a
   pre-P10 signature and separates the two real stores cleanly — 4,843 such rows in
   the mixed store, 0 in the clean one — so a store that lies in `meta` still fails.
   Also checked: no `source_role` outside {user, assistant, system}, unique sessions
   equal to the corpus-derived target, and 50 namespaces.

   Deliberately *not* checked by `ingested_at` date. A rebuild spanning two days of
   quota is the normal case here, so "all rows share a date" would fail a healthy
   store and pass an unhealthy one that fit inside a day — which is how the mix was
   first spotted, but is not a rule that generalises.
4. **Gates then formal A2.** Both gates ran against the current extractor and
   passed — recorded here because they measure the extractor, not the store, and so
   carry over to the clean build:
   - temporal gate: key consistency 100%, replacement recall 100%, false supersede
     0%, cross-run stability 100% — 4/4, gate open.
   - fidelity: **44.0% (22 of 50 stated values)**, against a historical baseline of
     36.6%. **This is not an improvement.** With n = 50 the interval around 44.0% is
     roughly ±14 points and the baseline sits well inside it; the two are not
     distinguishable. The 0.30 threshold is a catastrophic-regression floor, not an
     acceptance line — it is set far below the baseline precisely so that it fires
     on breakage rather than on noise, and this project has no measured run-to-run
     noise figure for fidelity, only for end-to-end accuracy. The number still
     requires comparison against the baseline by a human.

### 10.5 What the diagnostic run's failures pointed at

A first pass counted `source_session_recalled` and reported 17 of 20 failures as
"retrieved but still wrong". **That measure does not support the conclusion drawn
from it.** A recall flag says a memory from the right session was touched. It does
not say the answer-bearing turn was retrieved, that extraction preserved the exact
fact, or that the fact reached the answerer — and those are three different defects
with three different fixes.

So all twenty were audited by hand to the evidence layer, one at a time, separating
those four questions. `results/failure-analysis/a2-mixed-store-failures.json` records
each case with its verdict and the reason.

| stage | n | what it means |
|---|---:|---|
| **E** | 10 | the fact was never stored, or stored in a form that loses it |
| **M** | 3 | cross-session aggregation |
| **R** | 3 | the exact evidence was in the store and was not retrieved |
| **T** | 3 | temporal arithmetic |
| **A** | 1 | the evidence reached the answerer and was not used |

**Half of all failures are extraction losses.** Three gold sessions produced *zero*
memories. Others stored a shape that cannot answer the question: the coffee-limit
update was stored as "thinking of changing" rather than as a change, so supersession
never fired and the answerer reported the direction backwards; the Hawaii trip was
stored without its duration, and the answerer filled the gap by assuming a number.

The refined conclusion is stronger than the one it replaces, and points the same
way: **retrieval is where the least of the loss is**. Only 3 of 20 are retrieval
misses, and of the thirteen cases whose answer survives in the raw archive, **nine
are already returned by BM25 at rank 1** — the archive can answer them today, and
nothing asked it to, because this run had the fallback off. A better retriever has almost nothing left to win here, while extraction
fidelity has ten cases waiting.

All three retrieval misses are `single-session-assistant`, and one of them is
the Mayo question, which the demo answers correctly. A system cannot
give two answers to one question, so the discrepancy was worth chasing, and it was
not about the store:

```python
raw_fallback = cfg.fallback.enabled  # from the config, not from the variant name
```

`baselines.yaml` leaves `fallback.enabled` at its default of `False`. The variant
named `two_stage_fallback` does not turn the fallback on by itself; only the config
does, and the live regression used `configs/fallback.yaml`. So the diagnostic A2
measured the system **without** the conditional raw-conversation fallback — the
feature that exists precisely for those three cases.

The formal run is therefore two arms over the same store, manifest, answerer and
judge. The configs differ in `fallback.*` and nothing else — `top_k` 20, rerank off
in both — so the delta between them is what the fallback is worth, measured rather
than demonstrated:

| arm | config | artifact |
|---|---|---|
| memory only | `baselines.yaml` | `two_stage_hydrated.a2-clean-p10.jsonl` |
| memory + conditional fallback | `fallback.yaml` | `two_stage_hydrated.a2-clean-p10-fallback.jsonl` |

The first stays comparable with every earlier number; the second describes what is
actually shipped. Neither is the headline until both have run.

### 10.6 The clean P10 result

The rebuild completed on 2026-08-17: 2,348/2,348 unique sessions, 50/50
namespaces, 6,233 memories, 24,590 raw turns, vector index 6,233/6,233. Zero rows
with `scope NULL` — the pre-P10 signature that stands at 4,843 in the mixed store.
Both quality gates passed before the evaluation ran: the temporal gate on all four
metrics (key consistency 100%, replacement recall 100%, false supersede 0%,
cross-run stability 100%), and extraction fidelity at 44.0% against a 30.0% floor.

The formal A2 is two arms over the frozen `dev50` manifest, differing only in
`fallback.*`:

| | mixed store (diagnostic) | clean P10, memory only | clean P10, product |
|---|---|---|---|
| rows by extractor generation | 4,843 pre-P10 / 2,265 P10 | 100% P10 | 100% P10 |
| raw fallback | on | off | on |
| dev50 accuracy | 60.0% | 56.0% | **72.0%** |
| source-session recall | 94.0% | 94.0% | 94.0% |
| `single-session-assistant` | 1/6 | 2/6 | 6/6 |
| `multi-session` | — | 5/13 | 7/13 |
| median context | 1,346 tokens | 1,415 | 1,455 |
| p95 answer latency | — | 7.2s | 1.7s |

Paired over the same 50 questions the fallback fixed 9 and broke 1 — exact McNemar
**p = 0.022**.

That number has a history, and reporting it without one would overstate it. The
first product arm measured **66.0%**, at 6W-1L and **p = 0.125** — not significant.
Reading one of its failures found the truncation defect in section 10.7; fixing it
moved the arm to 72.0% and the pairing to significance, with the re-run beating the
old arm 3W-0L. The defect was real and its fix is mechanical, but both the fix and
its p-value come from the same 50 questions, which makes this an adaptive result.
It means "this survived one honest look at its own failures", not "this
generalises". The held-out set has since answered the second question: **70.0%,
a 2.0pp drop** (section 11). It has not answered it for this p-value, which was
never re-tested on unseen questions — and 9W-1L becomes p = 0.109 at 8W-2L, which
is one flipped question away.

Two comparisons here are *not* clean experiments and should not be read as one.
The mixed→clean columns differ in more than the extractor generation: the clean
store also re-extracts the 63% the pre-P10 run produced, so supersession chains
and deduplication differ. Read it as "the same pipeline rebuilt under one
extractor", not as an isolated measurement of the P10 fix. Fidelity 44.0% against
the earlier 36.6% is numerically higher on a sample too small to call an
improvement.

What the P10 fix was for does show up where it was predicted:
`single-session-assistant` moved 1/6 → 2/6 on memory alone, and to 5/6 once the
archive is reachable.

### 10.7 What the clean store broke, and the diagnosis that was wrong

After the rebuild, the Mayo question — the case the README opens with, and the one
the live regression demonstrated end to end — failed on the clean store in **both**
formal arms. A rebuild that improved retrieval broke the question that recovery
exists for. Two explanations were available and the convincing one was wrong, which
is the part worth recording.

**The explanation that was wrong.** `recover()` branched on whether retrieval had
returned anything:

```python
if memories:                                    # ← the entire condition
    turns = self.store.turns_for_memories(memories)[: self.max_turns]
    if turns:
        return RawEvidence(turns=turns, level="source_local", ...)
turns = self.store.search_turns(user_id, query, limit=self.max_turns)  # archive_wide
```

Nothing checked that the memories were *about* the question, so a confident wrong
retrieval could suppress the archive search entirely. The docstring even stated the
assumption — *"a memory that was found but lacks a detail points at the turn holding
it, which is strictly better evidence than a keyword search over everything"* — true
exactly when the retrieved memories are on topic, and unverified. This is a genuine
defect and it is fixed. **It is not what broke Mayo.**

**What actually broke it.** Retrieval had found the right conversation. At the
product's `top_k=20`, the memories retrieved for the Mayo question included one
anchored in `answer_sharegpt_81riySf_0` — the gold session — so the level chosen was
`source_local`, and correctly. `turns_for_memories` returned **16** candidate turns
ordered by session id, the gold turn sat at **index 9**, and `[:max_turns]` kept the
first three: all from `33a39aa7_1`, a conversation about live music. The right level,
the right conversation, and the answer discarded by an alphabetical slice.

The two defects hid each other. On the mixed store this question retrieved nothing,
fell through to the archive, and never met the slice — which is why a truncation bug
in the most-demonstrated path in the project went unseen until recall improved
enough to produce more than three candidates.

Both are one omission: nothing ranked the candidates against the question. The fix
ranks them, with a single FTS query serving twice — it supplies the archive-wide
candidates, and its top hit decides whether the memories found the right
conversation. The decision is per *session* rather than per turn, because inside one
conversation BM25 routinely prefers the user's question to the assistant's answer
(the question repeats the query's words); judged per turn that reads as "the archive
beat the memories" and abandons a memory that had found exactly the right place.

Neither defect is the `omelette` case (gold at BM25 rank 6 against `max_turns=3`),
which is genuinely about depth. Depth remains untouched and unmeasured.

**Effect.** `single-session-assistant` 5/6 → 6/6, `multi-session` 5/13 → 7/13, and
the arm 66.0% → 72.0% (3W-0L against the pre-fix arm). The inspector demos now run
on the clean store, and `scripts/record_golden.py` re-records them — a step that had
been manual, which is why both recordings had gone stale unnoticed.

**A note on what the demo recording revealed.** Requiring three runs to agree
surfaced something a single run hides: the first-pass answerer is not deterministic
between `need_source` and `no_evidence`, and the two route to different levels,
because `need_source` passes the retrieved memories to `recover()` and `no_evidence`
passes none. Both now recover the URL. The recording says so rather than presenting
one run's route as the only one.

---

## 11. Current limitations

1. **The held-out result exists, and it has no baselines.** `heldout100` ran once
   on 2026-08-19 behind five gates: **70.0% final, 50.0% from structured memory
   alone, 20.0% rescued by the archive**, against dev50's 72.0 / 54.0 / 18.0. The
   headline generalises at -2.0pp. `full_context` and `naive_rag` were not run on it,
   so no unseen comparison against any baseline exists, and "structured memory ties
   naive RAG" remains a dev-50 claim. The set is now spent and nothing may be tuned
   against it; `dev100` was frozen beforehand for further work.
2. **dev50's per-type numbers carried no information.** Each cell held 3 to 13
   questions. On unseen data its three 100% categories all fell and its two worst both
   rose, none of it distinguishable (Fisher p ≥ 0.12 everywhere). `temporal-reasoning`
   is **59.3%**, not the 46.2% wall this report described; `knowledge-update` fell from
   100% to **66.7%** and is now joint-weakest, a failure mode dev50 could not show.
3. **The answerer disagrees with itself and this was never measured.** Re-running the
   exact production configuration on five held-out questions did not reproduce three
   of their answers, including a hallucination that never recurred in nine attempts
   ([context-arms.md](../results/context-arms.md)). Every paired result in this report
   is a single run. A repeat protocol is pre-committed in
   [heldout-variance.md](../results/heldout-variance.md).
4. **The pilot is 31 unstratified questions.** An ingest-order prefix, not a sample.
   Category splits are descriptive only.
5. **The 72.0% is adaptive to dev50.** It was reached after fixing a defect found
   by reading one of the set's own failures (section 10.7), so its `p = 0.022` says
   the result survived one honest look, not that it generalises. The mixed store —
   4,843 pre-P10 rows and 2,265 P10 (section 10) — is no longer the default for
   anything, and results measured on it stay labelled diagnostic.
4. **The fallback's candidate ranking is new and measured once.** Section 10.7
   replaced a blind level choice and an alphabetical truncation with a single
   ranking. It is covered by unit tests and by one dev50 arm; `fallback.max_turns`
   itself has still never been swept, and the `omelette` case (gold at BM25 rank 6)
   says depth is a separate open question.
5. **14.5% of substantive sessions yield no memory** — 304 of 2,096 in the clean
   store, against 18.8% in the mixed one. Recorded as a baseline and deliberately
   not gated: this project has no evidence for where a healthy rate sits, and
   whether the cause is stochastic dropout or a systematic gap in the extraction
   policy has not been measured.
6. **A prompt change broke comparability.** `ANSWER_SYSTEM` was rewritten on
   2026-08-14; every number measured before it came from a different answerer. The
   trade was deliberate — the old prompt was measurably wrong — and is recorded
   rather than smoothed over.
7. **Single-writer.** SQLite. Ingestion and evaluation now take a cross-process lock
   (`llm_long_term_memory/locking.py`) after two simultaneous ingests were observed
   overwriting each other's quota accounting — 211 requests recorded against ~320
   made, with nothing wrong-looking in the store. The lock is advisory and
   pid-based; it stops a second run from another shell, which is the case that has
   actually occurred.
8. **No cost accounting.** Free tier, no verified price schedule.

---

## 12. Future work

**Done since the last revision:** the clean P10 rebuild, both gates, the formal
`dev50` A2 in two arms, and the mixed-vs-clean comparison (section 10.6). The
extractor-version guard on ingestion resume — the defect that made section 10
necessary — is in `pipeline.py` and held throughout the resume that completed the
rebuild. The fallback's candidate ranking (section 10.7) replaced a blind level
choice and an alphabetical truncation, taking the product arm to 72.0% and moving
the demos onto the clean store.

**Immediate, and in this order:**

1. **Explain the 14.5% zero-yield rate.** Re-run the 304 sessions that produced
   nothing under an unchanged prompt, model and configuration, and measure how many
   yield on a second and third attempt. That separates stochastic dropout from a
   systematic gap in the extraction policy, and the answer decides whether the
   remedy is a retry or a prompt change. Doing it the other way round — editing the
   prompt first — would leave both explanations still open.
2. **Exact-detail fidelity.** The recurring failure is a memory that keeps the gist
   and drops the identifier: the Mayo URL, `Garmin Forerunner`, `2-3 eggs`. Worth
   testing whether identity-bearing spans should survive extraction verbatim rather
   than being paraphrased into prose.

**Then, in order of evidence behind them:**

- **A regression corpus, grown not generated.** `results/raw-retrieval-regressions.json`
  holds 11 cases today. It exists to be run *before* adopting a retriever change —
  the cross-encoder result is exactly what such a corpus is for. Retrieval-solvable
  and reasoning-required cases are tagged separately, because only the former belong
  in a BM25-vs-dense comparison. Revisit dense at ≥50 validated retrieval-solvable
  cases with a clear Recall@3 gain.
- **Conditional dense retrieval, if any.** If a gap appears, the shape suggested by
  the evidence is a cascade — source-local, then BM25, then dense only when BM25
  scores poorly — rather than paying dense on every query. Related work points the
  same way: AgentIR and SelRoute route by query or by sparse-retrieval confidence
  rather than running one retriever for everything. *(Not independently verified for
  this report.)*
- **Temporal and aggregation layers**, which sections 8.3 identifies as the largest
  class of remaining failures that no retriever change can address.
- **Multimodal memory** as a demo capability, with the caveat that the caption-then-
  index path leaves the downstream pipeline unchanged and the schema work is the
  image anchor replacing character spans.

---

## Appendix — provenance of every headline number

| Claim | Source | Recomputed |
|---|---|---|
| 51.6% / 242 tokens, 51.6% / 13,057 tokens | `results/raw/two_stage.k10_plain.jsonl`, `naive_rag.jsonl` | yes, 2026-08-15 |
| 11W-2L p=0.022; 2W-1L p=1.000 | `results/a2-pilot.md`, `lltm eval compare` | yes |
| Reranker: identical answers at k=10, recall 87.1→83.9 | `results/rerank-pareto.md`, `two_stage.k10_{plain,ce}.jsonl` | yes |
| BM25 R@1 93.5/54.8/6.9 by query type | `results/raw-recall-diagnostic.md` | yes |
| 5/5 Mayo paraphrases at rank 1 | `results/raw-retrieval-regressions.json` | yes |
| 326 assistant / 4,517 user memories | `stores/two-stage-hydrated.db`, rows written 2026-08-14 (pre-P10 generation only; the store now totals 1,558 / 5,531) | yes, 2026-08-15 |
| 7,108/7,108 provenance anchors (session id and char span) | `stores/two-stage-hydrated.db` | yes, 2026-08-15 |
| Mixed store: 4,843 pre-P10 / 2,265 P10 rows | `stores/two-stage-hydrated.db`, grouped by `date(ingested_at)` | yes, 2026-08-15 |
| Diagnostic A2: 60.0% (30/50), recall 94.0%, 1,346 tokens, p95 42.2s | `results/raw/two_stage_hydrated.a2-mixed-store.jsonl` | yes, 2026-08-15 — **diagnostic, not a benchmark claim** |
| Pilot→full-store pairing: 16/31 → 18/31, p=0.500 | `two_stage_hydrated.jsonl` vs `.a2-mixed-store.jsonl`, exact McNemar | yes, 2026-08-15 |
| **Formal A2 base: 56.0% (28/50), recall 94.0%, 1,415 tokens, p95 7.2s** | `results/raw/two_stage_hydrated.a2-clean-p10.jsonl`, dev50 manifest, audited row-by-row | yes, 2026-08-17 |
| Formal A2 product, pre-fix: 66.0% (33/50) | `results/raw/two_stage_hydrated.a2-clean-p10-fallback.jsonl` | yes, 2026-08-17 — **superseded by the row below** |
| **Formal A2 product: 72.0% (36/50), recall 94.0%, 1,455 tokens, p95 1.7s** | `results/raw/two_stage_hydrated.a2-clean-p10-fallback-v2.jsonl`, audited row-by-row | yes, 2026-08-17 |
| **Fallback 9W-1L, exact McNemar p=0.022** | base vs product-v2, paired on question id | yes, 2026-08-17 — **adaptive, see 10.6** |
| Fix vs pre-fix arm: 3W-0L, p=0.250 | the two product files, paired | yes, 2026-08-17 |
| Clean store: 2,348/2,348 sessions, 50 namespaces, 6,233 memories, 0 `scope NULL` | `stores/two-stage-p10.db` | yes, 2026-08-17 |
| Zero-yield 304/2,096 (14.5%) substantive sessions | `SQLiteMemoryStore.zero_yield_sessions()` over `two-stage-p10.db` | yes, 2026-08-17 |
| Mayo: gold turn at index 9 of 16 source-local candidates, sliced off by `[:3]` | `turns_for_memories` over the 20 memories retrieved for the question | yes, 2026-08-17 |
| Fidelity 44.0% (22/50 stated values) vs 36.6% baseline | `results/raw/fidelity.json` | yes, 2026-08-15 — **not distinguishable at n=50** |
| Temporal gate 100/100/0/100, 4/4 pass | `results/raw/temporal-gate.json` | yes, 2026-08-15 |
| Corpus: 2,400 session entries, 2,348 unique ids | LongMemEval-S via `lme.load`, `Counter` over session ids | yes, 2026-08-15 |
| Warm-up 12,486ms → 181ms | live measurement, 2026-08-14 | single run |
| 26.0% vs 54.0%; abstention 100% vs 50%; repeat 48.0–54.0% | `results/table.md` (frozen v1, stratified 50) | not re-run |
| Live regression 6/7, 3/3 consistent | `results/raw/two_stage_fallback.live_v2_run{1,2,3}.jsonl` | yes, 2026-08-14 |
| Utility predictor RMSE 0.310 vs 0.263 | `results/p6-pilot.md` | not re-run |
| Rule-based Stage B: 43% predicate accuracy | `docs/DECISIONS.md` | not re-run |

External results (LoCoMo Hit@1 figures, MemMachine's architecture) are cited from the
papers named inline and were **not** reproduced here.
