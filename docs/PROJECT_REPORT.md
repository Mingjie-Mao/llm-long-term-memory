[![English](badges/lang-en-active.svg)](PROJECT_REPORT.md)[![中文](badges/lang-zh-idle.svg)](PROJECT_REPORT.zh-CN.md)

# Project report: why it is built this way, and what the experiments settled

The [README](../README.md) is the thirty-second version. This is the ten-to-twenty-minute
one: **what problem it solves, why the design looks like this, what was measured, and what
is still missing.**

The implementation as it stands, with interface and operational semantics, is in
[ARCHITECTURE.md](ARCHITECTURE.md). Splits, statistical conventions and every final number
are in [EVALUATION.md](EVALUATION.md).

## Contents

1. [What the project does](#1-what-the-project-does)
2. [Why long-term memory is needed](#2-why-long-term-memory-is-needed)
3. [System architecture](#3-system-architecture)
4. [Writing: fact extraction](#4-writing-fact-extraction)
5. [Temporal state](#5-temporal-state)
6. [Retrieval and context](#6-retrieval-and-context)
7. [Raw-source recovery](#7-raw-source-recovery)
8. [Results](#8-results)
9. [Failure analysis](#9-failure-analysis)
10. [Limitations](#10-limitations)
11. [What comes next](#11-what-comes-next)

---

## 1. What the project does

There are three common ways for an LLM application to remember a user across sessions.

**Plain LLM: put the whole history in the prompt.**

```
Entire history → Context → Answer
```

Accurate, because nothing was dropped. The cost grows linearly with the history: in this
project's final test that route spent 109,059 tokens per question. It also hands the model
the entire judgment of *which value is current* — old and new values of the same attribute
arrive together and the model has to guess between them.

**Plain RAG: retrieve passages.**

```
History → Chunk → Embed → Top-K → Context → Answer
```

Cheap, but what it returns is **the passage that most resembles the question**, not **the
fact that currently holds**. A user who said "I live in Canberra" three years ago and
"moved to Sydney" last year has two passages that both look relevant, and nothing in the
index says which one still counts.

**This project (LLTM): turn the history into facts first, then retrieve facts.**

```
Conversation history
  ↓
Fact extraction        one self-contained sentence + subject + attribute slot + value + time
  ↓
Temporal resolution    repeated values of one attribute become a timeline, not a contradiction
  ↓
Long-term store        SQLite (facts, state, raw turns) + vector index
  ↓
Retrieval per question only what this question needs; the window is never filled
  ↓
Raw fallback           when extraction dropped a number, date or exact wording
  ↓
Answer context
```

Four problems, specifically.

**One: what to keep.** Not conversations — **facts**: who the user is, what they prefer,
what happened, what state they are in now. Extraction *is* the compression, done once at
write time rather than improvised at answer time.

**Two: what happens when a fact changes.** Canberra → Sydney → Melbourne is not three
contradictory records, it is **one timeline**. A new fact does not delete the old one; it
closes the old one's validity interval. The old value stays in the store and simply does
not enter the context by default, so "where did I live then" is a query rather than a loss.

**Three: how to get it back.** Not by filling the window, but by retrieving the few
memories this question needs. In the final test that is 574 tokens against 109,059.

**Four: what to do when structured memory drops a detail.** Extraction is lossy, and that
is this system's largest source of failure. So the original turns are **kept verbatim and
stay searchable**: memory carries state and indexing, raw conversation carries exact
evidence. The answerer goes back to the source only when it reports that the specific it
needs is not in the memories it was given.

### What it is not

| | |
|---|---|
| Not a chatbot | It does not hold conversations. `/v1/answer` exists to evaluate whether the memory is useful |
| Not plain vector RAG | The unit of retrieval is a fact and its state, not a text chunk; after a hit, the temporal layer still decides whether it counts |
| Not an agent | No planning, no tool calls, no loop |
| Not working memory | No task state, no record of tools called, no step pointer |
| Not short-term memory | The current turn is whatever the caller puts in the prompt; this project does not model it |

**It is long-term memory infrastructure that something else calls.** The interfaces are
REST and MCP, so to a caller it is a tool rather than a framework.

---

## 2. Why long-term memory is needed

A larger context window does not remove the problem, and the reason is not capacity. It is
**three things that have nothing to do with length**.

**First, context is billed per call.** A window that holds a hundred thousand tokens is not
a reason to pay a hundred thousand tokens per question. Full history spent 109,059 tokens
per question in the final test, when what the question needed was usually a sentence or two.

**Second, long does not mean used correctly.** Handing the model old and new values of the
same attribute together asks it to decide at inference time which one still holds. Full
history scored **23.1%** on temporal questions in the dev set — it saw everything, and then
chose wrong. Information being in the context does not make the conclusion right.

**Third, session boundaries are real.** A user's history is not one long continuous text.
It is a pile of fragments, each with its own date and each possibly obsolete. Answering
"where do I live now" is not a matter of finding the most relevant passage; it is a matter
of knowing **which fact still holds**.

Long-term memory moves all three to write time, done once, instead of redoing them on every
answer.

### Three kinds of memory; only one is here

The usual split is working memory (task state: what was looked up, which tools ran, which
step we are on), short-term memory (the current conversation), and long-term memory
(preferences, facts and events across sessions).

**This project has only the third.**

| | What this project does | Status |
|---|---|---|
| Working memory | None. No task state, tool-call log or step pointer exists in the code | Entire layer absent |
| Short-term memory | Not modelled. The current turn is whatever the caller puts in the prompt | Out of scope |
| Long-term memory | All of the work: extraction, storage, timelines, retrieval, assembly | Built and measured |

One confusion worth clearing up: **a structured memory store is not working memory.** It
holds "who the user is, what they said, when it changed", not "how far this task has got".
Their lifetimes are opposites — working memory should be thrown away when the task ends;
long-term memory only starts being useful then. So working memory does not belong on this
index; it wants disposable, task-scoped, short-lived storage.

Inside long-term memory there are two label axes: five types (`semantic`, `episodic`,
`preference`, `procedural`, `profile`) and seven scopes (`profile`, `preference`, `plan`,
`recommendation`, `commitment`, `event`, `shared_context`). Scope is assigned by the
extractor and **has never been human-verified**, so it can filter but it cannot support a
conclusion.

### Boundaries

- **This is a prototype, not a product.** Writes serialise within one process, the lock is
  held across model calls, and SQLite and the vector index are two files that persist
  separately. Multi-process concurrency and cross-resource atomicity are unfinished, and
  both want a different store rather than a patch.
- **No SOTA claims.** The comparisons are against this project's own baselines — full
  history and plain retrieval — not against a leaderboard.
- **One benchmark, used up.** LongMemEval-S's 500 questions are partitioned across five
  disjoint sets, all spent, which is why v4 still has no final test. That is a stated
  blocking condition, not a footnote.

LongMemEval was chosen over the more frequently cited LoCoMo because of gold and judge
reliability: LoCoMo's annotation problems are widely discussed, and this project has no
annotation budget to re-check a gold standard it does not trust question by question. The
choice costs something too — LongMemEval's rubric does **not name the required fact in
words of its own**, so the fact-level diagnostic layer in section 9 cannot be built on this
data. Choosing a benchmark is choosing what can and cannot be measured.

---

## 3. System architecture

```
Write                                   Read
─────                                   ────
Conversation                            Question
 ↓                                       ↓
Two-stage extraction (A: sentences,     Hybrid recall (semantic ∪ BM25)
 B: structure)                           ↓
 ↓                                      Namespace and status filtering
Dedup adjudication                       ↓
 (DUPLICATE / UPDATE / DISTINCT)        Context assembly (≤30 memories, ≤3 sessions)
 ↓                                       ↓
SQLite + vector index  ←────────────→   Structured verdict
 ↓                                       ↓
Temporal rebuild                        answer ─────────────→ return
 ↓                                      need_source / no_evidence
Checkpoint                                  ↓
                                        Raw fallback (BM25 over original turns)
                                            ↓
                                        Second answer pass
```

Both paths share one store with separated responsibilities: writing turns conversations
into **facts that carry state**, reading turns a question into **the few facts that
currently hold**. The double-headed arrow in the middle is the raw text — it is both a
product of writing and the fallback when reading fails.

Three entry points connect to the same components: batch ingestion (`lltm ingest run`), the
REST/MCP service (`MemoryService`), and the LLM-free demo backend. Per-figure captions,
code mappings, storage fields and the real on/off state of every optional branch are in
[ARCHITECTURE.md](ARCHITECTURE.md).

---

## 4. Writing: fact extraction

The extractor (`TwoStageExtractor`) splits one extraction into two steps: stage A asks only
for sentences, stage B does the structuring.

| Field | From | Notes |
|---|---|---|
| `content` | Stage A | One self-contained sentence, pronouns resolved, numbers, dates and names verbatim |
| `source_role` | Stage A | Who said it: user / assistant / system |
| `subject` | Stage A | Who the fact is *about* — not who spoke |
| `scope` | Stage A | One of the seven scopes |
| `temporal_key` | Stage B | The attribute slot this fact occupies |
| `update_op` | Stage B | `coexists` / `replaces` / `none` |
| `object` | Stage B | The value in that slot |
| type, entities, importance, event time | Rules | Fields where the model has no advantage do not cost a call |

**The split was decided by measurement.** A single stage asks the model to emit nine fields
per fact, and **the output cost per fact suppresses how many facts it is willing to write**.
Once split, stage A writes only sentences, the session index becomes structural rather than
a field the model has to get right, and the `dropped_bad_index` class of failure disappears.

The cost is that a batch with facts in it normally takes two requests (stage A + stage B),
and deduplication may add more. Stage B is skipped when stage A found nothing.

**Assistant turns are extracted too.** Both what the user said and what the model answered
go through extraction, because "what did the assistant recommend" is frequently what gets
asked about later. The price is that a substantial share of memories begin "The assistant
recommended…", which becomes noise on counting questions.

### Structured memory and raw text, both kept

Extracted facts and **verbatim original turns** are stored together, with different jobs:

| | Responsibility |
|---|---|
| Structured memory | Compression, ranking, state updates. This is the main retrieval path |
| Raw turns | Recovering the link, number or wording extraction dropped. This is the fallback material |

Raw text is not a write-only log: `turns_fts` makes it keyword-searchable. That matters —
**extraction is lossy, and the backstop for that loss is not "extract better", it is "the
original is still there"**.

The storage shape is deliberately conservative: SQLite holds source data, state, the lexical
index and auxiliary relations; vectors are an `N × 384` normalised matrix persisted
separately, with the row-to-memory-id mapping maintained by the application. No external
vector database and no ANN index — exact search is fast enough at this size, and an
approximate index would introduce an unexplainable variable during diagnosis.

**There is no tiered compression here.** Nothing "summarises three years ago into a
paragraph and last month into a line". The design is different: **extraction is the
compression**, done once at write time, and the original turns stay verbatim and are never
deleted.

---

## 5. Temporal state

This is the layer the project does best, implemented in the temporal resolver
(`TemporalResolver`).

**Timelines are rebuilt, not patched.** The intuitive approach is to compare a new fact
against the current head and retire the loser, but that breaks the moment ingestion order
differs from event order — which happens immediately in a batch pipeline: a January fact
arriving after an August one would make January current again. So the resolver reads
**every** memory under a key, including retired ones, sorts by event time, and rewrites all
the intervals that timeline implies. The result is **idempotent and independent of arrival
order**.

**Restating something is not a change.** Saying "I use PyTorch" in March and again in August
moves nothing: consecutive mentions of the same value collapse into one interval owned by
the **earliest** of them, so "when did you switch" answers March.

**This layer makes no model call.** Sorting by event time is arithmetic over data the
extractor already attached. Putting a model call on this path would charge once per memory
for a decision that needs no model. Facts without a date are reported and left alone rather
than guessed at.

**Old values are not deleted.** The design writes validity intervals (`valid_from` /
`valid_to`), not deletions: old values stay in the store and simply do not enter the context
by default. History has two routes — replay at a point in time, or list the full change
history for a key.

Whether the previous fact is closed is decided by the **next** fact's update operation, not
by "this key had a replacement once, so chain everything": `coexists` allows both to stand,
and `removes` is recorded as a terminating operation.

Deduplication is a separate line: a new memory is compared against near neighbours and the
model adjudicates DUPLICATE / UPDATE / DISTINCT. 0.92 is only the **candidate** threshold,
not "drop it at the threshold". Measured on one ingestion of 22 dense conversations,
adjudication requests ran at roughly 0.7–1.1× the extraction requests, rising with store
density.

---

## 6. Retrieval and context

### Five signals, one of them on

The retriever implements five signals: semantic similarity, BM25, recency decay, importance
and entity overlap. **v2 enables semantic only; the other four weights are 0.**

That is not "unbuilt" — it was **built, measured and rejected** (`train150`, 150 questions,
zero API calls):

| Weighting | Top-3 hit | vs baseline |
|---|---:|---:|
| Semantic 1.0 only (shipped) | **95.3%** | — |
| + recency 0.3 | 95.3% | identical, position by position |
| + importance 0.3 | 90.0% | −5.3 |
| + BM25 0.5 | 89.3% | −6.0 |
| All five on | 87.3% | −8.0 |
| + entity 0.5 | 78.0% | **−17.3** |

**The recency row is not a tie, it is a dead signal.** The half-life is configured at 30
days while the freshest memory in the corpus is 932 days old; of 18,519 memories, **zero**
score `S_recency > 0.01`. That is a configuration defect, not a signal defect — on a corpus
made entirely of three-year-old conversations, any 30-day half-life flattens everything to
zero.

So on "good retrieval should surface what is most worth recalling now, not what is most
similar": **the goal is right, but this project does not currently achieve it by
weighting.** It achieves it by two other things — semantic recall, plus the temporal layer
keeping already-superseded facts out of the context. The weighting route measured as
ineffective on this data; whether it holds on a corpus with a live time span is **untested**.

BM25 is not wasted: it is the retriever for the **raw fallback**. So the worry that proper
nouns get diluted by embeddings lands on the fallback path rather than on main retrieval.

### Context: never filled

The budget knobs are at most 30 memories, at most 3 sessions, window radius 1; evidence
hydration capped at 800 tokens; raw fallback capped at 2,400 characters; output capped at
512 tokens. Median context in the final test is **574 tokens**, against 12,763 for plain
retrieval and 109,059 for full history.

The knapsack packer — greedy on value-per-token, with a redundancy penalty on what is
already selected and floor quotas per type — is **off in v2**, because 30 memories at 574
tokens never come close to the 2,000-token budget.

So the risk of "compression starts hallucinating the moment it loses something" has a
different form here. It is not compression loss, it is **extraction loss**. The backstop is
the next section.

---

## 7. Raw-source recovery

The answerer's first pass returns a structured verdict rather than prose:

| Verdict | Behaviour |
|---|---|
| `answer` | Return directly. No raw search, no second pass |
| `need_source` | Consider both the selected memories' sources and the BM25 ranking over raw turns; choose source-local or archive-wide |
| `no_evidence` | Still search raw turns, but without the selected memories' source anchors — archive-wide |

The recoverer (`RawFallback`) **ranks the raw turns first and truncates afterwards**. When
local provenance exists and the best BM25 hit falls inside it, local turns win; when there
is no BM25 hit but local turns exist, local turns are used; otherwise the archive-wide hit
is used. **The second answering pass runs only if raw text was actually found** — otherwise
the verdict's text is kept, or the system says it does not know.

Default caps: at most 3 turns and 2,400 characters of source.

The path works: the fallback fired on **35.0%** of `test100`, and **18.0%** of outcomes were
correct only after it. On `dev50` the same system scored 56.0% with the fallback off and
**72.0%** with it on.

**It has a structural blind spot, though**: the fallback fires only when the answerer
**notices** something is missing. The case where it does not notice, and answers confidently
from a memory that lost the number, is not recoverable this way. That is why extraction
fidelity is the weakest link, and it is the subject of section 9.

The other line of defence is abstention: the answerer is explicitly required to say it does
not know rather than guess when no fact supports an answer. On `dev50` both memory arms
scored 100% on abstention questions.

---

## 8. Results

Conventions, arm definitions and the index of evidence files are in
[EVALUATION.md](EVALUATION.md). Only the conclusions are here.

### 8.1 v2 final: context efficiency is the clearest outcome

`test100`, 100 questions, one run per arm, one judge:

| Method | Accuracy | Median answer context |
|---|---:|---:|
| v2 memory + conditional fallback | 72% | **574** |
| Full history | **86%** | 109,059 |
| Naive RAG | 65% | 12,763 |

**These three rows have to be read honestly.**

- v2 is 7 points above plain retrieval, but the paired test gives **p = 0.3368**. That is
  not evidence of a stable advantage; it is a failure to measure a difference.
- Full history is 14 points above v2 at **p = 0.0043**. That is a real disadvantage.
- What actually stands is the **context**: 574 against 109,059, two orders of magnitude.
  Even losing 14 points of accuracy, "84% of the accuracy on 0.5% of the context" is a
  meaningful engineering result.

The context ratio needs its conventions stated: memory and RAG are character estimates while
full history uses the provider's input tokens, so "about 190×" is an **approximate context
ratio**, not a billing promise.

### 8.2 Temporal reasoning and state updates: the clearest advantage over baselines

Type slices from the same final test: `knowledge-update` **75.0%**, `temporal-reasoning`
**74.1%**.

For comparison, on `dev50` the full-history baseline scored **23.1%** on temporal questions
and plain retrieval **46.2%**.

The gap comes from the design, and the mechanism is clear: full history hands the model old
and new values of an attribute together and asks it to work out which is current; this
project computes the timeline at write time and sends only the current interval by default.
**A deterministic question is solved with arithmetic instead of being re-judged by a model
on every question.**

### 8.3 Extraction fidelity: the bottleneck is batch size, not extractor design

This is the newest and the most valuable result.

The number attached to this project for a long time was "extraction fidelity is only 36.6%",
and it was treated as a problem with the extractor's design — three prompt rewrites, worked
examples, and an evidence-citing extractor were all spent on that reading.

**The experiment on 2026-09-16 says it is mostly a problem with the operating point.** Same
sixty held-out sessions, same prompts, same model, same ruler; only "how many sessions share
one request" changes:

| Sessions per request | Overall fidelity | Memories/session |
|---:|---:|---:|
| 15 (shipped) | **37.3%** | 1.53 |
| **8** | **64.9%** | 3.72 |
| 5 | 66.4% | 4.97 |
| 3 | 70.9% | 6.38 |
| 1 | 78.4% | 13.10 |

The 37.3% at batch 15 reproduces the previously recorded 36.6% almost exactly, so **the
ruler has not drifted** — that number was reliable, and it is a property of **the batch-15
operating point**.

**15 → 8 is the best-value improvement found so far.** The 134 specifics those sessions
state are asked of every arm, so the comparison is a paired exact McNemar test: 15 → 8 is
**+41 / −4, p = 9.3e-09**, while the very next step, 8 → 5, is **+8 / −6, p = 0.79** —
indistinguishable from noise on the same items.

So 8 is not picked from the visible bend in a curve, **it is picked from where the evidence
runs out**. Choosing an operating point by the shape of a line is choosing it from noise.

The mechanism is clear too: when fifteen sessions share one prompt, the output budget is
thinned to one and a half memories per session, and what gets squeezed out is exactly the
specifics — durations fall from 85.7% to 35.7%, proper nouns from 77.4% to 34.0%. What
survives is each session's topic sentence, and the questions ask about the numbers that
topic sentence left out.

Cost: rebuilding `dev100` at batch 8 takes roughly 4–6 quota days against 2.8 measured at
batch 15. The candidate config `configs/v2b-batch8.yaml` is written and differs from v2 in
exactly one parameter (asserted by a test), and **has not been run at scale**. It changes
the ingest fingerprint, so existing stores must be rebuilt rather than resumed; archived
conclusions are unaffected, because they record real readings at their own operating points.

**A negative result alongside it: requiring verbatim citations does not improve fidelity.**
The evidence-grounded extractor, at the same one-session-per-request setting, scored 64.2% —
14 points below its control, paired **−27 / +8, p = 0.0019**. The cause is that demanding a
citation cut what got written to a third. **On this corpus traceability and recall are
opposed, not complementary.** It still stands as a means of auditability — every memory
being able to point at its source is an independent value — but that is a different goal and
should not be charged to the fidelity account.

---

## 9. Failure analysis

For an answer to be right, it has to survive five layers in sequence, and **every layer
inherits the losses above it**.

```
Raw text → Extraction → Retrieval → Context assembly → Answer use
```

| Layer | What is lost | Measured |
|---|---|---|
| Raw text | Nothing. Stored verbatim | — |
| **Extraction** | Specifics the user actually said were never written down | **62.7%** of specifics lost at batch 15; first loss point for **10 of 14** analysed `dev50` failures |
| Retrieval | The memory is in the store but was not recalled | **0–6%** in the failure taxonomy |
| Context assembly | Recalled but did not reach the final prompt | Budget caps; v2's 574 tokens are far below them, so this layer is not currently the bottleneck |
| Answer use | The evidence was in the context but was not used correctly | Abstained despite having the source: **44–50%** |

Each wrong answer gets exactly one primary cause (E1 extraction loss / E2 retrieval miss /
E3 temporal error / E4 assembly loss / E5 answer reasoning failure / E6 judge error). The
table and the tooling are in [EVALUATION.md](EVALUATION.md).

### Why "hit the source session" is not "the needed fact reached the context"

This is the diagnostic most easily misread, and it deserves its own space.

v2 selected at least one source session on **94%** of questions while scoring **72%**. A more
extreme case: one comparison had **98.3%** source recall at **55.0%** accuracy.

The gap exists because **the two metrics measure different things**:

- "Source-session recall" says only that some selected memory's `source_session_id` points
  at a session labelled as an answer source.
- It does **not** say that memory contains the fact that supports the answer.

A session can run dozens of turns, and extraction writes one or two memories out of it.
**Those one or two can come from exactly the right session and still have dropped the number
the question asks about.** Retrieval brings them back, the metric records a hit, and the
answer is not in the context.

v2's 28 wrong answers break down accordingly: 3 never found a source session, 14 were still
wrong after the raw fallback, and **11 were wrong with source context already in hand**.
Three quarters of the failures happen *after* the source has been found.

So source recall is **necessary but not sufficient**. It is good for ruling retrieval out;
it is not good for claiming the evidence was complete. A stronger metric needs the benchmark
to name the required fact in words of its own, which LongMemEval does not do — so this
project can report the stricter variant for free (**every** labelled source session reached
the context) but cannot report fact-level coverage.

### A defect outside that chain: deduplication was not namespace-scoped

The deduplicator's neighbour search runs on the shared index with **no `user_id` condition**,
while retrieval has one.

Sampling 1,500 candidates on `test100` at zero cost: of the neighbours above threshold,
**3 were same-namespace and 42 were cross-namespace**, and two candidates had an entire
neighbour window made of foreign memories. This was the ordinary case, not a corner.

Every cross-namespace pair is a pair the deduplicator would have adjudicated had both been
in the store at the time, and a DUPLICATE verdict deletes a fact from a conversation that
never contained its supposed duplicate. A window filled with foreign memories also crowds
out the within-namespace duplicates this stage exists to catch.

**Fixed 2026-09-16.** The namespace filter is applied after the search, so `limit` still
means "how many index hits to look at" and the change can only ever **shrink** the window —
it removes adjudications and cannot add a DUPLICATE drop.

Half the fix is not in the dedup code. Changing the comparison scope without recording it
would allow a store built under the old policy to be resumed under the new one, leaving a
store whose rows come from two policies with nothing to tell them apart — which is exactly
what the ingest fingerprint exists to prevent. So `dedup_scope` was added to the
fingerprint, read off the deduplicator object rather than hardcoded. The cost is real and
was accepted: **every store written before the fix now refuses to resume.** Rebuild with
`--fresh`, or use a new store name. Archived conclusions are unaffected; they record real
readings under their own policy.

---

## 10. Limitations

### 10.1 Extraction still drops key facts

Even at batch 1 fidelity is only 78.4% — a fifth of the stated specifics are still never
written down. The shipped configuration is at 37.3%. This is the weakest link in the chain
and the largest category in the failure taxonomy.

The backstop that exists is the raw fallback, but it fires only when the answerer notices
something is missing — **if it does not notice, nothing saves it**.

### 10.2 Critical facts have no priority

A life-critical fact and small talk take exactly the same path. Importance is assigned by
rules, not judged by the model. Nothing currently gives a fact like an allergy a stronger
write guarantee or more resistance to eviction.

A concrete related defect: event time is the **session's** date rather than the fact's own,
so memories from one conversation share a timestamp (3,776 distinct values across 4,714
sessions). When it cannot be resolved it should be left null rather than filled with the
session date.

### 10.3 Decay and eviction are unvalidated on a growing corpus

Decay, access reinforcement and capacity eviction are all implemented and **all off by
default**, and none has been calibrated on a corpus that actually grows. What is known:
recency decay is a dead signal on the current corpus (a half-life configuration problem),
and eviction is free at 400 memories per namespace but costs 8.6 points at 200.

Which means "does the store turn into a junkyard as it grows" **has no measured answer in
this project**. As it stands memories have no time limit and no capacity limit; they leave
only by explicit deletion or by being superseded on a timeline.

### 10.4 There is no working-memory layer

The layer is entirely absent. If the goal is "remember the user", the current split is
enough; if the goal is to build an agent, working memory is a whole missing layer — and it
does not belong on this index.

Two further known engineering boundaries: there is no cross-process concurrency guarantee
(the in-process lock is bypassed by a multi-process deployment), and deletion is not atomic
across resources (SQLite rows go first, vectors second, so a crash in between leaves orphan
vectors rather than retrievable ghost memories — the safe direction, but not atomic).

---

## 11. What comes next

Ordered by whether existing evidence can state the cost and the benefit.

### 11.1 Raise write fidelity

**The first move is changing sessions per request from 15 to 8.** Both sides are measured:
2–3 quota days for 27.6 points, with the candidate config written and test-asserted. It has
not been executed because it changes the ingest fingerprint and existing stores can only be
rebuilt — and rebuilding changes none of the archived conclusions.

The second is separating event time from observed time. A zero-call change, justified by the
4,714 / 3,776 figure above.

The third is wiring the zero-call coverage check into ingestion: after extraction, compare
the specifics in the raw text by regex and spend a repair call only where something is
missing. The function exists and is not wired up.

**What is explicitly not on the list is another extraction-schema change.** The
evidence-citing extractor already falsified the direction "stricter structure raises
fidelity" — it cut output to a third and the net effect was negative.

### 11.2 Finish the layered evidence diagnostic

What can be reported at zero cost today is two layers: "at least one source session reached
the context" and "every source session did". The third — **whether the fact the question
needs reached the context** — requires the benchmark to name the required fact in words.

Until that is done, the 94%-against-72% gap in section 9 can only be described
qualitatively, not attributed quantitatively. The path is clear: it needs data with a
fact-level rubric, not more analysis.

### 11.3 Validate retrieval and forgetting on a longer, denser history

Every retrieval and forgetting conclusion here rests on a corpus made **entirely of
three-year-old conversations**. Two properties of that corpus directly determine two
conclusions: recency decay is a dead signal (because nothing is fresh), and eviction is free
at 400 (because the store is not big enough).

**Neither conclusion generalises.** What needs testing is the same machinery on a history
with a real time span that keeps growing: whether recency becomes a live signal, where the
free eviction threshold lands, and whether the cross-namespace dedup exposure gets worse or
gets diluted as the store grows.

That needs new data rather than new code, and it needs its own pre-registration — it is a
new question, not a re-run of an existing metric.

### Explicitly not doing

Written down so that "not done" is a decision rather than an omission.

- **Not adding a working-memory layer on this index.** Its lifetime, write frequency and
  retrieval pattern are the opposite of long-term memory's. Build it separately when there
  is actually an agent; the interface is writing the conclusion to `POST /v1/messages` when
  the task ends.
- **Not swapping the store to solve single-writer and dual-resource problems.** Postgres +
  pgvector solves both at once, but that is a rewrite rather than a patch — the
  prototype/product line. This project's current goal is not on that side.
- **Not looking for another candidate that beats full history.** v2 loses by 14 points at
  p = 0.0043 on the frozen final test, while the measured noise floor is 38 flipped verdicts
  out of 142 probes between two runs of the same configuration. At that signal-to-noise
  ratio, continuing to hunt for candidates has negative expected value. What is worth
  defending is the context-efficiency result, not an accuracy number bought with two orders
  of magnitude more context.
