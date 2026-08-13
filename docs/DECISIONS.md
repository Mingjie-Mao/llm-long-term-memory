# Design decisions

A running log of what was chosen, what was rejected, and — where a measurement
settled it — what the numbers said. Entries are append-only; when a decision is
reversed the original stays and a new entry supersedes it.

---

## D1 — Benchmark: LongMemEval, not LoCoMo

**Decision.** Report on LongMemEval-S. Do not report LoCoMo numbers.

**Why.** LoCoMo is the more commonly cited memory benchmark, but by 2026 it has
three problems: an audit found roughly 6.4% of its answer key is wrong, its LLM
judge accepts about 63% of intentionally wrong answers, and its conversations
average ~26k tokens — small enough to fit in a modern context window, so it does
not test long-term memory under pressure. Published systems already report ~92% on
it. A number produced on a saturated, noisy benchmark cannot support a claim.

LongMemEval-S is ~122k tokens of history per question (measured, see D5) and scores
five distinct abilities separately, including knowledge updates and abstention —
the two that a memory system is uniquely responsible for.

**Rejected.** BEAM (ICLR'26) and PersistBench (ICML'26) are better fits for the
forgetting/eviction work in P5, but both are new and small. Revisit in V6.

---

## D2 — No SOTA claims

**Decision.** The results table compares ChronoMem's own ablation variants against
`full_context` and `naive_rag` baselines. It never claims to beat a third-party
system.

**Why.** Cross-system memory numbers are only comparable under an identical judge
model and judge prompt, and published results do not share either — the public
dispute over competing LoCoMo claims is precisely this failure. A claim that cannot
be defended under questioning is worse than no claim.

---

## D3 — Three separately configured model roles

**Decision.** `extractor`, `answerer`, and `judge` are independent config fields.
The answerer and judge are pinned for the life of the project.

**Why.** They have different requirements. The extractor is high-volume and
structured, so it should be the cheapest adequate model. The answerer is the thing
under test and must be held constant, or differences between table rows stop being
attributable to the memory system. The judge must never change, or results from
different weeks are not comparable.

A deliberately mid-tier answerer is preferred over the strongest available one: a
very strong answerer compensates for retrieval defects with its own reasoning,
which flattens the ablation table and hides the work.

**Consequence.** Because the free tier no longer offers a Pro-class model, the judge
runs at the same tier as the answerer. Judge reliability is therefore not assumed —
see D4.

---

## D4 — Judge agreement is measured, not assumed

**Decision.** Hand-label 50 questions and report judge/human agreement in the README
alongside the accuracy numbers.

**Why.** D1 rejects LoCoMo partly because its judge is unreliable. Using an
LLM judge without measuring it would reproduce the same flaw. The cost is one hour
of labeling; the return is that every number in the table carries a stated
reliability bound. If agreement lands below ~90%, the judge prompt is reworked
before any results are published.

---

## D5 — Ingestion is request-bound, not cost-bound (measured)

**Context.** The project runs on the Gemini free tier, which caps requests/minute,
tokens/minute, and requests/day independently; exceeding any one returns 429. At
~1,500 requests/day, the number of requests an ingestion costs decides the
schedule.

**Measurement** (`chronomem data stats --variant s`, 2026-08-10):

| | |
|---|---|
| questions | 500 (30 abstention) |
| sessions, with repeats | 23,867 |
| sessions, unique | 19,195 |
| **session sharing factor** | **1.24x** |
| turns | 246,750 |
| characters | 244,648,856 |
| est. tokens (chars/4) | ~61.2M |
| median sessions / question | 48 |
| median est. tokens / question | ~122k |

**The hypothesis that mattered was wrong.** Before measuring, the working
assumption was that LongMemEval-S questions might share a common haystack, which
would have collapsed ingestion cost by an order of magnitude. They do not — the
sharing factor is 1.24x, so deduplicating unique sessions saves about 20%, not 90%.
This is exactly why P0 measures before P2 builds.

**Budget** (`chronomem data plan`, 19,195 unique sessions @ 1,500 req/day):

| sessions/request | requests | est. tokens/request | days |
|---|---|---|---|
| 1 | 19,195 | 2,562 | 12.8 |
| 5 | 3,839 | 12,813 | 2.6 |
| **10** | **1,920** | **25,626** | **1.3** |
| 20 | 960 | 51,252 | 0.6 |
| 40 | 480 | 102,505 | 0.3 |

**Decision.** Batch **10 sessions per extraction request**.

One session per request — the obvious implementation — costs 12.8 days per
ingestion and is not viable. Twenty sessions would fit inside a single day's quota,
but asks one call to extract structured facts from ~51k tokens spanning twenty
unrelated conversations, and extraction quality degrades with span. Ten is the
smallest batch whose schedule is acceptable, and 1.3 days is a non-event because
the pipeline checkpoints and resumes (D6).

Tokens/minute, not requests/minute, will be the live throttle at this batch size:
25.6k tokens/request against a 250k TPM ceiling allows ~10 requests/minute, so a
full run is a few hours of wall clock spread across two quota days.

---

## D6 — Ingestion is checkpointed and resumable

**Decision.** The ingestion driver persists progress every N batches, and
`RateLimiter.acquire()` returns rather than blocking when the wait exceeds
`max_wait` (default 300s).

**Why.** D5 establishes that a full ingestion spans more than one quota day. A
driver that blocks on `time.sleep()` until the quota resets would hold a process
open for fifteen hours and lose everything on a laptop lid close. Instead,
exhausting the daily quota is a normal exit: checkpoint, stop, resume tomorrow.

The daily counter is persisted and keyed by the Pacific-time date Google resets on,
so restarting the process does not reset the count and silently blow through the
quota.

---

## D7 — SQLite + FTS5 over Postgres/pgvector

**Decision.** SQLite in WAL mode, with FTS5 supplying BM25.

**Why.** A single file makes the demo reproducible by anyone who clones the repo,
with no service to stand up. FTS5 provides BM25 without adding Elasticsearch. At
this scale nothing is bought by a heavier database.

**Mitigation.** `MemoryStore` is a Protocol, not a base class, and the same test
suite is written to run against any implementation. A pgvector backend in V2 is a
new file, not a rewrite.

---

## D8 — Exact vector search over FAISS or an ANN index

**Decision.** Brute-force normalized inner product over a numpy array.

**Why.** FAISS `IndexFlatIP` is the same brute-force scan; at <1M vectors the
dependency buys a constant factor on an operation that is not the bottleneck — the
LLM calls are. Approximate indexes (HNSW, IVF) are rejected for a stronger reason:
their recall noise is indistinguishable from a regression in the memory algorithm,
which would corrupt the very table the project exists to produce.

**Revisit if.** Profiling in P6 shows search is hot. The `VectorIndex` Protocol
makes that a drop-in.

---

## D9 — Bi-temporal columns from the first migration

**Decision.** `event_time` / `valid_from` / `valid_to` (world time) and
`ingested_at` (system time) exist in the schema at P1, three phases before the
temporal resolution logic that uses them.

**Why.** Adding a temporal axis later means re-ingesting 61M tokens, which on this
quota is another 1.3 days and a fresh set of extraction-quality variance. Unused
columns are free; a re-ingest is not.

**Related.** Superseded memories are marked, never deleted. The ablation compares
"with temporal resolution" against "without", and the second variant needs the rows
the first one retired.

---

## D10 — Retrieval weights are config, not code

**Decision.** The five hybrid-retrieval signals live in a YAML file; a weight of
0.0 disables a signal. An ablation variant is a config file.

**Why.** If variants are code branches, reproducing row 3 of the table six weeks
later means checking out an old commit. As config, every row in the published table
maps to a file in `configs/` that anyone can rerun.

---

## D12 — Free-tier limits are discovered at runtime, not configured (measured)

**Context.** Google does not publish per-model free-tier limits; the documentation
points at a dashboard. The numbers turn out to differ by orders of magnitude
between models, so any hard-coded constant was going to be wrong.

**Measured** (probe, 2026-08-10):

| model | free-tier daily limit | usable? |
|---|---|---|
| gemini-2.5-flash / -flash-lite | — (404, withdrawn) | no |
| gemini-2.5-pro, gemini-3.1-pro-preview | `limit: 0` | no — Pro is not on the free tier |
| **gemini-3.6-flash** | **20 / day** | no |
| **gemini-3.5-flash** | **20 / day** | no |
| gemini-3.5-flash-lite | per-minute only (~15 rpm) | **yes** |
| gemini-3.1-flash-lite | per-minute only (~15 rpm) | **yes** |

The full-fat flash models allow 20 requests/day — not enough for one 50-question
evaluation, let alone an ablation. This invalidated the original role assignment
mid-run.

**Decision.** All three roles run on flash-lite models, and the limiter learns real
limits from 429 responses rather than trusting config.

The one place a limit is stated authoritatively is the `QuotaFailure` detail of a
429, which carries `quotaId` and `quotaValue`. `QuotaManager.learn()` parses it,
updates the live limiter, and persists it to `stores/quota/observed-limits.json` so
the next run starts calibrated. A `PerDay` violation raises `DailyQuotaExhausted`
immediately instead of consuming the retry budget on something that will not clear
for hours.

**Also decided here.** Quotas are metered `PerProjectPerModel`, so assigning the
extractor, answerer, and judge to three *different* models gives each an
independent budget. D3's three-role split was a methodological choice; it turns out
to buy throughput as well.

**Cost.** The judge is now weaker than ideal — flash-lite, and weaker than the
answerer, which is the wrong direction for a grader. Pro is not purchasable with a
free key, so the human-agreement check in D4 is load-bearing rather than optional.

---

## D13 — Dev subsets are stratified by question type (measured)

**The bug.** The first 50-question run returned 76.0% accuracy — and every row said
`single-session-user`. The dataset file is grouped by question type, so
`instances[:50]` samples exactly one category: the easy one that plain vector
search already handles. Temporal reasoning, knowledge updates, multi-session
reasoning, and abstention were all absent — which is to say the subset excluded
everything the project is about.

**Decision.** `stratify()` takes a seeded, proportional sample across all six types,
using largest-remainder allocation so the 6%-sized category is not rounded away. The
seed is fixed so every variant is scored on identical questions.

**Effect.** The same baseline re-measured on a stratified subset scores **54.0%**,
not 76.0%. The first number was not wrong so much as meaningless.

---

## D14 — Latency excludes time spent waiting on quota (measured)

**The bug.** The first stratified run reported p95 latency of 34.3s. Almost all of
that was the rate limiter sleeping to respect ~15 requests/minute — the measurement
was of this project's quota tier, not of the system.

**Decision.** `Completion.api_latency_ms` times only the API call. Runners report
that, not wall-clock around `generate()`.

**Effect.** The same run measures p95 = **1.9s**. Rate-limit queueing is still
recorded, separately, in the usage report where it belongs.

---

## D15 — Retries cover transport failures, not just API errors (measured)

A 50-question run died partway through on `httpx.RemoteProtocolError` — the server
dropped the connection. The client caught `errors.APIError` only, so a transport
failure killed the run.

Runs here last hours and, on this quota, span days; a dropped connection is a
certainty, not an edge case. The retry path now covers `httpx.HTTPError`,
`TimeoutError`, and `ConnectionError` with the same backoff. Failed attempts are
still recorded in the usage tracker, because they consumed quota even though they
returned nothing.

The run itself lost nothing: JSONL checkpointing (D6) meant resuming picked up at
question 21.

---

## D16 — `full_context` is a reference point, not a ceiling (measured)

**Measured** (LongMemEval-S, stratified 50, 2026-08-11):

| | full_context | naive_rag |
|---|---|---|
| accuracy | 56.0% | 54.0% |
| single-session-user | 100% | 71.4% |
| knowledge-update | 87.5% | 62.5% |
| multi-session | 30.8% | **38.5%** |
| temporal-reasoning | 23.1% | **38.5%** |
| abstention | 50.0% | **100%** |
| median context tokens | 109,260 | 13,057 |
| p95 API latency | 16.0s | 1.9s |

The baseline that was supposed to be the accuracy ceiling is +2 points for 8.4x
the tokens — and it is *beaten* by naive retrieval on the two categories the
project exists to fix, plus abstention.

**Consequences for the plan.**

1. The headline framing is not "approach full context with fewer tokens". Full
   context is a weak and expensive reference point. The target is the 23–38% band
   on temporal and multi-session questions, where both baselines fail.
2. Ordering: temporal resolution (P4) moves ahead of consolidation in priority.
   The measurement says that is where the loss is concentrated.
3. Abstention needs watching as a regression risk. Naive RAG scores 100% partly
   *because* its context is sparse — it has nothing to confabulate from. Any
   variant that packs more relevant material into the prompt may lose abstention
   while gaining accuracy, and the results table must keep the column visible so
   that trade is explicit rather than hidden inside an average.

---

## D17 — Dedup is two stages, because a threshold cannot decide this

**Decision.** Embedding similarity is a recall filter only; an LLM makes the actual
call, with a three-way verdict (DUPLICATE / UPDATE / DISTINCT).

**Why not a threshold.** "The user likes Python" and "The user does not like
Python" sit at roughly 0.95 cosine similarity — same content words, one negation.
Any threshold high enough to catch real restatements also swallows that pair, and
dropping it leaves the store asserting a preference the user has since reversed.
Any threshold low enough to separate them lets genuine restatements through. There
is no setting that works, so the threshold is used only to keep the expensive call
rare (it fires on a few percent of candidates) and the decision is made on meaning.

**Why three verdicts.** A binary duplicate/not-duplicate collapses UPDATE into a
wrong answer either way: call "the user moved to Sydney" a duplicate of "the user
lives in Canberra" and the move is lost; call it distinct and the store now asserts
two contradictory locations with nothing marking which is current. UPDATE is what
P4 consumes to close `valid_to` on the old fact.

**Also.** Candidates are matched against existing memories by `(subject,
predicate)` as well as by embedding, because "lives in Canberra" and "relocated to
Sydney" are far apart in embedding space and are exactly the collision that matters.

---

## D18 — Answer coverage: what it caught, and what it cannot see (measured)

A pre-ingest gate: extract from the `oracle` split (evidence sessions only) and
check whether the gold answer survives into memory. Costs ~20 requests and isolates
extraction quality from retrieval quality — if the answer was destroyed at write
time, nothing downstream can recover it.

**What it caught.** First run scored 26.3%. The failures were specific details being
generalised away: gold `The Glass Menagerie` against a memory reading "the user is
interested in acting"; gold `10` (hours of documentaries) against "the user has a
goal to reduce screen time". The extraction prompt's framing — "durable facts",
"conservative", "no transient detail" — was filtering out exactly the specifics the
benchmark asks about. Rewriting it to demand quantities and proper nouns took the
measurable rate to **50.0%**, and `single-session-user` from **1/3 to 3/3**. That
is a real defect, found for about forty requests, that would otherwise have been
invisible until it had already cost end-to-end accuracy.

**What it cannot see.** Reading the residual failures changed the conclusion. Most
of LongMemEval's gold answers are *computed*, not stated:

| gold | what was in the store |
|---|---|
| `3 weeks` | Farmers' Market 2023-02-26 **and** Spring Fling 2023-03-20 |
| `an hour and a half` | "30-minute commute" **and** "morning routine takes one hour" |
| `vegan chili came first` | chili post 2023-03-09 **and** #PlankChallenge 2023-03-15 |
| `$270` | Maui resort ">$300/night" **and** Tokyo hostel "$30/night" |

In every one of these the extractor did its job — exact dates included — and the
string match still scores a miss. Five of six `temporal-reasoning` failures are of
this kind, as are all of `multi-session`.

**Consequences.**

1. `multi-session` and `single-session-preference` are excluded from the headline
   rate (`UNMEASURABLE_TYPES`) and reported separately. The preference golds are
   paragraphs describing a desired response style, which no memory could contain.
2. **The number is a regression detector, not a target.** Tuning the extraction
   prompt to raise it past the point where specifics are retained would be fitting
   to an instrument that is blind to two thirds of the benchmark.
3. It reframes the rest of the project: the benchmark rewards *reasoning over*
   retrieved facts, not recall of them. Component facts are already being stored
   correctly. What decides accuracy is whether the P6 packer puts the right
   *combination* in front of the model — both dates, not just the more similar one.
   That strengthens the case for budget-aware packing over more retrieval tuning.

---

## D19 — Assistant turns are extracted too

**Decision.** Record specific things the assistant told this user — a named
product, a number, an item from a list — with `subject` set to `"assistant"`.

**Why.** The first extraction prompt said "do not extract anything said by the
assistant". LongMemEval's `single-session-assistant` category (56 of 500 questions)
asks exactly that: "what sealant did you recommend?". The prompt forbade storing
the only place the answer exists. This is not benchmark-fitting — "remind me what
you suggested" is a normal thing to ask a system with memory.

Scoped narrowly to concrete, user-specific content: the assistant's generic advice
and its statements about itself stay excluded, or the store fills with boilerplate.

---

## D20 — Single-valued vs multi-valued predicates (measured)

**Symptom.** A 60-session trial ingest cost 6 extraction requests and **72
adjudication requests** — dedup was twelve times more expensive than the work it
was supporting — and every one of those 72 calls returned DISTINCT. Zero
duplicates, zero updates, 92% of the request budget spent to change nothing.

Extrapolated to the full corpus that is ~25,000 requests instead of the ~1,920 D5
planned for: **16 days rather than 1.3**. The plan was invalid.

**Cause.** `_neighbours` treated any `(subject, predicate)` collision as worth
adjudicating. But `subject` is almost always `"user"`, and the common predicates are
generic, so `user/owns/peace lily` collided with `user/owns/Fitbit`, and
`user/prefers/sourdough` with `user/prefers/Philips Hue`. Every possession met every
other possession.

**Decision.** Split predicates by arity. A predicate is *single-valued* when the
user can hold one value at a time — `lives_in`, `works_as`, `works_at`, `studies`,
`has_goal`, `scheduled`. Everything else is multi-valued. Only single-valued
collisions are adjudicated.

**Result** (same 60 sessions, before and after):

| | before | after |
|---|---|---|
| extraction requests | 6 | 6 |
| adjudication requests | **72** | **3** |
| duplicates caught | 0 | 1 |
| memories written | 57 | 56 |

A 24x reduction in LLM calls that also *started* catching duplicates, with memory
count essentially unchanged — so the calls removed were the ones doing nothing, not
recall being thrown away. Full-corpus budget returns to ~2,880 requests.

**Why it is not just a cost fix.** The same distinction is a correctness
precondition for P4. Superseding "the user owns a peace lily" because they later
mention a snake plant would close `valid_to` on a fact that is still true. Arity is
what makes supersede meaningful, and it happened to surface here as a bill.

**Generalisation.** The predicate list is hand-maintained and English-specific,
which is a real limitation. It is the right trade for now — the alternative is an
LLM call to classify arity, which reintroduces the cost this removes — but a
predicate registry learned from the corpus is the obvious V2 improvement.

---

## D21 — Temporal resolution rebuilds timelines, it does not compare pairs

**Decision.** For each `(subject, predicate)` key, sort every fact by `event_time`
and rewrite the whole chain: each memory's `valid_to` becomes the next one's
`event_time`, and only the last stays open.

**Why not the obvious thing.** "The memory that just arrived supersedes the one
already there" is wrong here, because ingestion order has nothing to do with event
order. Sessions are batched arbitrarily (D5), so the fact arriving now is as likely
to be from January as from August; a pairwise rule would let a late-arriving
*older* fact become current. Worse, once a memory has been superseded it is no
longer the head, so a fact landing between two existing ones could never rewire the
link that now points past it.

Rebuilding is idempotent, order-independent, self-correcting, and costs no LLM
calls — it is all SQL, so it can be re-run over the whole store at the end of every
ingest.

**Consecutive restatements are folded, and the earliest member owns the interval.**
"I live in Canberra" in March and again in June did not move anyone. Counting that
as a supersede would report a move that never happened; letting June own the
interval would answer "when did you move to Canberra?" with the wrong date. So the
run collapses to one interval starting in March, later mentions are closed but
counted as `restatements` rather than `superseded`, and "has the user moved?" stays
distinct from "how often was it mentioned?".

**Undated facts are left alone.** Extraction cannot always find a date. Guessing a
position on the timeline would produce a confidently wrong "current" value — the
exact failure this module exists to prevent — so they stay active and are counted
in `skipped_undated`.

---

## D22 — Arity is the default rule; the user's wording overrides it

**Problem.** D20 decides supersede by predicate arity. But the project's own
headline example does not fit: `uses_tool` is genuinely multi-valued — using
PyTorch does not stop you using NumPy — so a static list would leave

    Jan  I use TensorFlow.  →  Aug  I've switched completely to PyTorch.

unresolved. Arity is a property of the *key*; what makes this a replacement is the
*sentence*.

**Decision.** Extraction emits `replaces_previous` when the user explicitly signals
a change ("I switched to X", "I no longer do Y", "I moved from A to B"). The
resolver resolves a key when the predicate is single-valued **or** any memory on it
carries that flag.

**Why this is cheap.** The flag is one more field on an extraction call that was
already being made, so it costs no additional requests — unlike the alternative of
an LLM pass to classify predicate arity, which would reintroduce exactly the cost
D20 removed.

**Risk.** A false positive closes a fact that is still true. The prompt therefore
asks for it only on explicit replacement language and says not to set it
speculatively; `uses_framework` is also in the arity list so the common case does
not depend on the flag firing.

---

## D23 — The arity list got 2 of 7 wrong, and the errors are asymmetric (measured)

**What happened.** After the first real ingest (450 sessions, 48 supersedes), a look
at the actual supersede chains showed this:

```
has_goal   read 20 books in 2023   -> marathon training     closed 2023-05-21
has_goal   marathon training       -> start a book club     closed 2023-05-22
has_goal   start a book club       -> hike 10 miles         closed 2023-05-22
has_goal   hike 10 miles           -> plant pollinator gard closed 2023-05-22
```

`has_goal` and `scheduled` were on the single-valued list. A person holds many goals
and many scheduled events at once, so the resolver had chained a set of unrelated,
simultaneously-true facts into a supersede sequence. Essentially every supersede in
the store was wrong, and each one had removed a true fact from retrieval.

**The errors are not symmetric, and the list should reflect that.**

| | consequence |
|---|---|
| missing a supersede | a stale fact competes with the current one — exactly the status quo the baselines already have. Bounded. |
| a wrong supersede | a still-true fact disappears from retrieval. Unrecoverable downstream, and directly produces wrong answers. |

So the list is now minimal — `lives_in`, `works_as`, `works_at`, `uses_framework` —
and anything doubtful stays off it. `studies` was dropped too: one can study several
subjects.

**Repair, not just abstain.** Correcting the list is useless if stores built under
the old one keep their damage, and rebuilding costs hours of quota. So when the
resolver finds a key it will not resolve, it now *releases* any memory sitting
superseded on that key and counts it as `repaired`. A nonzero `repaired` on a re-run
is the signal that an earlier arity call was wrong and this store had true facts
hidden.

**What this says about the approach.** Getting 2 of 7 wrong on the first pass is the
strongest evidence yet that a hand-maintained arity list is the weak point of the
design (already flagged in D20). It is still the right trade for now — the
alternative is an LLM call per predicate, reintroducing the cost D20 removed — but
the repair path is what makes it survivable, and a learned predicate registry moves
up the V2 list.

---

## D21 — Resolution rebuilds a timeline; it does not compare pairs

**Decision.** For each `(subject, predicate)` key, sort every memory by
`event_time` and rewrite the entire chain: each one's `valid_to` becomes the next
one's `event_time`, and only the last stays open.

**Why not the obvious thing.** "The memory that just arrived supersedes the one
already there" fails twice here, both for the same underlying reason — ingestion
order has nothing to do with event order, because sessions are batched arbitrarily
(D5).

1. A January fact routinely arrives *after* an August one. Pairwise, it would
   become current, and the store would assert TensorFlow.
2. Once a memory is superseded it is no longer the head, so a fact landing
   *between* two existing ones can never rewire the link that now points past it.
   Jan→Aug stays Jan→Aug even after March arrives, and Jan's `valid_to` is silently
   eight months too late.

Rebuilding is idempotent, order-independent, self-correcting, and costs no LLM
calls — it is all SQL. Both failures are pinned as tests.

**Restatements are not moves.** "I live in Canberra" in March and again in June
collapses to one interval owned by *March*, because "when did you move?" is
answered by the first mention. The June row is closed so the key has one live
value, but it is counted as a `restatement` and points back at the interval owner
rather than forward at a successor — keeping "did the user move?" separable from
"how often was it mentioned?".

---

## D22 — Arity is the default rule, not the whole rule

**Decision.** A key is resolvable if the predicate is single-valued **or** any
memory on it carries `replaces_previous`, which extraction sets when the user
explicitly signalled a change ("I switched to X", "I no longer do Y").

**Why.** D20's static list cannot express the project's own headline example.
`uses_tool` is genuinely multi-valued — using PyTorch does not stop you using
NumPy — so a list-only rule leaves the TensorFlow → PyTorch chain unresolved, which
is the case the README opens with. The user saying they switched is a supersede by
any reading, and it costs nothing extra to capture: one more boolean in a
structured output the extractor was already producing.

---

## D23 — The arity list was wrong, and the fix cost SQL rather than requests

**Measured.** A 200-session ingest produced 20 supersessions. Inspecting them:

| key | chain | verdict |
|---|---|---|
| `user/lives_in` | Tokyo → South Bay → Las Vegas | correct |
| `user/scheduled` | layover in London → cooking class → the 9:15 train → Friday game nights → Overland Expo | **wrong** |
| `user/has_goal` | collect 200 leads → build a DL portfolio → learn front-end → a 10:30 bedtime | **wrong** |

`scheduled` and `has_goal` are not single-valued. A person holds several goals and
several appointments at once, so every link in those chains was a false supersede
that retired a fact still in force — the precise failure D20 was written to prevent,
committed by D20's own list.

**Decision.** Both removed. The set is now `lives_in`, `works_as`, `works_at`,
`studies` — attributes a person can only hold one of. Entries have to earn their
place.

**The more useful consequence.** Because resolution rebuilds timelines from stored
data (D21) and touches no LLM, correcting the list is a *re-resolution*, not a
re-ingest: `chronomem resolve` repairs the store in seconds against ~2,880 requests
and a day of quota. That required one addition — when a key is no longer resolvable,
rows superseded under the old list are promoted back — without which the mistake
would have been baked in permanently.

Arity is a property of the predicate, not of the data. Keeping it out of the
ingested artifact is what makes it cheap to be wrong about.

---

## D24 — The server's refusal outranks the discovered limit (measured)

**Observed.** Ingestion stopped with `daily quota exhausted for
gemini-3.1-flash-lite; resets in 0.0h`. Both halves of that were wrong in an
instructive way.

The local counter stood at **367** against a limit of **500** discovered from an
earlier 429 (D12). So the server refused while the limiter still believed it had a
quarter of the day's budget left. The counter undercounts by construction: retries
that fail before a response, `count_tokens` calls, and requests issued by earlier
processes in the same day all spend real budget without passing through it.

And because `check()` still saw headroom, it returned `None`, so the error fell back
to a placeholder `Wait(0.0)` — reporting a 0.0h reset for a quota that was in fact
**15.8 hours** from resetting. An operator reading that would rerun immediately and
burn the retry budget for nothing.

**Decision.** A per-day 429 pins the counter to the limit (`mark_exhausted`). The
server is the authority; the local counter is a cost-saving estimate that exists to
avoid provoking 429s, not a source of truth about remaining budget.

**Generalisation.** D12 discovers limits from the API and treats them as fact. This
is the correction: discovered limits are still only an approximation of a quota
system with dimensions we cannot see. The design should degrade toward believing the
server, never toward believing its own bookkeeping.

---

## D24 — Each question is its own user (measured; invalidates the D5 saving)

**What the store looked like.** After the first namespaced-by-nothing ingest, the
`lives_in` timeline read:

```
Toronto -> Greenville -> Seattle -> Tokyo -> San Diego -> 95123 -> Tokyo
        -> Shanghai -> suburban Los Angeles -> Hyderabad -> Shibuya -> Las Vegas
```

and `works_as` ran software engineer -> marketing specialist -> accountant ->
manager -> musician -> Principal Data Scientist -> photographer. Almost all of it
closing within one week of May 2023.

**Cause.** LongMemEval builds a question's haystack by padding its evidence sessions
with distractors drawn from unrelated conversations. Two questions therefore share
no history at all — measured: Q1 has 47 sessions, Q2 has 43, and the intersection is
**zero**. Ingestion had been taking the union of all sessions and writing them under
a single `user_id`, merging fifty different simulated people into one store. The
consequences compound:

1. Temporal resolution chained unrelated personas' facts into supersede sequences.
2. Retrieval for one question could return another question's evidence — inflating
   or destroying accuracy for reasons unrelated to the memory system.
3. Every "current value" for a single-valued key was arbitrary.

**Decision.** Namespace ingestion and retrieval by `question_id`. Batches never
straddle a namespace, so no extraction call can attribute a fact to the wrong user.
The checkpoint key becomes `namespace:session_id`.

**This retires the D5 optimisation.** Deduplicating sessions across questions was
what produced the union in the first place, and it saved 20% of requests (sharing
factor 1.24x). A session appearing in two haystacks belongs to two different
simulated users and must now be extracted into both. The trade is not close: 20%
fewer requests is worth nothing if the store is wrong.

**How it was found.** Not by a test — every unit test passed throughout — but by
printing the actual supersede chains from a real store and reading them. The
lesson generalises: the tests confirm the resolver does what it was told, and only
looking at real output shows it was told the wrong thing.

---

## D27 — v1 is frozen, the dev 50 are burned, and fidelity is measured without gold

Three decisions taken together after D26, before any attempt to fix extraction.

**1. `results/frozen/chronomem-v1/` is immutable.** The 26% run, its four result
files, and the exact extraction prompt that produced it (sha256 `e6e7a4e4…`) are
kept verbatim. Aggressive compression cut context ~28x and cost half the accuracy;
that is the measurement the next phase is designed against, and a reader who cannot
see it has to take the diagnosis on trust. Tuning until the number looks better and
reporting only that would delete the reason for everything that follows.

**2. The 50-question subset is dev, not test.** Batch size, the predicate arity
list, the v2 extraction prompt, and the decision to build P4 before P3 were all
chosen by looking at those questions. A headline number produced on them measures
the fit of those choices as much as the system. `split_dev_test` draws a disjoint
stratified 100 from the remaining 450, to be run once, at the end.

**3. Fidelity is measured against the source text, not the answer key.** The
coverage gate (D18) needs a gold answer to appear verbatim in a memory, which is a
minority of LongMemEval and blind to the categories that failed. The new gate asks a
question with no answer key at all: of the quantities, durations, dates,
relative-time expressions and proper nouns *the user stated*, how many survive
extraction? It cannot be fitted to the evaluation set, it runs on sessions no
evaluation will touch, and it is per-category, so it says which clause of the prompt
to write rather than just that something is wrong.

---

## D28 — The fidelity metric was wrong twice before the extractor was (measured)

Both times, a low score turned out to be the denominator counting specifics no
extractor should keep. Both were caught by reading the misses instead of acting on
the number.

| | symptom | cause | effect on the score |
|---|---|---|---|
| quantity | "dropped: `1`, `2`, `3`, `4`" | a user pasted a numbered document; list markers counted as stated quantities | 90 → 39 stated; recall 18.9% → 38.5% |
| proper_noun | "dropped: `as i'm`" | capitalised sentence openers matched the multi-word pattern | ~10% of matches |
| quantity | `16GB` never counted | `\b\d+\b` has no word boundary between `16` and `GB`, so unit-suffixed numbers were invisible | silently excluded the most useful cases |

A third correction excluded interrogative sentences and assistant turns: a user
asking *"how tall was Osama bin Laden?"* has stated nothing about themselves, and
scoring those proper nouns as losses would push the prompt toward storing trivia.

**The rule this establishes.** A metric that disagrees with the system is not
evidence about the system until its denominator has been read. Tuning a prompt
against an uninspected denominator does not produce a better extractor; it produces
one fitted to the metric's mistakes. All four corrections are pinned as tests.

---

## D29 — v2 improves extraction, and not enough to spend an ingest on (measured)

Same 60 held-out sessions, same batch size, same corrected metric. v1 restored from
git to be scored under the same rules rather than compared against its own older,
looser measurement.

| facet | v1 | v2 |
|---|---|---|
| duration | 35.7% | **50.0%** |
| relative_time | 25.0% | **50.0%** |
| quantity | 33.3% | 38.5% |
| proper_noun | 16.7% | 16.7% |
| **overall** | **25.0%** | **30.8%** |
| memories/session | 0.9 | 1.1 |

The prompt rewrite works where it was specific — durations and relative time, both
named with examples, both roughly doubled. It does nothing for proper nouns, which
were named with examples too.

**Decision: do not run the full ingest yet.** v1's 25% fidelity produced 26%
end-to-end accuracy. 30.8% is a real improvement and not one that plausibly changes
the outcome, and an ingest plus two evaluation runs costs most of a day's quota. The
gate exists precisely so that this judgement can be made for four requests.

**Batch size is not the constraint, which was worth knowing.** Extracting one
session per request instead of fifteen triples memory density and does not improve
fidelity at all:

| sessions/request | memories/session | overall fidelity |
|---|---|---|
| 15 | 1.1 | 24.0% |
| 5 | 2.2 | 23.9% |
| 1 | 3.4 | 22.4% |

More records about the same subset of the content. The model is not running out of
room; it is deciding most specifics are not worth recording. That rules out the
cheapest hypothesis and keeps the D5 batch size on evidence rather than on quota
arithmetic alone.

---

## D30 — Worked examples beat rules, and the metric was wrong a third time (measured)

**What other systems do.** Mem0's fact-extraction prompt was read directly rather
than guessed at. Three differences from ChronoMem's mattered:

| | Mem0 | ChronoMem v2 |
|---|---|---|
| teaching device | six input→output examples | prose rules |
| granularity | one sentence split into several facts, shown | asserted in a rule |
| output schema | a flat list of strings | nine fields per memory |

**Adding worked examples (v4).** Same 60 held-out sessions, same batch size, same
metric:

| facet | v1 | v2 (rules) | v4 (rules + examples) |
|---|---|---|---|
| quantity | 33.3% | 42.6% | **51.9%** |
| duration | 35.7% | 50.0% | **57.1%** |
| relative_time | 25.0% | 50.0% | 33.3% |
| proper_noun | 16.7% | 17.0% | 17.0% |
| **overall** | **25.0%** | **33.6%** | **36.6%** |

Showing a four-sentence turn expanded into five records moved quantity retention
nine points where a rule saying "one record per fact" had not. `relative_time` fell,
on a base of twelve — two items, at or below the noise of this sample.

**The v3 detour is kept as a negative result.** Framing extraction as pure
transcription — "you do not decide what is worth remembering" — produced *more*
memories per session and lower fidelity across every facet (31.3% overall). Removing
the model's judgement did not make it more faithful; it made it verbose.

**Proper nouns never moved across four prompt versions, and the extractor was
right.** Inspecting the misses:

```
"I'd like to know more about The 7½ Deaths of Evelyn Hardcastle."
"I'm curious to know more about the author, Stuart Turton."
"I've heard great things about the Sonos One."
```

Every one is a request for information phrased as a statement. The denominator
excludes sentences ending in `?`, which does not catch these, so the extractor was
being penalised for correctly declining to record "the user owns a Sonos One".

That is the third time a low score was the metric rather than the system — after
list markers counted as quantities and sentence openers as proper nouns. The
practice that caught all three was reading the misses before acting on the number,
and it has now paid for itself three times over.

**A real design question sits underneath it.** Mem0 explicitly tracks "Plans and
Intentions"; ChronoMem's prompt says not to record things the user considered but
did not do. "The user is considering a Sonos One" is a defensible memory, and
LongMemEval asks preference questions where it would matter. Deferred rather than
changed mid-comparison: it alters what a memory *means*, so it needs its own
before-and-after rather than being folded into a fidelity fix.

**Still not enough to ingest.** 36.6% against v1's 25.0% is a 46% relative
improvement and remains far from a representation that can carry the answers. The
next lever is the output schema — nine fields per memory against Mem0's bare
strings — which is a structural change, not another prompt edit.

---

## D26 — ChronoMem loses to both baselines, and the cause is upstream of P4 (measured)

**Result** (LongMemEval-S, stratified 50, same answerer and judge as every other row):

| variant | accuracy | temporal | know-update | evid. recall | ctx tokens |
|---|---|---|---|---|---|
| `full_context` | 56.0% | 23.1% | 87.5% | — | 109,260 |
| `naive_rag` | 54.0% | 46.2% | 75.0% | 94.0% | 13,057 |
| `chronomem_no_temporal` | **26.0%** | 7.7% | 37.5% | 80.0% | 331 |
| `chronomem` | **26.0%** | 7.7% | 62.5% | 80.0% | 465 |

Temporal resolution changes nothing detectable: 3 wins, 3 losses, p = 1.000.

**Where the loss is, precisely.** Retrieval is not the problem. The evidence
session's memories are recalled for **40 of 50** questions — and of those 40, only
**12 are answered correctly**. **28 of 50** answers are "I do not know". The right
memories are in the prompt and the answer is not in them.

One case traced end to end. *"How long have I been collecting vintage cameras?"*,
gold `three months`. The evidence session yielded three memories, ranked first:

```
- The user owns 17 vintage cameras, including a Brownie Hawkeye acquired in May 2023.
- The user owns a rare 1978 pressing of Fleetwood Mac's Rumours.
- The user owns a Mondo poster featuring Hogwarts castle.
```

The duration was never extracted. The neighbouring question (`25` postcards) failed
the same way and the model answered `17` — the nearest number in context.

**This is the coverage gate's prediction arriving end to end.** D18 measured
measurable answer coverage at 50% and said explicitly that the number was a
regression detector rather than a target. 50% is the ceiling this store can support;
26% is what remains after retrieval and reasoning take their share.

**The sequencing call was wrong.** P4 was promoted ahead of P3 because temporal
reasoning was the worst category for both baselines (D16). That reasoning treated a
category score as a diagnosis. It was a symptom: temporal questions need dates and
durations, and those are exactly the specifics extraction drops. The timeline
machinery is correct — 35 supersessions, chains verified by hand, 16 tests — and it
is **not load-bearing**, because it orders facts that no longer contain the answer.
Building it did not waste the quota it cost, but it could not have paid off before
the representation did.

**What this changes.** The next constraint is extraction fidelity, not ranking and
not the timeline. The concrete handle is memories per session: **0.8**, against
sessions of ~12 substantive turns. Options, cheapest first:

1. Raise extraction density — the current prompt asks for facts worth remembering,
   which quietly excludes durations, counts, and relative time expressions.
2. Smaller batches. 15 sessions per request was chosen for quota (D5 revision), and
   compression per session may be the price.
3. Keep the source session text addressable so the packer can fall back to it. This
   changes what is being measured and needs its own row rather than a silent switch.

**Reported, not buried.** A memory system that scores half of naive RAG is the
result. It is in the README table with the others.

---

## D25 — One namespace per question, and sessions are not shared across them

**The bug.** The first real ingest wrote every question's haystack under a single
`user_id`. Inspecting the resulting supersede chains:

```
lives_in   Greenville, South Carolina -> Seattle
lives_in   Tokyo                      -> San_Diego_92101
lives_in   Tokyo                      -> Shanghai
lives_in   95123                      -> Tokyo
```

Ten `lives_in` values under one identity. LongMemEval pads each question's haystack
with distractor sessions drawn from unrelated conversations, so the 2,348 sessions
behind 50 questions are **50 different simulated people**. Merging them did two
kinds of damage: retrieval for one question could return another's evidence, and the
temporal resolver — working exactly as designed — chained strangers' cities into a
move history and marked most of them superseded, i.e. removed them from retrieval.

**Decision.** The namespace is the `question_id`. Batches never straddle one, the
checkpoint key is `namespace:session_id`, and retrieval filters on it.

**Consequence for D5.** Sessions must *not* be deduplicated across questions: the
same session in two haystacks is two units of work belonging to two personas.
Ingestion is therefore 23,867 session-instances rather than 19,195 unique — 24%
more, and the 1.24x sharing factor stops being a saving at all.

**What let it through.** Every layer was individually correct; nothing tested that a
fact written under one question could not surface under another. There is now a test
for the invariant itself rather than for the components. The failure was only
visible by reading the supersede chains — the counters said 83 supersedes and looked
healthy, which is a reminder that aggregate metrics do not show a systematically
wrong store.

---

## D25 — The binding constraint flips between models; match the model to the shape
of the work (measured)

**What happened.** Moving the extractor to `gemma-4-31b-it` — chosen for its own
quota pool and a generous 1,500 requests/day — made ingestion *worse*, not better.
Gemma's free-tier allowance is **16,000 input tokens per minute**, and a
ten-session batch is ~25,600 tokens. A single request could never be sent at all.

The limiter did not say that. It computed "wait until the window frees up", found
an empty deque, and died with `IndexError: deque index out of range`.

**Two fixes, both about honesty of failure.**

1. A request larger than the entire per-minute allowance now raises
   `RequestTooLarge`, naming the number and the remedy. It is not a rate-limit
   condition — waiting cannot help — so reporting it as one was the actual bug.
2. Batch size is derived from the extractor's observed TPM rather than the constant
   D5 chose. D5 picked ten sessions against a 250k-TPM budget; that number was
   never a property of the corpus, only of the model that happened to be
   configured. `fit_batch_size` dropped it to 5 for gemma automatically.

**The deeper point: which limit binds is a property of the model, not the task.**

| model | RPD | TPM | one dev-subset pass (~6.1M tokens) |
|---|---|---|---|
| gemma-4-31b-it | 1,500 | 16k | **6.4 hours**, TPM-bound |
| gemini-3.1-flash-lite | 500 | 250k | **~25 min**, request-bound |

D5 concluded "ingestion is request-bound, not cost-bound". That was true of the
model measured, and stopped being true one model later.

**Resulting allocation — each role gets the model whose constraint profile fits its
work, not the largest available:**

| role | model | why |
|---|---|---|
| extractor | `gemini-3.1-flash-lite` | 6.1M tokens needs the 250k TPM; nothing else finishes in minutes |
| answerer | `gemini-3.5-flash-lite` | the system under test, pinned |
| judge | `gemma-4-31b-it` | inputs are ~200 tokens, so the TPM that disqualifies it for extraction is irrelevant, while its 1,500 RPD is exactly what judging needs |

The 16k TPM that makes gemma useless for extraction costs nothing for judging. That
also restores the separate-pools property of D3, which this config had quietly lost
by putting extractor and judge on the same model — a large ingest was starving the
evaluation.

---

## D24 — Three roles are only independent if they are on three quota pools

**Symptom.** An ingest stopped with the extractor's daily budget spent, and the
evaluation could not run either: `gemini-3.1-flash-lite` was configured as both
extractor and judge, and its discovered limit is 500 requests/day — not the 1,500
the other flash-lite model gets. One ingest (~470 requests) consumed the pool the
judge needed.

**Why it was missed.** D3 chose three separate models for a methodological reason:
the answerer must be held constant, and the judge must not grade its own prose.
Free-tier metering being *per model* (D5) then makes the same split a throughput
decision — but only if the three models are actually distinct. Two roles pointing
at one model satisfies the methodology and silently defeats the throughput.

**Decision.** Judge moves to `gemma-4-31b-it`: its own 1,500/day, structured output
verified, and a different family from the answerer so the no-self-grading property
is strengthened rather than weakened.

**Cost, stated plainly.** D3 pins the judge for the life of the project, so both
baseline rows have to be re-graded under the new judge before the `chronomem` rows
can be compared against them. That is 100 answerer plus 100 judge requests — cheap,
but it has to happen, and reporting rows graded by two different judges in one table
would be exactly the flaw D1 rejects LoCoMo for.

**Assignment after the fix**, with discovered limits:

| role | model | requests/day | why |
|---|---|---|---|
| extractor | `gemini-3.1-flash-lite` | 500 | ~470 per full dev-subset ingest |
| answerer | `gemini-3.5-flash-lite` | 1,500 | pinned since P0; cannot move |
| judge | `gemma-4-31b-it` | 1,500 | own pool, separate family |

---

## D25 — The artifact is the record; a run that disagrees with it fails loudly

**Symptom.** A run printed `50 questions, 56.0%`. The file it had just written held
30 results. Both numbers were plausible, and the console one nearly reached the
README.

**Cause.** Mine, not the code's: a `until pgrep ...; do sleep; done` loop used to
wait for one evaluation before starting the next. In the gap between the first
run's last request and the second run's first, no matching process existed, the
loop declared victory, and two evaluations wrote the same paths concurrently.

**Why a guard rather than just fixing the loop.** The loop was one instance of a
class — a concurrent writer, a truncation, a lost write, a killed process — and
every member of that class produces the same symptom: a plausible accuracy that
nothing in the system objects to. Silent wrongness is the failure mode this
project is least able to absorb, because the only output is a table of numbers.

**Decision.** `run_eval` re-reads the file it wrote and raises `ArtifactMismatch`
unless the row count matches the report. No number is returned when the two
disagree.

**Found while writing its tests.** `_load_done` (the resume path) tolerates the
partial final line a killed process leaves behind; `load_report` (the reporting
path) did not, and crashed. Since interruption by daily quota is the *normal* case
here (D6), that meant any interrupted run could never be reported on. Fixed.

**Correction to earlier numbers.** The `naive_rag` 54.0% and `full_context` 56.0%
figures reported before this fix came from the contaminated run and are withdrawn.
The table is regenerated from artifacts written by a single serialized run.

---

## D26 — Compare in pairs, not headlines (measured)

**Measured.** `naive_rag`, re-run with nothing changed, scored **48.0%** and
**54.0%**. `temperature=0` does not make a hosted model deterministic, and the
judge's borderline calls move independently, so the noise compounds. Four flips out
of fifty is eight points — larger than any improvement this project is likely to
produce.

**Consequence.** The accuracy column cannot support a claim at n=50. Reporting
"+4 points from temporal resolution" would be reporting noise.

**Decision.** Every comparison is paired. Both variants answer the same questions,
so questions they both get right and both get wrong carry no information about
which is better; only the disagreements do. `chronomem eval compare` runs an exact
McNemar test over those:

    b01 = A wrong, B right      b10 = A right, B wrong
    under the null, wins ~ Binomial(b01 + b10, 0.5)

Exact rather than chi-square: the approximation is unreliable below ~25 discordant
pairs and this project will usually have fewer.

**First application, and it changed the headline.** `full_context` vs `naive_rag`:

| | |
|---|---|
| both right | 19 |
| both wrong | 14 |
| naive_rag wins | 8 |
| naive_rag losses | 9 |
| **p-value** | **1.000** |

The two-point gap in the accuracy column is nothing. The real finding is that
**109,260 context tokens buy no measurable accuracy over 13,057** — an 8.4x cost
for a difference the test cannot distinguish from a coin flip.

That is a better result than a small win. A small win would have been unpublishable
noise; this is a clean negative that sets the target for everything after it: the
ceiling is not "approach full context", because full context is not above naive
retrieval.

**Standing rule.** No variant is reported as an improvement on headline accuracy
alone. It has to win the paired test, and the observed run-to-run spread is
reported next to the table so a reader can see the noise floor.

---

## D11 — Embeddings run locally

**Decision.** `all-MiniLM-L6-v2` via sentence-transformers on the local machine
(MPS), not an embedding API.

**Why.** Corpus-wide embedding is on the order of 61M tokens plus a re-embed
whenever the chunking changes. Locally that is free and unmetered; through an API
it would consume the same daily request quota that D5 shows is already the binding
constraint on the project.
