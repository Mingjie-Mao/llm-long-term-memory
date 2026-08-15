# Roadmap — from experiment to product

**Direction set 2026-08-14.** LLTM is a **persistent memory layer for LLM
agents**, not a LongMemEval experiment. The experiments were the justification for
the design; they are no longer the deliverable. Someone should be able to open the
repo, understand it in 30 seconds, run it in 2 minutes, and plug it into their own
agent in 5.

The benchmark work does not get thrown away — it becomes the evidence that the
design choices were real. It stops being the headline.

---

## Where the evidence landed

Measured on 31 fully-ingested questions ([results/a2-pilot.md](../results/a2-pilot.md)),
paired McNemar, same answerer and judge throughout:

| Variant | Accuracy | Median ctx tokens | vs `naive_rag` |
|---|---:|---:|---|
| `full_context` | 64.5% | 109,605 | — |
| `naive_rag` | 51.6% | 13,057 | baseline |
| `two_stage_hydrated` | 51.6% | 1,318 | ~10x less context |
| **`two_stage`** | **48.4%** | **439** | **~30x less context**, 6W-7L, p = 1.000 |
| `chronomem` (v1) | 19.4% | 465 | — |

Causal decomposition of the 19.4% → 51.6% gain:

| Step | Delta | Paired |
|---|---:|---|
| extraction rewrite (`chronomem` → `two_stage`) | **+29.0pp** | 11W-2L, **p = 0.022** |
| hydration (`two_stage` → `two_stage_hydrated`) | +3.2pp | 2W-1L, p = 1.000 |

**The product thesis follows from this, not from a hunch:**

> A compact structured memory is usually sufficient to answer from. Raw evidence
> should be recovered *selectively*, not carried by default.

So the shipping default is **`two_stage`**, and hydration becomes a **conditional
fallback** rather than an always-on stage. Hydration is not shown to be useless — it
is untested at n=3 disagreements — but it is not entitled to a 3x context cost by
default.

Architecturally that makes LLTM two stores rather than one, which is the shape
the product should have anyway:

```
                    Conversation
                         │
             ┌───────────┴───────────┐
             ↓                       ↓
     Raw conversation           Extractor
        archive                     │
             │                      ↓
             │             Structured memory
             │                      │
Query ───────┼──────────────────────┘
             │              memory retrieval
             │                      │
             │            enough to answer?
             │              ┌───────┴───────┐
             │             yes              no
             │              │                │
             │              ↓                ↓
             └────────> answer      raw-evidence search
                                             │
                                    source-cited answer
```

Structured memory is the understood long-term state; the conversation archive is the
ground truth for when compression turns out to have been lossy. The `turns` table
and the provenance anchors already implement the right-hand path — 4,843/4,843
memories resolve to a source turn — so this is a policy change, not new plumbing.

---

## Done

- **P7-0 Windows support.** `encoding="utf-8"` at 65 call sites, CI matrix across
  ubuntu/windows/macos, and `filterwarnings = ["error::EncodingWarning"]` so the
  defect cannot return silently on any platform.
- **Protocol frozen.** `results/manifests/dev50.json` (verified identical to the
  question set behind the frozen v1 rows) and `dev31-pilot.json`. Reported runs name
  a manifest via `--questions`; `--limit` is exploration only, because a stratified
  sample is scattered across the split rather than being a dataset-order prefix.
- **Staged recall instrumentation.** `recall_stages` records candidates → ranked →
  selected → hydrated, so a recall drop can be attributed to the stage that caused
  it instead of showing up as one end-of-pipeline number.
- **Cross-encoder reranker.** Implemented, tested, unmeasured. Now an optimisation
  item, not a milestone.
- **`single-session-assistant` root cause.** Diagnosed with zero quota; see
  [results/assistant-gap.md](../results/assistant-gap.md). Feeds P10.

---

## ⚠ Results break here: the answerer prompt changed 2026-08-14

`ANSWER_SYSTEM` was rewritten (P7.5 below). Every variant shares it, so **every
number measured before this change was produced by a different answerer** and is not
directly comparable to anything measured after it. That includes the whole dev31
pilot and the rerank sweep.

This is a deliberate trade, not an accident: the old prompt was measurably wrong —
it told the model to decline whenever the history "does not contain the
information", which made it refuse on advice questions where the memories had been
retrieved correctly. Keeping a known-bad answerer to preserve comparability with
runs that have not been published would be the wrong way round.

The frozen v1 rows in `results/table.md` stay as they are, labelled with the prompt
they were produced under. A2 runs on the new prompt, and the pilot numbers are
re-derived from it or dropped.

---

## P7.5 — Memory-aware answering ✅ 2026-08-14

The [preference audit](../results/preference-audit.md) found two defects downstream
of memory, both now fixed:

**Judge routed by question type.** LongMemEval's `single-session-preference` gold is
a *rubric* for a well-personalized reply, not a reference answer, and LongMemEval's
own evaluation uses a separate prompt for it. Grading it as a reference answer
failed a reply that had done exactly what the rubric asked. `Judge._prompt_for` now
routes: abstention outranks everything, `single-session-preference` gets the rubric
prompt, everything else keeps the reference prompt — and an unknown category falls
back to reference rather than silently becoming rubric-graded.

**Answerer treats memories as context, not a lookup table.** The old prompt produced
100% abstention, which is a strength, and also produced "I do not know what phone you
use" on a battery question where it had retrieved the user's power bank. The new one
draws the distinction that matters — *missing fact* versus *available context* —
so it still declines when a fact was never discussed but personalizes when the
memories support a useful reply.

**Scope-aware context rendering.** Memories are grouped under headings by `scope`
("User preferences", "Current plans", "Previously recommended by the assistant")
rather than flattened into one bullet list, so the model can tell a constraint from a
candidate answer. A store with no scopes renders exactly as before, so this changes
nothing for pre-P10 data.

Cross-encoder reranking moved to its own `rerank` extra: running the product no
longer downloads torch and a reranker model for a feature that is off by default and
[measured not to help](../results/rerank-pareto.md).

**Still open:** the three preference questions are not yet regression tests against a
live model — that needs quota. The raw-conversation fallback (hydrate only when
structured memory is insufficient) is designed but not implemented.

---

## Done — no embeddings for the raw fallback (measured) ✅ 2026-08-15

Whether BM25 needs a dense counterpart was answered by **evidence recall**, not
end-to-end accuracy — accuracy folds retrieval, answering and judging into one
number and cannot say which moved. Gold turns come from LongMemEval's own
`has_answer` flags; no API calls.

| query type | n | R@1 | R@3 | R@5 | MRR |
|---|---:|---:|---:|---:|---:|
| keyword (upper bound) | 31 | 93.5% | 96.8% | 96.8% | 0.946 |
| natural (realistic) | 31 | 54.8% | 74.2% | 74.2% | 0.640 |
| disjoint (constructed) | 29 | 6.9% | 6.9% | 6.9% | 0.069 |

**The 6.9% row is an artifact and must not be cited.** The manual spot check — the
step meant to validate the construction — found that deleting words shared with the
gold turn deletes the *question* too: *"How long have I been collecting vintage
cameras?"* becomes `"long"`. No retriever answers that, dense included.

Hand-written paraphrases tell the opposite story. Five phrasings of the Mayo Clinic
question, including ones with no "Mayo", "Clinic" or "YouTube", all retrieve the gold
turn at **rank 1** — genuine paraphrases keep topic words (posture, ergonomics,
sitting, back pain) and drop only the proper noun, which BM25 never needed.

And the 26% that `natural` misses is not a vocabulary problem: all eight are
`preference` (the gold is a rubric), `temporal-reasoning` (the answer is computed
from dates) or `multi-session` (the answer aggregates across turns). Swapping the
retriever changes which wrong turn comes back.

**Decision: no embeddings.** Full write-up, including the two conditions that would
reopen it, in [results/raw-recall-diagnostic.md](../results/raw-recall-diagnostic.md).

## Done — the Docker image, actually built ✅ 2026-08-15

The README had promised a two-minute start for a Dockerfile that had never been
built. Building and running it found three defects, all of them already public:

1. **Search returned 500.** The image omitted the `embed` extra on the reasoning
   that search "only needs an encoder for the query" — which is the entire read
   path. Fixed by including it by default.
2. **`/healthz` reported `ok` while search was broken.** Worse than no health check:
   it tells an orchestrator to route traffic to a container that cannot answer.
   Health now reports `degraded` with `search_available` and a remedy, still 200
   because browse and timeline do work.
3. **`ARG EXTRAS` was never used** — the `RUN` hard-coded its extras, so overriding
   the build arg did nothing at all.

A missing encoder now raises a typed `EncoderUnavailable`, mapped to **503 with the
command that fixes it**, rather than a bare 500.

**Then the image was 24.4GB.** The default torch wheel is the CUDA build, and
`torch.cuda.is_available()` is False inside the container, so the whole GPU stack was
dead weight.

The first explanation for why swapping in the CPU wheel did not fix it was wrong.
Layer ordering looked like the culprit — install then replace, with the old copy
still underneath — but merging the swap into one `RUN` produced a *larger* image
(15.2GB) than doing it in two (9.03GB), which the theory cannot explain. Looking
inside the container gave the real answer:

```
nvidia/  2.9GB      CUDA libraries, orphaned
triton/  649MB      orphaned
torch/   577MB      correctly the CPU build
```

`uv pip install --reinstall torch` swaps torch and leaves its former dependencies
behind, because nothing asks uv to drop packages that are merely no longer required.
The fix is to delete them explicitly, in the same layer.

**24.4GB → 2.95GB**, with `/healthz` ok, search returning ranked memories and the raw
fallback recovering its turn from the built image. The venv is 1.02GB of that, most of
it PyTorch, which the encoder genuinely needs. Getting materially below this means not
shipping PyTorch at all — running the encoder out of process, or converting it — which
changes the embeddings and so cannot be evaluated while a benchmark run is in flight.
It is a deployment optimisation, deliberately queued behind the measurement it would
otherwise invalidate.

Two lessons: a Dockerfile that has not been built is a guess, and a plausible
explanation for a measurement is not a diagnosis — this one survived until the
numbers contradicted it.

## Done — shareable inspector state ✅ 2026-08-15

State is split by who needs to see it:

* **URL** — `namespace` and `q`. Anything identifying *what is shown* lives here, so
  a view survives a reload and pastes into someone else's browser.
  `history.replaceState` updates it without navigating (a reload would restart the
  search that just finished).
* **localStorage** — chat transcript and namespace. Personal, and meaningless in
  another browser.
* **Neither** — memories and raw turns. They are live data; a browser copy goes
  stale against the store and shows a state the backend no longer holds.

Fixed demo routes for the README: `?demo=mayo` (structured memory dropped the URL,
the archive still has it), `?demo=timeline` (one key, five values over time),
`?demo=collectibles` (a multi-valued key whose facts coexist). Each rewrites itself
into a plain shareable link on load. Plus a **Copy view link** button, so a bug
report can be a URL instead of a list of steps.

Restored searches go through the same request binding as typed ones, verified by
firing a user search immediately after a restore and asserting the panel shows the
later query.

**A bug worth recording.** `const history = [...messages]` inside the search
function shadowed `window.history`, so `history.replaceState()` became a method
lookup on an Array and blanked the panel with a TypeError — while the chat pane kept
rendering from localStorage, which made it look like a backend failure. Now asserted
against in `test_the_inspector_page_carries_its_state_in_the_url`.

## Done — recorded golden runs + optional live answering ✅ 2026-08-15

A demo page has two bad options and one good one. Answering on every page load spends
quota on every visitor and every refresh. Hard-coding an answer is a mock wearing a
result's clothes. The third is to **record a real run, label it as recorded, and offer
to re-run it**.

Default view for `?demo=mayo`:

```
ANSWER  [recorded run — not executed on page load]
  "…titled 'How to Sit Properly at a Desk to Avoid Back Pain', and you can
   watch it here: https://www.youtube.com/watch?v=UfOvNlX9Hh0"

  Structured memory            insufficient
  Raw conversation fallback    triggered — archive_wide

  recorded 2026-08-15 · 3 run(s) agreed · judge PASS ·
  answerer memory-aware-v2 · judge lme-type-aware-v2
```

Recordings come from the live regression artifacts, not from a hand-written fixture —
`results/golden-runs.json` is generated from
`results/raw/two_stage_fallback.live_v2_run*.jsonl`.

**Staleness is the risk this creates, and it is checked.** Each recording stores a
fingerprint over the namespace, query, the three prompt/extractor versions, and the
store's state (row count, latest ingest, extractor stamp). `GET /v1/golden/{name}`
recomputes it and returns `is_current`; a mismatch renders a red **STALE** badge and
a line telling the reader to re-run. Verified by tampering with a stored fingerprint
and asserting the API reports `is_current: false` while still returning the answer —
labelled, not hidden.

**Live answering is one click, never automatic.** `POST /v1/answer` runs the real
path through the same `MemoryRunner` the benchmark harness drives, so the demo and
the measurements cannot drift. Measured on a click: **3,782 ms, 2 answerer calls, 20
memories of 82 candidates, 970 context tokens**, evidence
`answer_sharegpt_81riySf_0:1` — the gold turn, recovered live. Without a credential
the button reports that plainly instead of appearing broken.

Two tests guard the property that matters: one asserts the `/v1/answer` call appears
only inside the click handler, the other that the recorded badge exists — an
unlabelled recording is the failure mode.

**Two real bugs surfaced building this.** `load_runs(path=GOLDEN_PATH)` bound the
constant at definition time, so the path could never be redirected by a test or a
deployment. And `lifespan` closed *any* service including an injected one while
leaving the global pointing at it, so a second startup in one process used a closed
store — it now closes only what it created.

## Done — raw-conversation fallback ✅ 2026-08-15

**Memory first, source when needed.** Structured memory is a lossy compression; the
pilot showed the compression is usually sufficient (`two_stage` matched naive RAG at
a fiftieth of the context) but the failures have a shape — extraction keeps the gist
and drops the artifact. *"The assistant recommended a Mayo Clinic resource"* is true
and cannot answer *"what was the URL?"*.

Attaching raw evidence to every answer was already measured: `two_stage_hydrated`
tripled context for no detectable gain, which is a slow slide back into naive RAG.
So recovery is **conditional**, and the answerer decides:

```
first call -> AnswerVerdict{status: answer | need_source | no_evidence}
   answer       -> done, one LLM call
   need_source  -> source-local: the turns those memories came from
   no_evidence  -> archive-wide: BM25 over every turn in the namespace
                -> nothing found: decline, and stay declined
```

Two levels, cheapest first. Source-local is precise because retrieval already found
the right memory and only the detail is missing; archive-wide exists because
extraction can miss a fact entirely.

**Verified against a real failure.** Question `41275add` ("the Mayo Clinic video you
recommended") is one of the four `single-session-assistant` questions every variant
got wrong. Structured memory holds nothing relevant; the archive search returns the
source turn containing `youtube.com/watch?v=UfOvNlX9Hh0` — the gold answer.

Shipped: `retrieve/fallback.py`, `turns_fts` + triggers, `store.turns_for_memories`
and `store.search_turns`, `AnswerVerdict`, a `two_stage_fallback` variant,
`POST /v1/raw/search`, and the inspector's fallback panel. Off by default
(`fallback.enabled`), because it changes what a variant is and must be an ablation
row rather than a silent upgrade.

**An indexing trap worth recording.** `turns_fts` is an FTS5 *external content*
table, so `SELECT count(*)` reads the underlying `turns` table and reports full
coverage even when nothing has been indexed. Detecting "needs backfill" that way
always said "already built", which would have left every pre-existing store
searching an empty archive — and "no evidence" is indistinguishable from "never
indexed" from the outside. Now marked explicitly in `meta`, with a test that empties
the index and asserts reopening rebuilds it.

**Still to do:** the four preference/assistant cases are regression-tested against
scripted models, not a live one — the end-to-end run needs answerer quota. The
inspector shows the fallback layer but does not yet generate answers.

## Done — result versioning ✅ 2026-08-14

Every result row now records what produced it:

```json
{"answer_prompt_version": "memory-aware-v2",
 "judge_prompt_version":  "lme-type-aware-v2",
 "extractor_version":     "two-stage-p10-v2"}
```

Two design points that matter more than the fields themselves:

* **The extractor version comes from the store, not the checkout.** It is written
  into the store's `meta` table at ingest time and read back at eval time, because
  it describes *the data being evaluated*. A store built last month must not be
  relabelled by today's working tree.
* **The default is `None`, not a guessed version.** Rows written before stamping
  genuinely do not know which prompt produced them, and saying so is information.
  The frozen v1 artifacts keep loading and keep reporting `None`.

## Done — P8 inspector MVP ✅ 2026-08-14

`GET /` serves a single self-contained page — no build step, no framework, a client
of the documented endpoints only, so anything it displays an integrator can fetch.

It answers **"why was this memory used?"** by splitting the retrieval into three
groups:

* **Used** — grouped by `scope`, each with its five normalized signals drawn as
  bars, and an expandable panel showing the source turn with the extracted span
  highlighted, plus the supersession chain for that key.
* **Not used — no longer true** — superseded facts, naming what replaced them.
* **Not used — ranked below the cut** — still-true facts that lost on score. This is
  the half that indicts the ranking rather than the timeline, and it needed a new
  `explain=true` flag on search to expose.

Verified live against the real 4,843-memory store: a query returns 20 of 79
candidates, each traceable to a session and turn.

**A bug the UI caught in its first minute, and then a bug in the UI itself.** The
timeline first rendered "↓ superseded" between every pair of entries in a key. On
`collectibles` that produced a fake history — G.I. Joe *replaced by* stamps
*replaced by* a Mickey Mantle card — which looked convincingly like a data defect
until the store showed all eight rows `active` with no `superseded_by`. Most
predicates are multi-valued; the arrow now requires an entry to actually name the
next as its replacement, and a multi-valued key says so instead. Pinned by
`test_timeline_marks_which_entries_were_actually_replaced`.

The static file is served `Cache-Control: no-store`, after a stale cached copy made
a fixed renderer appear unchanged.

**Still to do for P8:** the chat pane is search-only — it shows which memories a
question retrieves, but does not yet generate an answer from them (that needs
answerer quota), and there is no write path in the UI.

---

## Done — P7 read path ✅ 2026-08-14

`src/llm_long_term_memory/api/`, three layers:

- **`service.py`** — the domain, no HTTP. One composition root owning store, index,
  encoder and retriever for the process lifetime. The MCP server and the inspector
  will call this object, not the HTTP API, so none of them grows a second copy of
  the logic.
- **`models.py`** — wire models, deliberately not the domain dataclasses. `Memory`
  carries internal strength bookkeeping the wire has no business seeing, and lacks
  the one field a client most needs: why a memory was *not* returned.
- **`app.py`** — HTTP translation only.

Endpoints: `/healthz`, `/v1/config`, `POST /v1/memories/search`, `GET /v1/memories`,
`GET /v1/memories/{id}`, `GET /v1/timeline`, `POST /v1/messages`,
`DELETE /v1/memories/{id}`.

**14 contract tests over a temp store with a stub encoder** — no API key, no
network, no torch, so CI can run them. The load-bearing cases are negative:
cross-namespace reads return **404, not 403** (403 would confirm an id exists), and
a missing `user_id` is refused before any store access.

`rejected` is computed from each returned memory's own supersession chain, not from
the retrieval trace — temporal filtering removes superseded rows before they become
candidates, so they are not in the trace at all. Walking the chain also scopes the
answer usefully: *"you are seeing Sydney because Canberra was replaced by it"*.

**Still to do for P7:** structured request logging, Docker + compose, and the write
path end-to-end (it returns 503 without an extractor, which is correct but untested
against a live one).

---

## P7 — REST API ★★★★★

The service boundary. Everything downstream (web demo, MCP, inspector) is a client
of this, so nothing else starts until the contract is stable.

```
POST   /v1/messages            ingest a turn, return the memories it created
POST   /v1/memories/search     retrieve, with signals and provenance
GET    /v1/memories            browse one namespace, filter by type/status/scope
GET    /v1/memories/{id}       one memory, with its source turn and timeline
DELETE /v1/memories/{id}       forget
GET    /v1/timeline            supersession chains for a (subject, predicate)
GET    /healthz  GET /v1/config
```

Shape of the two that matter:

```jsonc
// POST /v1/messages
{"user_id": "michelle", "role": "user",
 "content": "I stopped drinking coffee last month."}
->
{"memories": [{"id": "m_8f2a", "subject": "user", "predicate": "coffee_consumption",
               "object": "stopped", "update_op": "replaces",
               "supersedes": "m_41c9", "valid_from": "2026-07-14",
               "source": {"session_id": "s_12", "turn_index": 4,
                          "char_start": 0, "char_end": 41}}],
 "usage": {"extractor_calls": 1, "input_tokens": 312, "output_tokens": 88}}

// POST /v1/memories/search
{"user_id": "michelle", "query": "What does she drink?"}
->
{"memories": [{"id": "m_8f2a", "content": "The user stopped drinking coffee...",
               "score": 0.81,
               "signals": {"semantic": 0.9, "bm25": 0.4, "recency": 0.7,
                           "importance": 0.5, "entity": 0.0},
               "status": "active", "valid_from": "2026-07-14", "valid_to": null,
               "source": {...}}],
 "rejected": [{"id": "m_41c9", "reason": "superseded", "superseded_by": "m_8f2a"}]}
```

**`rejected` is not padding.** Showing *why* a memory was not returned — superseded,
below the budget, outside the namespace — is the thing no competitor surfaces, and
it is what the inspector renders.

Build order: composition root → read endpoints → write endpoint → structured
logging → Docker. The original contracts are in [P7_P8_PLAN.md](P7_P8_PLAN.md), now historical.

**The service default is `two_stage`**, pinned by commit SHA in a release manifest.

---

## P8 — Web chat + Memory Inspector ★★★★★

The thing that makes the project legible in 30 seconds. Two panes:

**Left — chat across sessions.** The scripted walkthrough is the demo: state a fact
in session 1, start a new session, ask a question that requires it. Then *change*
the fact and watch the timeline fork.

**Right — the inspector.** Memories grouped by scope (profile / plans / current
state / history), each expandable to its source turn with the matched span
highlighted, and superseded facts shown as a chain rather than deleted:

```
lives_in:  Canberra  ──replaced 2026-08-14──>  Sydney
           source: session 17, turn 3
           "I'm moving to Sydney next month."
```

This visualises provenance, temporal resolution, and supersession at once. It is
LLTM's actual differentiator, and today it is only visible in terminal logs.

Built against the P7 API, never a second direct store implementation. Ships with a
seeded offline walkthrough so it runs without an API key.

---

## P9 — MCP server ★★★★★

Natural fit, and the shortest path to "someone else's agent uses this".

```
remember(user_id, content, role)     search_memory(user_id, query, k)
get_memory(user_id, memory_id)       forget_memory(user_id, memory_id)
get_timeline(user_id, subject, predicate)
```

Reuses the P7 request models. Compact results by default, with an opt-in inspector
payload carrying scores, provenance, and rejections.

---

## P10 — Speaker-agnostic memory ★★★★ (code shipped; needs re-ingestion)

The [assistant gap](../results/assistant-gap.md) is a product bug, not a benchmark
artifact. "What time did you say the museum closes?" must work.

**The load-bearing fix is separating `source_role` from `subject`.** Who said it and
who it is about are different questions, and the old code conflated them in one
line:

```python
subject = "assistant" if fact.lower().startswith("the assistant") else "user"
```

That made `subject` a speaker flag with two possible values, so a fact about any
third party was filed under `user`. "Andy wore a blue shirt", said by the user, was
stored as a fact about the user — and no amount of prompt tuning could have fixed it,
because the schema had nowhere else to put it.

**Shipped:**

- `Memory.source_role` (`user | assistant | system`) and `Memory.scope`
  (`profile | preference | plan | recommendation | commitment | event |
  shared_context`), with SQL columns and an additive migration.
- **The migration recovers `source_role` exactly on existing stores.** The old
  extractor set `subject='assistant'` precisely when the fact began "The assistant",
  so that is a derivation rather than a guess — verified against the real store: 326
  assistant memories recovered, 4,517 user, zero mismatches, 4,843 rows preserved.
  `scope` stays NULL on old rows because it is genuinely unknown.
- **Stage A now emits `source_role`, `subject` and `scope`.** It has to: Stage B
  sees fact strings with no conversation attached, so speaker identity is
  unrecoverable after Stage A. An unrecognised `scope` is dropped rather than
  stored, because a made-up value is indistinguishable from a real one once it is
  in a filterable column.
- **Prompt rewritten from "what did the user say about themselves" to "what might
  need to be referred back to later, regardless of who said it"**, with the four
  worked examples the old prompt lacked: detail survival, an assistant-supplied
  recommendation, a third-party subject, and a negative example. Explicit rule that
  URLs, product names, figures and titles must never be dropped for appearing inside
  a list — the specific behaviour that lost Mod Podge and the Mayo Clinic link.
- Tests pinning speaker/subject independence, scope validation, and defaults.

**Still to do:** re-ingest (~600 extractor requests, a full quota day), then extend
the coverage gate to assistant-answer questions — it currently cannot see this bug
at all — and re-run A2 on the new store.

---

## P11 — Image memory ★★★

Deferred as an experiment, justified as a **showcase**. "This is my dog Coco" in
session 1, "what was my dog called?" in session 2, with the original photo in the
inspector, demonstrates more in five seconds than any table.

Caption-then-index keeps the whole downstream pipeline unchanged; the schema work is
the image anchor (`attachment_id` + optional bbox) replacing character spans. The
MemLens / Mem-Gallery evaluation stays available if it is ever wanted, but it is no
longer the reason to build this.

---

## Demoted to optimisations

Real work, no longer milestones. Each needs a measured reason to be picked up:

- **Cross-encoder Pareto — measured, and the answer is no.** The sweep ran
  (`final_k ∈ {20,10,5} × rerank ∈ {off,on}`, 31 questions). At k=10 the reranked
  and plain arms answered **every question identically** while reranking *lowered*
  source recall at every k. It stays disabled; see
  [results/rerank-pareto.md](../results/rerank-pareto.md). The sweep's real finding
  is elsewhere: context falls 67% from k=20 to k=5 with no distinguishable accuracy
  cost, and k=10 matches `naive_rag` at **54x less context**.
- **Conditional hydration.** `never` (48.4% / 439) and `always` (51.6% / 1,318) are
  measured; the target is ~51% at 600–800 tokens. Needs the failure audit first, to
  know *which* question types actually require raw evidence.
- **Full R/E/H/A/J/S failure attribution** on the 50, with `S` for storage/schema
  policy — the class the assistant gap belongs to, which is neither retrieval nor
  extraction.
- **A4 held-out run.** Still the only clean number the project will ever have. Run
  once, pre-registered, after the product default is frozen.

---

## Schedule, against the quota that actually binds

The free tier caps **requests per day, per model**: extractor and answerer 500 RPD,
judge 1,500 RPD. Measured ingest rate is **4.0 sessions per request** (namespace
boundaries make most batches partial), so a full dev-50 ingest is ~600 requests —
**more than one quota day**. Evaluation is ~2 requests per question (one answerer,
one judge) and draws on different pools, so **eval and ingest do not compete**.

| Day | Extractor (500/day) | Answerer + judge | Ships |
|---|---|---|---|
| **1** | finish ingest, 872 sessions ≈ 220 req | A2 on `dev50`: resume `two_stage` + `two_stage_hydrated`, 19 questions each ≈ 76 req | gates archived, **A2 table** |
| **2** | — | preference 3-question hand audit (no API) | P7 skeleton: composition root, `/healthz`, `/v1/config`, read endpoints |
| **3** | — | — | P7 write path + structured logging + Docker |
| **4** | — | — | P8 inspector against the API, seeded offline walkthrough |
| **5** | — | — | P8 chat pane; P9 MCP tools |
| **6** | P10 re-ingest ≈ 600 req (**a full quota day**) | — | `memory_scope` schema + prompt fix + gate |
| **7** | — | A2 re-run on the P10 store ≈ 100 req | assistant-gap result, README rewrite |

Two scheduling facts worth stating plainly:

- **Day 1 is the only day that needs both pools.** Ingest and eval can run
  concurrently because they draw on different models — but never run two ingests at
  once: there is no lock, and they overwrite each other's entries in the shared
  quota file (that cost ~50 requests on 2026-08-14).
- **P7–P9 need no API quota at all.** Days 2–5 are pure engineering, which is why
  the product work is scheduled into the days the benchmark cannot use anyway.

---

## README, rewritten in this order

1. One line: *a persistent memory layer for LLM agents.*
2. A GIF: fact stated in session 1, recalled in session 2, then changed and the
   timeline forking.
3. Quickstart — Docker, then MCP.
4. What makes it different: every memory traces to its source turn; changed facts
   supersede rather than pile up; retrieval is explainable signal by signal.
5. **Then** the benchmark — as evidence the design works, not as the product:
   *matches naive RAG's accuracy on LongMemEval-S using ~30x less context.*

---

## Not doing

Postgres/pgvector, multi-writer SQLite, auth and billing, hosted deployment, graph
traversal, SFT of the extractor, more retrieval signals, rank fusion. None is
justified by current evidence, and each competes with getting the product visible.
