# Pre-registration — session-coherent context on `dev100`

**Written 2026-08-20, before `train150` or `dev100` has been ingested.** Neither
store exists; no arm of this experiment has been run at any scale. The five-question
probe that motivates it ran on `heldout100`, which was already spent.

Development happens on `train150`, where individual failures may be read. The
decision happens on `dev100`, where only the endpoints below may be looked at.
That boundary is [the data protocol](data-protocol.md) and this experiment is the
first to run under it.

---

## The question

Retrieval selects twenty memories by score and hands them to the answerer as a
flat list. [context-arms.md](context-arms.md) found that on five held-out
questions, presenting the same information as **the gold sessions' memories in
event order** changed the behaviour — and that trimming `top_k` without changing
the shape made it *worse*, so the variable is not how much is supplied.

What it actually bought was agreement, not accuracy:

| arm | cells where three runs agreed |
|---|---:|
| `flat20` | 3 of 5 |
| `flatN` | 3 of 5 |
| `coherent` | **5 of 5** |

Two objections make that unusable as it stands. **`coherent` is an oracle** — it
uses `answer_session_ids`. And five questions chosen for having failed is not a
sample. So:

> **Can retrieval assemble a session-coherent context without the gold labels,
> and does the shape change survive at n=100 with repeats?**

### Why this is likely to be feasible at all

The gold session is already being retrieved. On `heldout100`, **94 of 100**
questions had at least one retrieved memory whose source was a gold session
(`source_session_recalled`), and source-session recall has been 93.5-94.0% across
every store measured. Nothing needs to be found that is not already found. What is
missing is that the gold session's memories arrive scattered among twenty, ranked
against memories from unrelated conversations, rather than as a unit.

That makes the intervention a **composition** change rather than a retrieval one,
which matters because [failure-stages.md](failure-stages.md) put the retrieval
ceiling at zero and this does not contradict it.

---

## Arms

Three, on one store. Only the context handed to the answerer differs — same
memories in the store, same retrieval, same answerer, same judge.

| arm | context |
|---|---|
| `flat20` | **baseline.** `top_k` 20 memories, retrieval rank order |
| `coherent-auto` | retrieve → group the retrieved memories by `source_session_id` → rank sessions by aggregate score → take the top sessions → within each, every memory from that session in event order |
| `coherent-oracle` | the same construction, but the sessions come from `answer_session_ids` |

**`coherent-oracle` is the ceiling and is not shippable.** It exists to split a
null result on `coherent-auto` into its two causes: session *selection* failed, or
session *shape* does not help. Without it a null is uninterpretable, which is the
mistake `ku_oracle` made by having no arm between the oracle and production.

The session budget (how many sessions, and whether a session contributes all its
memories or a window) is a **development decision made on `train150`** and frozen
before `dev100` is touched. It is not swept on `dev100`.

---

## Endpoints

Every arm is run **three times** on all 100 questions. That is the protocol now,
and here it is also the measurement: the effect being chased was first seen as
run-to-run disagreement.

### Primary — accuracy over repeats

Mean of three runs per arm with the range, and a paired McNemar against `flat20`
on **majority-of-three verdicts**. Majority voting is the point: it denoises each
question's verdict before the pairing, so the test sees the arm rather than the
run. A single-run paired comparison at this scale is what
[the protocol](data-protocol.md) now forbids.

### Secondary — agreement, and why it is only secondary

The fraction of questions on which all three runs agree. This is the effect the
probe actually saw, and it is reported as a rate with its count rather than a
test, because **at the observed base rate this design cannot resolve it.**
`heldout100` flipped 3 of 100 between two runs; three runs might disagree on 5-8.
Halving 8 to 4 is a comparison of rare events on 100 questions and will not reach
significance. Registering that now so a null is not later read as "coherence does
not help consistency".

### Free, and run before any answerer quota is spent

Both computable from the store and the retriever with no LLM calls, on `train150`
and `dev100` alike:

| metric | why it gates the rest |
|---|---|
| **session recall @ top-M** | does the gold session appear among the M sessions `coherent-auto` selects? If this is low, `coherent-auto` cannot work and the experiment is a ceiling measurement only |
| context tokens per arm | a coherent session is more memories than a rank cut; if this triples the context it is a slide back toward `naive_rag` |

**If session recall @ top-M is below 80% on `train150`, `coherent-auto` is not
run on `dev100`** — the development work moves to session selection first, and no
validation quota is spent on an arm already known to be starved.

### Pre-declared slices

Reported for these three and no others, chosen because coherence should matter
where an answer spans events rather than sits in one fact:

- `temporal-reasoning` (27 of `dev100`)
- `knowledge-update` (16)
- `multi-session` (27)

Slices are **descriptive only**. At 16-27 questions each with a ±3pp run band,
none can carry a decision — that is what
[the data protocol](data-protocol.md) says and this is the first experiment held
to it.

---

## Decision rule

1. **Adopt `coherent-auto`** if it does not degrade majority-vote accuracy against
   `flat20` and does not increase median context tokens by more than 50%. It does
   not have to *improve* accuracy: the registered hypothesis is that it improves
   consistency at equal accuracy, and demanding an accuracy win would repeat the
   batch pre-registration's original error of setting a bar the design cannot clear.
2. **If `coherent-auto` degrades accuracy but `coherent-oracle` does not**, the
   gap is session selection. Do not adopt; open session ranking as the next work
   item on `train150`, and report the size of the gap as its ceiling.
3. **If both degrade accuracy**, the shape is not the variable. Close P6, and
   record that the five-question probe was noise — which is the outcome the
   probe's own control already warned about.
4. **If context tokens more than double**, do not adopt regardless of accuracy.
   The product claim is 75x less context than the transcript, and a composition
   change that spends that margin is a different product.
5. The session budget is frozen from `train150` before `dev100` runs. If it has
   to move afterwards, that is a new experiment on a new set, not an adjustment.

Implementation note (2026-08-22, still before any `dev100` access): the oracle arm
forces the benchmark's gold session ids but keeps the frozen maximum-session,
window, active/superseded, ordering and total-memory rules.  If extraction produced
no memory for a gold session, the oracle context is empty rather than silently
falling back to a retrieved distractor.  Every result row is labelled
`coherent-oracle`, so this ceiling cannot be mistaken for a product run.

The validation runner now enforces the three arm names above, their order, exactly
three repeats, the canonical 100-question manifest and the registered freeze/store
names. After all arms complete, it applies the decision rule in code and writes a
hash-bound `dev100-decision.json`; there is no manual post-result reinterpretation.

## Registered prediction

- Session recall @ top-3 on `train150` is **above 90%**, since memory-level
  source-session recall has been 93.5-94.0% everywhere it has been measured.
- `coherent-auto` matches `coherent-oracle` within 2 points of accuracy — session
  selection is not the bottleneck, because the sessions are already being found.
- Majority-vote accuracy differs between `flat20` and `coherent-auto` by **less
  than 3 points**, in either direction, and the comparison does not reach
  significance.
- Agreement is **higher for both coherent arms**, by a margin too small to certify
  at n=100.
- Context tokens rise **30-80%**: a whole session's memories against a rank cut of
  twenty, offset by fewer sessions being represented.

If the third and fourth predictions both hold, the honest summary is that this
experiment cost three quota days to establish a direction it could not measure,
and the next version of it needs `test100`-scale n or a slice where the effect is
concentrated — not another arm.

## Dependencies

`train150` and `dev100` both need ingesting first, and `dev100`'s store is the
one the [batch-size experiment](prereg-batch-size.md) builds as its `batch15`
arm. **This runs on that store**, so the two experiments share an ingest and the
order is: ingest `train150` → develop here → ingest `dev100` at `batch15` →
batch arms and context arms both read it.

Running this on a store built at a different batch size would confound the two
pre-registrations, and neither would be interpretable.

---

## Amendment 2 — two reported baselines, added 2026-08-25

**Written before `dev100` was ingested or evaluated.** The partial `dev100` store built
on 2026-08-25 was discarded (see below), so no arm of this experiment has produced a
score.

### What changes

Two arms are appended. The registered arm list becomes, in this order:

| order | arm | runner | role |
|---:|---|---|---|
| 1 | `flat20` | `two_stage_fallback` | paired baseline — **decision arm** |
| 2 | `coherent-auto` | `two_stage_coherent` | candidate — **decision arm** |
| 3 | `coherent-oracle` | `two_stage_coherent_oracle` | ceiling — **decision arm** |
| 4 | `naive_rag` | `naive_rag` | **reported baseline** |
| 5 | `memory-only` | `two_stage_memory_only` | **reported baseline** |

Cost: 5 arms x 3 repeats instead of 3 x 3, about 2,050 answerer calls instead of 1,233.
No extra extractor quota — every arm reads the same store.

### Why

Every baseline number this project has is from `dev50`, a set that was read question by
question for two weeks. The strongest sentence the final table can currently support is
"the system scores X%", and the obvious question — *compared with what, on data nothing
was fitted to?* — has no answer. `test100` already registers `full_context` and
`naive_rag`; adding them one stage earlier means the comparison arrives with a measured
run-to-run band rather than as a single shot.

`memory-only` needs a new runner name. `two_stage` and `two_stage_fallback` both read
`fallback.enabled`, so memory-only was previously reachable only by editing the config,
which a single-config freeze forbids. `two_stage_memory_only` forces the archive off
regardless of config. It is **not** the same measurement as the "answered from memory
alone" split of the fallback arm: in that arm the answerer knows it may ask for source,
which can change what it does.

### What does not change

**The decision rule is untouched.** `select_dev_candidate` computes the product choice
from `flat20`, `coherent-auto` and `coherent-oracle` only; the two baselines are
reported and cannot move it. This is enforced in code and covered by a test that gives a
baseline a higher score than every decision arm and asserts the selection is unmoved.

Everything else stands: three repeats, majority-of-three verdicts, aggregate-only
reading, the pre-declared slices, and the four registered outcomes.

### Why the partial store was discarded

`dev100` was 37% ingested (1,789 / 4,791 sessions, 408 extractor calls) under a
pre-ingest freeze captured while the source tree was uncommitted. Committing that work
moved `git HEAD`, and the freeze compared it, so the store could no longer be shown to
have been produced under the frozen system. The 408 calls are written off. Keeping them
would have meant either an after-the-fact freeze or never committing the code the freeze
describes; both are worse than one quota day.

The enforcement defect itself is fixed: fields whose names end in `_for_reference_only`
are now recorded and not compared, so a commit no longer invalidates a freeze while
every file that affects a result stays hashed individually.
