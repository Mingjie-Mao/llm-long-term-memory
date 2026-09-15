# BEAM — pre-registration

> **Signed 2026-09-15 by Mingjie Mao, the project owner, before any provider call on BEAM.**
> Nothing below may be edited. An amendment is a new dated section appended to this file,
> naming what changed and why, and it may not be written to match a result already seen.
>
> Ingestion of the development half began under this registration on 2026-09-15.

LongMemEval-S is spent: all 500 questions are read, and by
[the data protocol](data-protocol.md) every number after v2 is a development number. BEAM
([Hugging Face](https://huggingface.co/datasets/Mohammadta/BEAM), CC BY-SA 4.0, revision
`3205395e`) is the outside data that makes a conclusion possible again. This registers what
will be measured on it, before anything is.

## What this measures, and what it does not

It measures whether a memory system's answers to questions about a long conversation are
better than the baselines', on a benchmark whose questions, gold and rubrics this project
did not write.

It does not produce a number comparable with BEAM's published tables. Two departures from
BEAM's own protocol were made to fit a daily request quota, both registered in
[`beam_judge.py`](../src/llm_long_term_memory/evaluation/beam_judge.py): every rubric item of
a question is graded in one call rather than one call per item, and event order is read from
positions the judge reports rather than from a pairwise alignment call per candidate pair.
Every figure here compares this project's arms with each other.

## The data

| | |
|---|---|
| scales | 100K and 500K |
| split | whole conversations, seed `20260914`, conversations sharing a seed id on the same side ([`beam-split.json`](manifests/beam-split.json)) |
| development | 22 conversations, 440 questions ([`beam-dev.json`](manifests/beam-dev.json)) |
| final | 33 conversations, 660 questions ([`beam-test.json`](manifests/beam-test.json)), registered by `register_hidden_set.py`, no id, text or near-duplicate overlap with the 500 LongMemEval questions already read |
| unit of analysis | the conversation: its twenty questions are answered from one store, so they are one observation |

**The development half may be read question by question.** The final half is exported only
with `--final-run`, answered once, and never read before that. Every ability contributes
exactly two questions per conversation, so no stratum needs re-balancing.

## The metric

### What the primary covers

**Eight of the ten abilities**: `abstention`, `contradiction_resolution`, `event_ordering`,
`information_extraction`, `knowledge_update`, `multi_session_reasoning`,
`preference_following`, `temporal_reasoning`.

**`instruction_following` and `summarization` are reported separately**, always, beside the
primary.

The primary answers one question and only one: **can the system keep, retrieve and correctly
use the facts a question needs?** Seven of the eight name a fact in every rubric item. The
eighth, `preference_following`, is included because a preference the conversation established
*is* such a fact. `instruction_following` is excluded as a behavioural constraint rather than
a locatable fact.

**That line is a judgement, and a close one.** No measurement this project has separates the
two: the manner-of-writing screen puts them at 4 rubric items of 68 and 3 of 73, and the
coverage analyzer decides 1 of 68 and 3 of 73. So the choice is not defended by data — it is
declared, and then made auditable. Every alternative composition is a fixed secondary,
computed and reported on every arm whatever it says: the 7 fact abilities alone, the
registered 8, the 8 plus `instruction_following`, and all ten. A reader can see exactly what
a different line would have given, which is the only honest way to draw one.

The proposal this replaces kept `instruction_following`, `preference_following` and
`summarization` out of the primary because "how the answer is written moves these as much as
what memory kept". That reason was checkable for nothing and does not survive the check
([`beam-ability-taxonomy.json`](analysis/beam-ability-taxonomy.json), zero provider calls):
every rubric item of the development half was screened for wording that names a manner of
writing, and every match was read once by hand.

| ability | rubric items | screened | manner of writing, after reading |
|---|---:|---:|---:|
| instruction_following | 68 | 4 | **4** |
| preference_following | 73 | 5 | **3** |
| summarization | 234 | 11 | **0** |
| the other seven | 790 | 4 | **0** |

Seven items in 1,165. BEAM's own `instruction_type` label cannot be used instead: it marks 31
of the 44 development instruction-following questions `format_instruction`, and among them
are "always specify practice durations" and "always provide player attendance numbers" —
content, filed under format.

So what these two abilities actually ask is: did the system remember a standing preference or
instruction the user stated in an earlier session, and apply it to a question that does not
mention it? That is long-term memory, and it is why `preference_following` is in. The reason
`instruction_following` is nonetheless out is the one stated above — a behavioural constraint
is not a fact the four evidence layers can trace — and not this screen, which does not
separate them.

`summarization` is separated for a different and stated reason: it is graded as **coverage**.
234 rubric items over 44 development questions, median 5 and up to 12, each a narrative span
rather than a named fact, against an answerer whose system prompt ends "Answer directly and
concisely". Answer *length* is the largest lever on that score and length is not a memory
property. Both arms share the prompt, so a paired comparison on it stays valid — this is about
not letting length dominate the headline. Whether it is also the noisiest of the ten is
**unmeasured**; the noise run reports discordance per ability, and if summarization is no
noisier the separation will have cost nothing.

### How a score is formed

Each rubric item scores 1.0, 0.5 or 0.0 on BEAM's scale. Then, in this order:

    question      the mean of its own rubric items
    conversation  the mean of its questions
    arm           the mean over conversations

Question-level averaging first is not cosmetic. Summarization holds 234 of the development
half's 1,165 items over 44 of its 440 questions; averaging items would hand it five times the
weight its questions earn. Conversation-level averaging second is what keeps twenty
correlated questions from being counted as twenty observations.

**Primary**: mean rubric score over the nine abilities, formed that way.

### The evidence layers — mandatory, on every arm of every run

A single source-recall number cannot decide anything. `recall_stages.selected` is
`bool(evidence & selected_sessions)` — **any** labelled source session hitting the context
makes it true — so LongMemEval's 98.3% against 55% accuracy could not be read as a
43-point reasoning gap, and the 2026-09-13 review says so in those words. BEAM can do
better, because its rubric items name the required fact in words. Four layers, registered
now so that no analysis of them is chosen after a result is seen:

| | layer | asks |
|---|---|---|
| 1 | `any_source_session` | did at least one labelled source session reach the context? |
| 2 | `all_source_sessions` | did **every** labelled source session reach it? |
| 3 | `required_fact_coverage` | did the rubric's own required facts reach the final context? |
| 4 | `answer_utilisation` | of the facts that did reach it, how many did the answer use? |

Together they separate four failures that have until now been one number, and that call
for four different fixes:

    never extracted  ·  extracted but not retrieved  ·  retrieved but dropped from the
    context  ·  in the context and not used

Per question the same walk is reported as a ladder, which is how one failure is read:

    Gold required facts:         8
    In the source turns:         8 / 8
    Extracted:                   6 / 8
    Retrieved:                   6 / 8
    Selected context:            5 / 8
    Mentioned in answer:         4 / 8
    Final rubric score:       0.50

Zero provider calls: every figure is string matching over text already on disk.
`beam_report.py` marks a report **incomplete** when any of the four is absent, because a
metric reported only when someone remembers is not mandatory.

**What the instrument will not do.** A rubric item earns a verdict only when it carries a
distinctive token — money, a percentage, a written date, a number with a unit from a closed
list, or a multi-word proper noun. A bare number does not qualify: "26" occurs somewhere in
any 500K-token conversation, and matching on it would manufacture evidence. On the
development half that leaves **261 of 1,165 rubric items decidable (22%)**, concentrated
where answers are numbers and dates — temporal reasoning 66 of 67, knowledge update 36 of
46 — and near zero in instruction following (1 of 68), preference following (3 of 73) and
contradiction resolution (11 of 176), whose rubrics name behaviours rather than facts. The
undecidable share is reported beside every coverage figure and **an undecidable item is
never counted as a loss**.

**The instrument's own ceiling, measured.** BEAM labels which sessions hold each question's
evidence, so a required fact ought to be findable in its own labelled source. It is, for
**191 of 221 evidence-bearing items — 86%**
([`beam-fact-matcher-validation.json`](analysis/beam-fact-matcher-validation.json)). Nothing
read through this matcher is more reliable than that, and the figure is reported with any
result read through it.

**Two of the buckets are not the pipeline's fault**, and both are counted separately:
a required fact absent from its own labelled source (`suspected_annotation_gap`), and a
rubric scored zero on an answer that contains every fact it asked for
(`suspected_judge_error`). MemTrace's taxonomy names them Annotation Error and
LLM-as-a-Judge Error (arXiv:2605.28732); this project has mistaken a metric for a system
failure three times (D30), which is why they are columns rather than assumptions.

**Derived facts are not losses.** A fact absent from the entire conversation was never
stated — it is the computed answer to a duration or aggregation question. 40 development
items are of that kind, 26 of them in temporal reasoning. They are excluded from layers 1
to 3 and kept only for layer 4; counting them as extraction loss would make temporal
reasoning read as a total extraction failure.

**Secondary, all pre-declared and all reported whatever they say**:

1. mean rubric score over all ten abilities
2. mean rubric score over the seven fact abilities — the primary minus the two
   standing-directive ones, so a reader can see whether they carry the result
3. binary verdict (question mean ≥ 0.5), on the primary abilities and on all ten
4. event-ordering Kendall τ-b, rescaled
5. source-session recall
6. a per-ability table of every metric above

Nothing here is chosen after the fact. (2) exists precisely so that the composition decision
above can be audited against the result rather than trusted.

### How a comparison is decided

Paired sign-flip test over conversations, on each conversation's mean difference, two-sided
([`clustered.py`](../src/llm_long_term_memory/evaluation/clustered.py)). A conversation only
one arm finished is dropped whole: half a conversation is a mean over a different question
set, not a paired difference, and on a quota-interrupted run half a conversation is the
normal state.

Every figure is recomputed from the per-item grades saved on each row, so changing how they
are aggregated never costs a judge call.

## The instrument's own noise

[The gate protocol](gate-protocol.md) forbids registering a threshold against a noise floor
nobody measured, and nothing has been measured on BEAM. So **no gate is registered here.**
The numbers in [`beam-power.json`](analysis/beam-power.json) are scenario estimates carried
over from LongMemEval `heldout100` (4% run-to-run discordance) and the v4 probe pair (19%);
they size the plan and decide nothing.

The noise run measures it, in three passes, because run-to-run difference has two sources and
separating them costs one extra judging pass rather than a whole answering pass:

| pass | what runs | what it costs | what it isolates |
|---|---|---:|---|
| A | answer + judge | 590 + 440 | the measurement itself |
| B | judge again, over pass A's saved answers | 440 | **judge sampling alone** |
| C | answer + judge again | 590 + 440 | answerer + judge together |

B minus A is the judge; C minus A is both. If the judge dominates, repeats are cheap — a
saved answer can be re-graded any number of times. If the answerer dominates, only repeated
answering helps, and that is the expensive case worth knowing about before committing to it.

Reported from it: per-question and per-conversation discordance, the standard deviation of a
conversation's mean primary score across passes, the intra-conversation correlation, and all
of these **per ability**. The gate for any candidate comparison is registered in an addendum
to this file afterwards, computed from those numbers, and not before.

**Pass C may be deferred.** It is needed before the first candidate comparison, not before
the first measurement, and deferring it is how the first spend stays inside two quota days.

## The order of runs

```
ingest dev with frozen v2  ->  pass A  ->  zero-cost stratification  ->  pass B, pass C
   ->  one candidate, one variable  ->  gate addendum  ->  dev comparison  ->  freeze  ->  final run
```

- Baselines — whole history and naive RAG — **are not run on the development half**. They are
  run once, at the end, on the final half, and only at 100K where a whole history fits.
- The final half is answered **once**, with all arms in that single run
  ([roadmap](../docs/ROADMAP.md) 4.4). It is not brought forward for any reason.
- A candidate that can be falsified offline is falsified offline first. Retrieval is
  deterministic given the store, so most retrieval claims cost nothing to test
  ([`QUOTA_DISCIPLINE.md`](../docs/QUOTA_DISCIPLINE.md) step 2).

## Budget

Counted, not estimated, unless marked otherwise. Free tier: 500 requests per model per day.

| stage | requests | basis |
|---|---:|---|
| dev ingest, extraction | **512** | 3,679 session chunks, 15 per batch, 256 namespace-bounded batches, two stages — counted from the real batching |
| dev ingest, Stage A input | **≈8.0M tokens** | the conversation text itself, counted |
| dev ingest, deduplication | **not predictable** | see below |
| answering, one pass | ≈590 | 440 questions at LongMemEval's measured 1.34 requests per question, the raw-source fallback included |
| judging, one pass | 440 | one call per question |

**Deduplication is the term that cannot be counted in advance,** and it is the one that has
already broken an estimate. On the v2 `dev100` ingest, adjudication *overtook* extraction —
236 extraction requests against 264 adjudications in a single day, 576 against 421
cumulatively — because dedup compares each new fact against what the store already holds and
a denser store surfaces more candidates ([v2-progress.md](v2-progress.md)). The ingest is
checkpointed per batch and resumes by id, so the cost of being wrong here is a day, not a run;
the run report records the measured adjudication rate on day one and the remaining budget is
re-derived from it rather than from this table.

**Stop conditions**, registered:

- The ingest stops at the daily quota and resumes; that is the normal path, not a failure.
- A judge reply that does not cover the rubric exactly once on the 1.0/0.5/0.0 scale is
  refused rather than repaired, which stops the run. Rows already bought survive and resume
  does not re-buy them — rehearsed, see below.
- If the primary's per-conversation spread across noise passes is wider than the effect any
  plausible candidate could produce, the candidate comparison is **not run**. A gate the
  instrument cannot read is a coin flip wearing a threshold.

## Resolvability

`tools/check_gate_is_resolvable.py` reads the v4 probe pair's measured discordance and cannot
speak for BEAM, whose instrument has never been run. **No gate is registered, so there is
nothing for it to refuse.** What stands in its place until the noise run:

| stratum | n | cluster | MDE at 2σ, 4% discordance | at 19% |
|---|---:|---:|---:|---:|
| dev, the 9 primary abilities | 396 | 18 | 2.0 – 3.8 pts | 4.4 – 8.3 pts |
| dev, one ability | 44 | 2 | 6.0 – 6.5 pts | 13.1 – 14.1 pts |
| test, the 9 primary abilities | 594 | 18 | 1.6 – 3.1 pts | 3.6 – 6.7 pts |
| test, 100K only, where whole history fits | 240 | 20 | 2.6 – 5.1 pts | 5.6 – 11.0 pts |

Ranges span intra-conversation correlation 0 to 0.15. Units are accuracy points at the binary
verdict; the primary is a continuous score, whose spread the noise run measures directly. The
per-ability row is the important one: **no single ability can carry a conclusion**, whatever
it shows, and none will be read as one.

## Three rules locked before the first request

Locked here because each is a knob that could otherwise be turned after seeing a result.

**1. `ranked_memory_ids` completeness.** Every memory retrieval ranked, before the context
budget chose among them, is written to the row. Without it "retrieval never found the fact"
and "retrieval found it and composition dropped it" are the same row, and the distinction is
unrecoverable once the run is paid for. A run whose rows lack the field on any question is
not a valid input to the evidence layers.

**2. Denominators and the undecidable rule.** A rubric item is *decidable* only when it
carries a distinctive token: money, a percentage, a written date, a number with a unit from
the closed list in `beam_coverage.UNITS`, or a multi-word proper noun. A bare number never
qualifies. An item that is not decidable is reported as `undecidable` and **is never counted
in any numerator or denominator of a loss**. An item whose tokens appear nowhere in the whole
conversation is `derived` — the computed answer to a duration or aggregation question — and
is excluded from layers 1 to 3, entering only layer 4. Negative requirements ("should avoid")
are set aside entirely. These four classes partition every rubric item and the counts of all
four are published with every figure.

**3. First-loss classification.** Stages are ordered `source → extracted → retrieved →
context → answer`. A question's first loss is the earliest stage holding fewer required
tokens than the stage before it. Two outcomes are *not* pipeline failures and are counted
separately: a fact absent from its own labelled source (`suspected_annotation_gap`, MemTrace's
Annotation Error) and a zero rubric score on an answer carrying every fact the rubric asked
for (`suspected_judge_error`, its LLM-as-a-Judge Error).

**4. How these numbers may be written down.** The matcher decides about a fifth of rubric
items, so a figure from it is quoted **only** as a share of automatically decidable
evidence-bearing rubric items — never as a share of BEAM questions or of all rubric items.
`tools/beam_fact_coverage.py` emits the sentence itself, with its denominator filled in, so
the qualification travels with the number:

> Among automatically decidable evidence-bearing rubric items (N of M graded items on these
> questions), the most common first loss was …

"Extraction loss = x% on BEAM" is not a sentence this instrument can produce, and it is the
shape of claim this project has already had to retract once.

## Before the first paid request

The whole path was rehearsed with a fake provider — BEAM export, ingestion, store, retrieval,
answering, the raw-source fallback, judging, rows, and the aggregate report — over one 100K
and one 500K conversation, 245 session chunks, 40 questions, **zero provider calls**
([`beam-dry-run.json`](analysis/beam-dry-run.json), `tools/beam_dry_run.py`). Both ways a long
run stops were drilled: a daily-quota stop returns a partial report and resumes without
re-buying an answer, and an off-scale judge reply stops the run with its bought rows intact.

It found one defect, and it blocks the ingest:

> **Deduplication is not namespace-scoped.** Retrieval searches the shared index and then
> keeps only `memory.user_id == namespace`; `Deduplicator._neighbours` searches the same index
> and adjudicates whatever comes back, with no such filter. So a fact extracted from one
> conversation can be adjudicated against — and dropped as a `DUPLICATE` of — a fact belonging
> to a different one.
>
> Measured on the frozen `test100` store, 12,471 memories across 100 namespaces
> ([`dedup-namespace-leak.json`](analysis/dedup-namespace-leak.json)): of the neighbours above
> the 0.92 threshold reachable inside dedup's three-hit window, **42 crossed a namespace and
> 3 did not**. Dedup only ever looks at three index hits, so foreign memories also crowd out
> the within-namespace duplicates it exists to catch.
>
> BEAM makes this worse rather than better. The split deliberately keeps conversations
> generated from one seed on the same side, so a 100K conversation and its 500K twin are
> ingested into one index — and the rehearsal's two conversations were exactly such a pair.

### Is fixing it a measurement fix or a system change? — measured

A fix that changes only bookkeeping can be applied to the frozen system and recorded as an
amendment; one that changes what reaches the model is a second arm, and calling it `v2`
afterwards is what freezing exists to prevent. That is not a matter of opinion, so it was
measured: the same two conversations ingested four times with a fake provider — frozen and
namespace-scoped dedup, each under the rehearsal's own verdict rule and under a worst case
that forces every cross-conversation pair to `DUPLICATE`
([`dedup-fix-behavioral-diff.json`](analysis/dedup-fix-behavioral-diff.json), zero calls).

| | memories written | memories the fix saves | questions with different `ranked_memory_ids` | questions with a different context |
|---|---:|---:|---:|---:|
| benign verdicts | 914 → 915 | 1 | **0 of 40** | **0 of 40** |
| every cross-pair forced to DUPLICATE | 909 → 915 | 6 | **0 of 40** | **0 of 40** |

**The store changes; what reached the answerer did not.** Every saved memory belonged to
`beam-500K-8`, the seed twin of the other conversation, and none of them was retrieved for any
of that conversation's twenty questions on this slice.

So it is a **system change by the strict rule** — which memories are deduplicated differs —
with a measured blast radius of zero on what the model saw, over 40 questions. The dry run
cannot settle more than that: whether a real model calls a cross-conversation pair a duplicate
is a model decision, and the frozen store was built by a model whose pairwise verdicts were
never recorded.

**Decided: option C.** The tool exists — `tools/dual_policy_ingest.py`, rehearsed with a
fake provider over the same two conversations, reproducing the diff above exactly (914 → 915
memories, the same conversation) with **34 extraction requests in the recording pass and 0 in
the replay**. Extraction runs once and is replayed from a per-batch cache; deduplication runs
twice, because a dropped memory changes the store and therefore every later neighbour search,
and pretending otherwise would build a second store no run could have produced. The wrappers
fingerprint as the extractor they wrap, so neither store claims to have been written by
something that does not exist, and a batch missing from the cache is refused rather than
quietly re-extracted.

| | what runs | extraction requests | what may be claimed |
|---|---|---:|---|
| A. two arms | ingest twice, frozen and fixed | 1,024 | "frozen v2 on BEAM", plus the fix as a registered one-variable comparison |
| B. one arm, amended | ingest once with the fix | 512 | "v2 with one recorded repair", diff attached |
| **C. dual-policy ingest** | one extraction pass, replayed under both policies | **512** | **both stores, for the price of one — chosen** |

C buys what A buys at B's extraction cost, which is the expensive and predictable half; only
adjudication runs twice, and that was always the unpredictable term. The registered arm stays
the frozen policy, so "the frozen v2, measured on outside data" is still the sentence, and the
namespace-scoped store sits beside it as a one-variable comparison that cost one extra dedup
pass rather than a second ingest. If the fix turns out to change nothing measurable, that is a
result too, and it will have cost almost nothing to learn.

## What needs a signature

**Signed.** Composition, aggregation, the four evidence
layers, the noise design, the three locked rules, the deduplication decision and the budget are
all fixed above. Every tool the plan depends on is built and rehearsed against a fake provider:
the dry run, the fact-coverage analyzer, the judge-only pass, the dual-policy ingest. Nothing
here may be edited once the first request is sent; an amendment is a new dated section.

Both tools the plan depends on are built and rehearsed, not planned. The evidence-layer
analyzer ran inside the fake-provider dry run over its 40 questions and produced all four
layers and nine per-question ladders ([`beam-dry-run.json`](analysis/beam-dry-run.json)); one
field had to be added to the rows first — `notes.ranked_memory_ids` — and it is additive, with
no prompt, model or behaviour change. `tools/rejudge.py` performs pass B: it re-grades answers
already bought without touching the store or the answerer, records for every grading the
answer it graded, the judge's model and prompt version, a hash of the exact prompt, temperature,
seed, the raw reply and the parsed per-item grades, resumes by answer id, refuses to overwrite
the answers it reads, and requires `BEAM_PREREG` to spend anything. Its `--summarise` mode
reports verdict discordance and score drift, overall and per ability. Two rehearsed passes over
the dry run's 40 answers disagreed, so the variance arithmetic is exercised rather than merely
run.

Nothing in this file may be edited once it is signed and the first request is sent. An
amendment is a new dated section, not a rewrite.

---

## Amendment 1 — 2026-09-15: two referenced documents moved

Recorded as an appended amendment rather than an edit above, because a signed
registration is not rewritten, even to repair a link. Nothing about the measurement
changes; only where two of its referents live.

`docs/ROADMAP.md` and `docs/QUOTA_DISCIPLINE.md` were folded into
[the project report](../docs/PROJECT_REPORT.zh-CN.md) when the project's prose was
consolidated to two documents. The links above still name the old paths, which is what
they said when this was signed. Read them as:

| link above | now |
|---|---|
| `../docs/ROADMAP.md` | the report's **后续完整计划** section |
| `../docs/QUOTA_DISCIPLINE.md` | the report's **花配额之前** section |

Both documents remain in git history at their original paths, so the registration can be
read exactly as it stood on the day it was signed.
