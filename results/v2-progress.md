# v2 execution log

This is a checkpoint log, not a results table, and it is append-only: earlier entries
describe the state on their own date and are not rewritten when it changes. `train150` is
development data; `dev100` was left untouched until the candidate configuration was frozen,
and `test100` remains sealed until v2 is frozen.

## 2026-08-27 — dev100 at 76.4%, and the resume rate is falling

| item | verified state |
|---|---:|
| terminal sessions | 3,660 / 4,791 |
| successful sessions | 3,073 |
| explicit empty-result sessions | 587 |
| content-policy blocks | 0 |
| memories / index ids / vector rows | 9,569 / 9,569 / 9,569 |
| extraction + adjudication requests | 576 + 421 |
| archived pending sessions | 15 (raw-only, no memories) |
| state checker | `issues: []`, SAFE TO RESUME |

One quota day advanced the store by 1,477 sessions and 3,993 memories, then stopped at
exactly 500/500 with exit 2. The 15 archived pending sessions are the batch the quota
interrupted: they hold no memories and carry no terminal state, so the next run replays
them rather than counting them as empty.

**The store is not drifting from the development set.** Yield is 2.61 memories per
session cumulatively and 2.70 over this day's sessions alone, against train150's 2.6.
The empty rate is 16.0% cumulative and 16.9% over this day's sessions, against
train150's 15.6%. Both sit inside the range train150 established, which is the check
that matters before dev100 is allowed to decide anything.

### The falling rate is adjudication, not extraction

Sessions per request has fallen 4.39 → 3.45 → 2.87 across the three resumes. The cause
is visible in the split: this day spent **236 extraction requests against 264
adjudication requests**, so adjudication has overtaken extraction. Deduplication
compares each new fact against what the store already holds, so its cost rises with
store density — a denser store surfaces more near-duplicate candidates to adjudicate.

This is expected behaviour rather than a fault, but it has a consequence worth
recording before it is discovered the expensive way: **`test100` will cost more per
session than `dev100` did**, and the request estimates in `v2-runbook.md` were written
against the old rate. They should be re-derived from the measured rate, not carried
over.

### The supersession gap reproduces on a third store

| store | `replaces` signals | rows actually superseded | ratio |
|---|---:|---:|---:|
| dev100 (partial) | 431 | 47 | **9.2 : 1** |
| train150 | 833 | 76 | **11.0 : 1** |
| heldout100 (§3.5) | — | — | 14.8 : 1 |

A third independent store shows the same shape, which strengthens the reading in
[REPORT §3.5](../docs/PROJECT_REPORT.zh-CN.md) that this is a second face of extraction loss rather
than a defect in the temporal layer. It still does not separate the two surviving
explanations — the earlier value was never extracted, or the extractor sets the flag on
update-shaped wording regardless of whether a predecessor exists. That separation needs
semantic judgement over the raw archive and has not been run.

Nothing here is a dev100 *result*: these are ingest-process counts, not question scores,
and no arm has been run. The sealed-row rule is untouched.

## 2026-08-26 — dev100 rebuilt from a clean freeze, 45.6% ingested

| item | verified state |
|---|---:|
| terminal sessions | 2,183 / 4,791 |
| successful sessions | 1,845 |
| explicit empty-result sessions | 338 |
| content-policy blocks | 0 |
| memories / index ids / vector rows | 5,576 / 5,576 / 5,576 |
| extraction + adjudication requests | 340 + 157 |
| archived pending sessions | 0 |
| state checker | `issues: []`, SAFE TO RESUME |

The run stopped cleanly at the daily quota with exit code 2 and no non-terminal batch.
Yield matches train150 exactly at 2.6 memories per session, and the empty rate over
substantive sessions is 13.8% against train150's 15.6%.

This is the **second** dev100 store. The first reached 37% (1,789 sessions, 408 extractor
calls) on 2026-08-25 and was discarded — not for anything it measured, but because its
pre-ingest freeze had been captured against an uncommitted tree, and committing that work
moved `git HEAD`, which the freeze compared leaf by leaf. The store could no longer be
shown to have been produced under the frozen system. The calls are written off rather than
re-freezing after the fact. The enforcement defect is fixed: fields whose names end in
`_for_reference_only` are now recorded and not compared, so a commit no longer invalidates
a freeze while every file that affects a result stays hashed individually.

The pre-registration was amended before the rebuild, while no dev100 result existed:
`naive_rag` and `memory-only` join as **reported baselines**, taking the protocol to five
arms x three repeats. They cannot move the product choice — the decision still comes from
`flat20` / `coherent-auto` / `coherent-oracle`, and a test gives a baseline the highest
score and asserts the selection is unmoved. `two_stage_memory_only` was added because
`two_stage` and `two_stage_fallback` both read `fallback.enabled`, so a memory-only arm was
previously unreachable inside a single-config freeze.

## 2026-08-25 — train150 complete, v2 candidate selected and frozen

| item | verified state |
|---|---:|
| terminal sessions | **7,180 / 7,180** |
| successful sessions | 5,927 |
| explicit empty-result sessions | 1,253 |
| content-policy blocks | 0 |
| memories / index ids / vector rows | 18,519 / 18,519 / 18,519 |
| ingest API calls / tokens | 2,592 / 20,844,680 |
| failed API calls | 37 — all transport (23 DNS, 10x 503, 4 disconnects), none content or logic |
| state checker | COMPLETE, `issues: []` |

The final 197 sessions cost 92 calls and added 502 memories, 2.5 per session, consistent
with the 2.6 corpus average.

`finalize_train150.py` then ran with no API calls and produced the final zero-yield audit,
the seven fixed context-grid artifacts, the selection record, and **`configs/v2.yaml`**.
An independent earlier execution of the same finalizer produced a byte-identical config,
which the tool's own guard confirmed by refusing to overwrite a differing one.

**Selected by the fixed rule: `mean` aggregation, window radius 1, cap 30** — Top-3
session recall 95.3%, assembled recall 95.3%, median context 140.5 tokens against 354.5
flat, one question truncated. All three gates pass. Worth recording that `mean-r1-cap40`
ties on both recall figures and truncates zero questions rather than one — the row a human
would have picked — and the rule chose cap30 because "smaller hard cap" was written before
the data arrived.

The final zero-yield audit over all 7,180 sessions: 1,253 empty, of which 267 are below
the minimum turn count and 986 are analysed in detail. **Zero content-policy refusals and
zero identical-input-different-outcome cases**, 13 annotated evidence misses, 5
source-format problems, 175 same-text-different-date outcome differences, 198 short
sessions that may hold no durable fact, and **862 left undetermined** rather than
relabelled as random failure.

## 2026-08-24 — train150 paused with 197 sessions remaining

| item | verified state |
|---|---:|
| terminal sessions | 6,983 / 7,180 |
| successful sessions (at least one stored memory) | 5,780 |
| explicit empty-result sessions | 1,203 |
| content-policy blocks | 0 |
| pending sessions | 197 |
| memories / unique vector ids | 18,017 / 18,017 |
| ingest API calls / tokens | 2,500 / 20,272,542 |
| failed API calls | 37 |

The resumed provider run advanced 967 sessions and stopped normally at exactly
500/500 daily calls. SQLite quick-check, foreign keys, extractor fingerprint,
checkpoint memory count, index ids and vector shape all agree. Unlike the two prior
quota boundaries, there is no raw-only pending batch: all 6,983 archived sessions
are terminal, and the remaining 197 have not been misclassified.

The current partial zero-yield partition contains 1,203 empty sessions: 13 annotated
evidence misses, 3 source-format problems, 159 date-variant outcome differences,
192 short no-durable-fact candidates, and 836 comparison-pending, consistent-zero,
or otherwise unresolved cases. Content-policy refusals remain zero. These counts
stay provisional until the final 197 sessions are terminal.

The fixed provisional context candidate was also rerun over 145/150 fully ingested
train questions. `mean` aggregation gives **95.2% Top-3** and **95.2% assembled
source-session recall**, with 140 median context tokens and one truncated question.
It remains well above the registered 80% gate, but the final five questions can
still change the reported result and no v2 config is published yet.

## 2026-08-23 — train150 resumed and paused safely at the next quota boundary

| item | verified state |
|---|---:|
| terminal sessions | 6,016 / 7,180 |
| successful sessions (at least one stored memory) | 5,007 |
| explicit empty-result sessions | 1,009 |
| content-policy blocks | 0 |
| pending sessions | 1,164 |
| memories / unique vector ids | 15,644 / 15,644 |
| ingest API calls / tokens | 2,000 / 17,412,776 |
| failed API calls | 37 |

The Pacific-date quota reset was confirmed by a real provider run. An initial
sandboxed attempt stopped on DNS resolution before writing a terminal batch; the
same checkpoint was verified and resumed with network access. The successful run
advanced 1,050 sessions and then stopped normally at 500/500. SQLite quick-check,
foreign keys, extractor fingerprint, checkpoint memory count, index ids and vector
shape all agree. One 15-session raw-only batch remains intentionally non-terminal
and safe to replay.

The partial zero-yield partition now covers 1,009 empty sessions: 11 annotated
evidence misses, 3 source-format problems, 116 date-variant outcome differences,
161 short no-durable-fact candidates, and 718 comparison-pending, consistent-zero,
or otherwise unresolved cases. Content-policy refusals remain zero. The final
causal report still waits for all 7,180 sessions; unresolved cases are not relabelled
as random failures merely because they are empty.

The fixed provisional context candidate was rerun offline over the now-complete
125/150 train questions. `mean` session aggregation still gives **96.0% Top-3** and
**96.0% assembled source-session recall**, with 146 median context tokens and one
truncated question. This is stable against the earlier 103-question diagnostic, but
remains explicitly partial and cannot publish `configs/v2.yaml`.
The production finalizer was exercised at 6,016/7,180 and returned exit 2 without
creating that file, so the partial diagnostic cannot accidentally cross the freeze
boundary.

## 2026-08-22 — train150 paused on daily quota

| item | verified state |
|---|---:|
| terminal sessions | 4,966 / 7,180 |
| successful sessions (at least one stored memory) | 4,129 |
| explicit empty-result sessions | 837 |
| memories | 12,924 |
| completed questions | 103 / 150 |
| content-policy blocks | 0 |
| namespace/provenance mismatches | 0 |
| tests | 529 passing |
| line coverage | 80% overall; critical experiment paths 88–100% |

The ingest is paused at the provider's 500-request daily limit. SQLite quick-check
and foreign keys pass; the 12,924 database memories exactly match 12,924 unique
vector ids. The database has one additional 15-session raw archive batch from the
deduplication call that hit quota. Those sessions have zero memories and no terminal
checkpoint state, so the next run replays them rather than counting them as empty.
Resuming does not reprocess a completed session. Immediately before the resume, the current extractor
fingerprint was re-derived and exactly matched the store: batch 15,
`two-stage-p10-v2`, prompt `017e6620b1b9`, schema `530989df0aec`, and dedup 0.9200.

### Correctness repairs made before resuming

- Source sessions now have user-scoped internal ids.  The legacy store was migrated
  without re-extracting its 9,588 memories, with a SQLite backup retained.
- A quota/network failure no longer marks or analyses its in-flight batch as a false
  zero-yield result. Content-policy refusals still retain their raw text and receive
  an explicit terminal state. A real dedup-stage quota stop left one raw-only batch;
  an automated replay test proves it resumes without duplicate sessions or memories.
- `ingest run` now reports completed, blocked and total-terminal sessions separately,
  and returns exit code 2 while resumable work remains.
- Session-recall evaluation decodes internal ids and refuses to publish a final gate
  from an incomplete store.

### Provisional diagnostics — not frozen results

The zero-yield audit over completed data finds 837 empty sessions in total: 185 have
fewer than six turns and 652 are substantive. Every empty session now has a detailed
row, including the short ones rather than silently dropping them. Across all 837,
there are 8 annotated evidence misses, 80 identical-text sessions whose synthetic
dates differ and whose outcomes differ, 40 date variants awaiting their other
occurrence, 17 consistently empty date variants, 139 short single chats that may
contain no durable fact, 3 short malformed sources, and 550 substantive single
occurrences without enough evidence for a causal label. No content-policy refusal
has occurred so far.

An earlier draft called the 28 outcome differences “identical repeated-source”
evidence.  A new hash over date, role and text showed that LongMemEval reuses the
same conversation text under different synthetic dates; the extractor sees that
date, so these are near-repeats, not controlled repeats, and they no longer count as
proof of randomness. The causal evidence remains the separate randomized
batch-position pilot. In this partial observational store, positions 0–3 are empty
in 106/1,302 attempts (8.1%), versus 546/3,055 (17.9%) at positions 4–14, a 2.20× risk.
At the front-position rate, about 249 later-position zeros would be expected; the
observed excess is about 297. That is a population estimate, not an individual
session label.

On the 103 fully ingested questions, the correct source session appears in the
production path's top-20 memory input for 98.1% of questions. Session aggregation is the
bottleneck:

| aggregation | recall@1 | recall@2 | recall@3 | recall@5 |
|---|---:|---:|---:|---:|
| max | 86.4% | 93.2% | **95.1%** | 98.1% |
| mean | 83.5% | 94.2% | **96.1%** | 96.1% |
| sum_top3 | 49.5% | 67.0% | 75.7% | 90.3% |
| sum | 44.7% | 66.0% | 75.7% | 90.3% |

These figures use the production path's top 20 retrieved memories.  An earlier
partial diagnostic accidentally aggregated 50 returned memories by confusing the
retriever's internal `candidate_limit` with the answer path's `top_k`; the script
now reads `retrieval.top_k` from the frozen config and records both values.

Ranking recall is not enough: the hard memory cap can discard an oversized session
after it ranked in the top three.  The current full-session/20-memory budget and the
best provisional train-only candidate compare as follows:

| context candidate | rank @3 | assembled gold-session recall | median context tokens | truncated questions |
|---|---:|---:|---:|---:|
| `max`, whole sessions, cap 20 | 95.1% | 90.3% | 178 | 18 |
| `mean`, radius 1, cap 30 | **96.1%** | **96.1%** | **146** | **1** |
| flat top 20 | — | memory-level 98.1% | 357 | — |

The candidate is not yet written to `fallback.yaml`: the complete 150-question
train sweep must reproduce it first.  Its separate provisional artifact is
`results/analysis/train150-context-candidate.partial.json`.

It misses 4/103 questions: two have no gold-session memory anywhere in the 20-memory
input, and two rank the gold session below third. The context builder therefore reaches
the available Top-3 ceiling (99/103) on this partial cohort; further window tuning
cannot repair the remaining four and is stopped here rather than fitted to their
individual answers.

The old approximately-64% figure is not retained: its script mixed incomplete
questions into the denominator and compared scoped database ids directly with
public dataset ids.  The full train150 sweep, not this partial table, decides the
frozen aggregate.

### Next checkpoint

Resume the same ingest after Pacific midnight.  Once all 7,180 sessions are
terminal, regenerate the zero-yield audit with `--require-complete`, rerun the full
aggregation table, select and record one configuration, then freeze it before
touching `dev100`.

## 2026-08-22 — offline validation and final-test safeguards

While the extractor quota remains at 500/500, work that needs no provider calls was
completed.  This does not advance the train150 denominator and does not inspect
`dev100` or `test100`.

| safeguard | verified result |
|---|---|
| critical CLI/quota/raw-recall tests | quota stop is resumable and non-zero exit; atomic single-writer lock prevents duplicate ingest; raw source ids remain scoped correctly |
| aggregate-only validation report | accuracy, three-run spread, majority vote, agreement, staged recall, context, usage and aggregate failure stages |
| usage recovery | result/usage artifacts must resume as a pair; prior calls load once, every scored question checkpoints cost first, and corrupt usage is never overwritten silently |
| `dev100` runner | checks the candidate freeze, seals individual rows, prints only counts until every arm completes |
| dev decision | exact three-arm/three-repeat protocol is enforced; the product choice is recomputed from the aggregate arm statistics instead of trusting an editable saved label |
| sealed output lineage | every dev/test JSONL and usage file is hashed into its aggregate; decisions and completed ledgers revalidate the raw files |
| oracle ceiling arm | gold session selection is explicitly labelled and otherwise uses the same session/window/context budget |
| v2 freeze | rejects store drift; pre/post freezes share one code/config/data/protocol lineage, including all train150 selection evidence, and test100 also binds the dev decision |
| `test100` runner | atomic protocol lock rechecks the durable ledger after acquisition; orphan rows/usage/reports and every second completed run are refused |
| formal protocol lock | canonical 100-question manifests, freeze/store names and exact baseline order are mandatory; test v2 must equal the dev decision |
| final report semantics | one-shot test output omits fake majority/agreement/variance fields and reports single-run accuracy, recall, usage and failures |
| frozen ingest runner | dev100/test100 stores must be generated under a pre-ingest source/config/data hash, not merely frozen after the fact |
| train finalizer | refuses partial/corrupt train150; verifies a temporary candidate first and only then atomically publishes `v2.yaml` |
| Top-3 denominator | a final gate now requires all 150 manifest questions to have gold sessions and be scored; skipped rows cannot shrink the denominator |
| ingest state checker | one read-only command jointly verifies checkpoint, raw archives, SQLite, memory/index/vector counts and extractor fingerprint before every resume |
| final baselines/runbook | test100 baselines are registered before dev/test access; exact resumable commands are in `v2-runbook.md` |
| productization gate | account identity, deletion/export, restore, cost, concurrency, privacy and operations have explicit post-test acceptance criteria |
| regression suite | **529 passed**, Ruff and formatting clean, `git diff --check` clean |

The aggregate failure labels are deliberately modest: they say where the correct
**source session** disappeared (`candidates`, `top_k`, or context assembly).  If the
right session reached the model and the answer is wrong, extraction loss, answer
reasoning and judge error remain combined; the report does not pretend source-session
recall proves the exact fact survived.

No candidate config has been written and no freeze has been captured yet.  Both wait
for all 7,180 train150 sessions and the full, final train-only ranking table.

### Fixed context grid and selection rule

The finalizer's search space is now fixed before the remaining 74 train questions
arrive.  It evaluates seven nearby controls only: `max/whole/cap20`,
`mean/whole/cap20`, `mean/radius1/cap20|30|40`, `mean/radius2/cap30`, and
`max/radius1/cap30`.

Candidates must have Top-3 source-session recall ≥80%, assembled source-session
recall ≥80%, and median context no more than 1.5× flat top-20.  Selection then uses,
in order: higher assembled recall, higher Top-3 recall, smaller hard memory cap,
smaller window, and fewer median tokens.  If none passes, no config is written.
`scripts/finalize_train150.py` tested the guard against the current partial store and
correctly stopped at 3,648/7,180 without creating a final artifact.
