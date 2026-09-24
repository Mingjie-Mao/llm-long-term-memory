# v2e: saying which dates the facts state — pre-registration

Date: 2026-09-22 (Australia/Sydney)

## Classification

**DEVELOPMENT mechanism experiment.** Not an unseen final evaluation.

The 48 questions are the v2c reasoning-48 set. Their per-question rows have been read
in full, twice: once attributing v2c's eleven failures, once probing this mechanism.
They are development evidence. Nothing here may be reported as a benchmark estimate or
placed on a version curve with the frozen 72% on `test100`.

## What changed underneath, and why this can be measured at all

Splitting `event_time` (what the fact says) from `observed_at` (when it was said) gave
new ingests a signal. It gave existing stores nothing: they were written when
`event_time` was the conversation's date for every row, so `event_time_is_stated` reads
True on **8,968 of 8,968** memories in `stores/v2c-reasoning48.db` and discriminates
nothing.

`tools/backfill_event_time.py` recomputes it with no model calls, because both inputs —
the fact's words and the date it was said — are already columns. On that store:

| | rows |
|---|---:|
| carried a time before | 8,968 (every row) |
| carry a **stated** time after | 463 |
| assumed date withdrawn | 8,505 |

The gate below runs on a backfilled **copy**. The committed store is not modified.

## The mechanism that is NOT registered, and the measurement that stopped it

The motivating failure is `gpt4_74aed68e`: "how many days between replacing the spark
plugs and the racing event", gold 29, answered "zero". After the backfill the two
memories are finally distinguishable —

- `The user replaced their car's spark plugs with NGK brand plugs on February 14th` —
  stated, `2023-02-14`, **not retrieved**
- `The user replaced their car's spark plugs with NGK brand plugs.` — unstated,
  observed `2023-03-15`, **retrieved**

— so the obvious mechanism is to fetch the dated statement of the same fact when a
temporal question's evidence is undated. Two things stop it:

1. The pair does not share a key. Stage B gave them different predicates
   (`car_maintenance` and `spark_plugs`), so a key lookup returns nothing and the
   mechanism would have to match on content, which is a retrieval change.
2. Across all 48 questions, **30 are temporal or aggregation, 30 of those have undated
   evidence selected, and the "a dated statement of the same fact exists but was not
   retrieved" pattern occurs exactly once** — on the question it was designed from.

Building a retrieval mechanism from one post-selected case on a set whose rows have
been read is the failure the v2b decision already recorded: its post-selected repeat
had no valid p-value. So it is not built.

## The mechanism that is registered

`valid_from` is the date a fact was *stated*, so an undated fact has always rendered as
`(since 2023-03-15)` — a mention date wearing an event date's clothes. Under v2e it
renders as `(mentioned 2023-03-15; the fact states no date)`.

That is the entire change. This is registered rather than the retrieval variant because
its scope is the population, not one row: **30 of 30** temporal questions carry undated
selected evidence.

## Arms

| arm | variant | label |
|---|---|---|
| A — baseline | `two_stage_v2c` | the committed `two_stage_v2c.reasoning48-v2c8.jsonl`; **not re-run** |
| B — candidate | `two_stage_v2e` | `v2e-reasoning48` |

Identical store (backfilled copy), retrieval, `top_k`, answer policy (`v2c`), prompts,
models and judge. `render_memory`'s `mark_unstated` is the only difference, and it is
off by default so every recorded arm is untouched.

## Primary endpoint and gates

Primary: **paired win/loss against arm A over all 48 questions.**

Promote only if all hold:

1. **net positive** — wins exceed losses;
2. **no temporal regression** — no question of type `temporal-reasoning` correct in A
   and wrong in B;
3. **the mechanism is visible** — every temporal question's context contains at least
   one `states no date` marker, which the offline gate predicts for 30 of 30;
4. **context cost unchanged** — median context tokens within 10% of v2c's 636; the
   marker is a few words and must not be paid for as a rewrite;
5. **frozen and sealed artifacts unmodified.**

## The risk this mechanism carries, stated before the numbers

**It can lose questions, and that would not be a bug.** Telling a reader that a date is
only a mention date removes a cue it was previously guessing from. Where the guess
happened to be right, the answer may become an abstention. A net-negative result here
means the reader was extracting value from a date it should not have trusted, which is
worth knowing and is a legitimate outcome, not a failed implementation.

For the same reason gate 2 restricts the no-regression check to `temporal-reasoning`
rather than all types: a change in how dates are labelled should not move a preference
question, and if it does, that is answerer variance rather than mechanism.

## Limitations

- Development set, read in full, twice.
- One run per arm, reusing a baseline measured on a different day. No significance is
  claimed and none will be reported.
- The backfill is a deterministic re-read of committed text, not a re-ingest. It
  recovers a stated date for 463 of 8,968 memories; a re-ingest with the current
  extractor might differ, and this does not estimate that.
- `stated_event_time` is deliberately narrow: relative expressions and dates that an
  offset is measured from are left undated. The anchor guard was added after four of
  the first twenty recovered dates turned out to be anchors — "about a month before
  May 21, 2023" is not an event on 21 May. Others may remain.
