# v3.3 on the synthesis probes — an instrument reading, and four defects in the instrument

**Run 2026-09-05 (UTC day), `two_stage_reasoned_evidence` + `configs/v3-phase5-compact.yaml`
against `train150`.** 216 of 229 probes; the daily answerer quota ran out with 13
`current_state` probes left. One answerer call per probe plus a second on fallback, no
judge calls — grading is derived, so no `gemma` quota was spent. Median context 1,120
tokens, consistent with the 1,115 measured on `dev60`.

Rows: [`results/raw/probes.v3.3.jsonl`](../raw/probes.v3.3.jsonl).
Aggregate: [`synthesis-probes.v3.3.json`](synthesis-probes.v3.3.json).

## Read this table only as four separate measurements

Pooling them would average a synthesis result with a retrieval ceiling. `interpretable`
means the context held every fact the probe needs *and* no evidence anchor's stored
`event_time` contradicts a date written in its own text.

| operation | n | correct | abstained | wrong | no number given |
|---|---:|---:|---:|---:|---:|
| `current_state` | 30 | **86.7%** | 3.3% | 3 | 0 |
| `comparison` | 46 | **58.7%** | 0.0% | 17 | 2 |
| `duration` | 47 | **42.5%** | 4.3% | 25 | 0 |
| `count` | 25 | **4.0%** | 8.0% | 18 | 4 |

`current_state` is 30 rather than 49: 13 probes were not reached, and 6 more carry an
anchor conflict. It is the least complete row here and the strongest one, which is the
wrong way round for confidence.

## The abstention bucket did not appear

The failure taxonomy put abstention at 44–50% of LongMemEval failures, and that is the
bucket v4 was scoped to attack. **On these probes it is 0–8%.** The model answers, with
confidence, and is wrong.

That is not a refutation of the taxonomy — it is a difference between the two
instruments, and the most likely cause is the probes' own phrasing. Every probe states
its operation in the question ("How many distinct...", "Which of these happened
first..."), which is close to what the proposed `operation-first` fix would supply
explicitly. If that reading is right, the fix's value is already visible here; but the
probe templates are a confound, not a controlled comparison, so this is a hypothesis to
test, not a result.

## Four defects in the instrument, all found without a provider call

Three were found before the run and two of those changed how it must be read.

**1. 43% of `count` probes are unanswerable from the context they get.** Replaying v3.3's
retrieval offline: on 26 of 60 count probes the top-20 context does not contain every
item the ground truth counts. A perfect counter still loses.

    truth 10   most the context allows   8
    truth  8   most the context allows   5
    truth  4   most the context allows   2

`duration` and `comparison` are clean here — 100% of their evidence is retrieved — and
`current_state`'s active member was retrieved on 49 of 49.

**2. The `count` question does not say which predicates are in scope.** Ground truth is
`COUNT` over one mapped `relation_type`; the question is natural language and the store
splits one concept across several predicates. So the model counts things the truth
excludes and is scored wrong for it:

    "How many distinct possessions does the user have on record?"
    truth = 8   counts clothing_brands + collectibles + luxury_goods
                excludes furniture_purchases, which maps to `acquired`
    model = 9   counted the furniture too

On another, the truth counts a `movie_preference` as a watched film. **The 4.0% figure
is not a measurement of counting ability.** These probes need re-specifying before they
mean anything.

**3. 45% of `duration` probes have a gold one day below what is computable.** Ground
truth subtracts two full timestamps and floors; a time of day is never rendered into the
context, so the answerer only ever sees two dates. On 27 of 60 probes the two readings
differ by exactly one day. The grader now accepts either and records which — 11 rows are
correct only on the visible-date reading, which is the difference between 19.1% and
42.5%.

**4. 22% of `duration`/`comparison` probes are anchored on an `event_time` that
contradicts their own text.**

    event_time 2021-12-10, text says "March 17th, 2021"
    event_time 2023-05-24, text says "July 2, 2023"

Same root cause as the `assistant_recommendation` filter from the first round:
`event_time` holds the session date, not the event date. The `source_role='user'` and
set/event arity filters did not remove it. These rows are reported in their own stratum.

## Two grading defects, fixed by re-scoring the saved replies

Model output is the expensive artifact and verdicts are derived from it, so both fixes
cost nothing.

- The first grader looked for a bare `A`/`B` and scored **36 of 60** `comparison` replies
  unparseable. The answerer replies in prose — *"X ... happened before Y"* — so the
  grader now matches the reply against the two described events. 58.7%, not 39%.
- It read a count reply's *first* integer, which was `$200` or a day-of-month. A stated
  total now wins, then a count-noun phrase, then the last number that is not a bare year.

The remaining `no number given` rows are real: asked "how many", the model listed the
items and never stated a total.

## What this does and does not license

It answers *"which operation is weakest"* for `comparison`, `duration` and
`current_state`, on the clean strata only. It does **not** answer it for `count`, which
needs defects 1 and 2 fixed first. It is not accuracy, and must not be quoted beside a
LongMemEval figure.

## Next, in order

1. Re-specify the `count` probes so the question names its scope and the answer is
   reachable from the context. Free.
2. Regenerate `duration`/`comparison` with an anchor rule that rejects an `event_time`
   contradicting its own text. Free, and it fixes a store defect worth knowing about
   independently.
3. Finish the 13 unreached `current_state` probes.
4. Only then treat any of these as a v4.0 baseline.

---

# The probe set was regenerated after this reading

The set this reading was taken on is archived at
[`../archive/synthesis-probes-v1/`](../archive/synthesis-probes-v1/) with its rows. The
numbers above belong to that set and **must not be compared with any reading on the new
one**: probe ids, questions and ground truth all changed.

## Fixed in the generator

| defect | before | after |
|---|---:|---:|
| `duration`/`comparison` anchored on an `event_time` its own text contradicts | 27 / 120 | **0 / 120** |
| `count` question does not name the predicates it counts | 60 / 60 | **0 / 60** |
| `count` namespace also holds a confusable relation (`owns` vs `acquired`) | 8 / 60 | **0 / 60** |
| `count` asks about *the user* but counts another subject's facts | 6 / 60 | **0 / 60** |

The last row was found while verifying the other two. It is the same mismatch the
`_dated` filter was written for in the first round — a question about the user scored
against the assistant's facts — arriving through a door that filter does not cover.

Each rule is a generator rule, not a hand-removal: the set still reproduces byte for
byte from `(seed, store fingerprint, per_kind)`, and four tests pin the rules.

## Not fixed, and not fixable here

**`count` evidence still does not fit in the retrieved context.** Naming the scope in
the question was expected to help — the scope words are part of the retrieval query —
and it did not: full-evidence retrieval went from 56.7% to **58.3%**, inside noise.

Set size is not the lever either. Even the smallest sets miss:

    members  3   21/30 fully retrieved
    members  4    4/8
    members  5    2/7
    members  9    1/4

So this is not a defect in the probe set. Top-20 semantic retrieval over a namespace of
100–180 memories does not reliably surface every member of one relation, which is the
same ceiling [`predicate-vocabulary.md`](predicate-vocabulary.md) quantified from the
other direction. A generator rule that dropped the probes the retriever happens to miss
would make the probe set a function of the system under test, which is the one thing it
must not be.

**It therefore stays in the runner's `context_incomplete` stratum**, where a count
result is reported separately from an answerable one. Closing it is data-layer work — a
controlled vocabulary and an exhaustive scan instead of a similarity ranking — not
probe-set work.

## Also fixed: the `duration` gold is now what the answerer can compute

`render_memory` formats every anchor as `%Y-%m-%d` and never shows a time of day.
Ground truth used to subtract two full timestamps and floor, which put the gold one day
below the computable answer whenever the later event's clock time was the earlier of the
two — **27 of 60 probes**. The generator now drops the times before subtracting:

    duration golds unreachable from the rendered dates:  0 / 60   (was 27)

The grader's second-reading fallback was removed with it. Accepting both answers would
now hide a generator bug rather than correct one, and the generator is the right place
for this to be true.

Checked at the same time: `render_memory` shows `valid_from or event_time`, so a
`valid_from` that disagreed would be a third silent mismatch. On all 240 anchors
`valid_from` is set and agrees with `event_time`.

## Still open

Nothing on `duration` or `comparison`. On `count`, the retrieval ceiling above.
