# Raw-primary t1 (dates resolved before the reader) against v1 — heldout100, three runs per arm — pre-registration

Date: 2026-10-07 (Australia/Sydney). Written before any answer or judge call of this
experiment. Git HEAD `4a4cc64` plus the uncommitted working tree; sha256 prefixes at
registration below.

## Classification

**A development comparison, recorded as regression evidence.** `heldout100` is exposed:
v1 was decided on it in aggregate (`results/raw-primary-heldout100-decision.md`), and
older-generation failures were read by `tools/failure_taxonomy.py`. It was not used to
design t1, except for one aggregate count before registration (how many of its temporal
questions' gold turns contain a relative-date phrase: 20 of 27). No answer of it was read.
Nothing here is an unseen result or an accuracy figure. LongMemEval-S has no unseen
questions left; an unseen check needs another dataset.

## Why

`results/analysis/error-taxonomy-dev100-v18-v1.md`: the v1 arm lost 26 dev100 questions.
None was a retrieval miss. Nine were temporal-reasoning with every gold turn in context:
two events said the same day ("a month ago", "3 weeks ago") ordered the wrong way round,
"today" three weeks before the question answered as six weeks, "this month" in January
not matched to "two months ago" from March. Seven of the nine have their key date
computable from a phrase in a gold turn and its conversation date.

## What t1 is

`two_stage_raw_primary_t1` = v1 (`two_stage_raw_primary`) with exactly two changes, both
in how the 4,000-token question-found excerpts are rendered:

1. each conversation header gives its weekday and how long before the question it was
   ("Conversation on 2023-03-20 (Monday; 12 days, about 1.7 weeks, before the
   question)");
2. in the user's turns only, each relative-date phrase keeps its words and gains the date
   it resolves to from its own conversation's date, `≈` marking approximations: "a month
   ago [≈ 2023-04-24]", "for about 2 months now [since ≈ 2023-03-24]", "last weekend
   [= 2023-05-20 to 2023-05-21]" (`src/llm_long_term_memory/retrieve/time_notes.py`).
   Ambiguous phrases ("this Friday", "this May", "this past month", durations such as "in
   14 days") and titles ("Yesterday" the song) are left alone. Added before any call, on
   2026-10-07: a duration that runs up to the conversation ("for about 4 years and 3
   months now", "I've been working professionally for 9 years") gains its start date
   ("[since ≈ 2019-02-24]"); a past duration ("went to Japan for two weeks") does not.
   From the one dev100 temporal failure t1 did not reach (years-and-months subtraction);
   a hand read of 15 random train150 cases found none wrong and one missed (curly
   apostrophe), fixed with a test.

Retrieval, turn selection, memory context, answer prompt (`memory-aware-v2`), output
limit (512), fallback, models: unchanged. Same store, same turns, same order.

## Evidence before this run (zero calls)

| check | train150 | heldout100 |
|---|---:|---:|
| questions with ≥ 1 note | 115 / 150 | 68 / 100 |
| temporal questions with a note on a gold turn | 30 / 40 | 16 / 27 |
| median added context (estimate) | 142 tokens | 138 tokens |

Stub-client context precheck (`tools/context_precheck.py`, heldout100): v1 median 5,587,
p90 5,685, max 5,881; t1 median 5,735, p90 5,898, max 6,143. A hand read of 40 random
train150 annotations found one wrong ("over the weekend" said on a Sunday, dated to the
weekend before); fixed, with a test, before this registration. Unit tests:
`tests/test_time_notes.py` (including tenant isolation of the rendered excerpts).

## Data and system

Store `stores/heldout100.db`, config `configs/fallback.yaml`, manifest
`results/manifests/heldout100.json`, answerer `gemini-3.5-flash-lite`, judge
`gemma-4-31b-it`, temperature 0. Both arms run fresh; no earlier heldout100 row is reused.

## Resolution

On dev100, v1's three runs had 5 of 26 temporal questions unstable. With three runs per
arm and arms compared on each question's mean correctness, the type-level mean net has a
standard deviation near 0.9 when nothing changes; +2.0 is about two of them. The overall
floor for one run per arm on dev100 is 7, about 4 for a three-run mean
(`results/analysis/resolution-dev100.md`). No significance is claimed from any of it.

## Decision rules

t1 passes if **all** hold (`tools/raw_primary_gate_t1.py`):

1. temporal-reasoning mean net (Σ over its questions of t1's mean correctness − v1's)
   **≥ +2.0**;
2. overall mean net **≥ 0**;
3. no question type's mean net below **−2.0**;
4. median context of t1 over its three runs **≤ 6,000** tokens.

Reported, not deciding: per-run scores, per-type mean nets, median output tokens,
fallback second calls, an exact sign test over questions whose mean moved.

**A pass** makes date notes part of the next candidate, to be confirmed with the other
changes on a set that has not been used. **A failure** is recorded and t1 is not tuned
on heldout100.

## Cost

About 1.1 answer calls per question: ~340 per arm over three runs, ~680 in all, and 600
judge calls — two answerer quota days (500/day) before provider failures. Runs resume
from their result files. If all six are not complete by **2026-10-12**, the experiment
is recorded as incomplete.

## Order

Run together with `results/prereg-raw-primary-count-latest-v1.md`, which adds a third
arm (t2) on the same set: v1 rep1, t1 rep1, t2 rep1, then rep2, then rep3, so a quota
stop never leaves one arm more than one repeat ahead. Neither decision reads the other's
gate; t1's rules here are unchanged by the third arm.

## Reproduction

```bash
for rep in 1 2 3; do
  lltm eval run two_stage_raw_primary    --config configs/fallback.yaml --store-name heldout100 \
    --questions results/manifests/heldout100.json --label heldout100-t1-rep$rep --experiment-class regression
  lltm eval run two_stage_raw_primary_t1 --config configs/fallback.yaml --store-name heldout100 \
    --questions results/manifests/heldout100.json --label heldout100-t1-rep$rep --experiment-class regression
  lltm eval run two_stage_raw_primary_t2 --config configs/fallback.yaml --store-name heldout100 \
    --questions results/manifests/heldout100.json --label heldout100-t1-rep$rep --experiment-class regression
done
python tools/raw_primary_gate_t1.py
```

## Registration hashes (sha256, first 16)

Taken after t2 was added to the same files (its policy is off for t1), before any call.

| file | sha256 |
|---|---|
| `src/llm_long_term_memory/retrieve/time_notes.py` | `55d13ca8a02d486f` |
| `src/llm_long_term_memory/retrieve/excerpts.py` | `95c6cf71dbd004a3` |
| `src/llm_long_term_memory/evaluation/runners/memory.py` | `2b66de1491dd5dd1` |
| `src/llm_long_term_memory/cli.py` | `1e6da645d864d27a` |
| `src/llm_long_term_memory/answering.py` | `fda1b52d78d41d0d` |
| `configs/fallback.yaml` | `397c0c59908079c9` |
| `tools/raw_primary_gate_t1.py` | `b1d151f849364943` |
| `tools/raw_primary_gate_t2.py` | `83cc20ce3a4f9d99` |
| `stores/heldout100.db` | `c8db4c2c0cec5f0a` |
| `results/manifests/heldout100.json` | `19f833d8d9457706` |

## Withdrawn before any call — 2026-10-07

No run of this registration was started. Two small probes on train150
(`results/t-probe-v1-decision.md`, `results/prereg-t-probe-v2.md`) showed that the date
notes alone did not move temporal failures and that the notes path did; the plan is
replaced by `results/prereg-raw-primary-t3-t4-heldout100-v1.md` (v1, t3, t4 on heldout100).
