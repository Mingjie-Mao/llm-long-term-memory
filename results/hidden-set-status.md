# The final test set — what exists, what does not, and what a candidate must prove

> **Update 2026-09-14: BEAM is adopted, and its final half is registered.** Its dataset turned
> out to be public (revision `3205395e` of the 100K and 500K files, CC BY-SA 4.0). All 1,100
> questions passed `register_hidden_set.py`'s id, verbatim and near-duplicate checks against
> the 500 LongMemEval questions, and the set was split 40/60 by conversation
> ([`beam-split.json`](manifests/beam-split.json)). The 660-question final half is registered
> as [`beam-test`](manifests/beam-test.json), to be answered once, after every choice is
> frozen. Step 2 below concerned LoCoMo's audit and does not apply; step 3's resolution was
> computed per stratum by `tools/beam_power.py`, as scenarios until BEAM's own noise floor is
> measured. The plan is in the [roadmap](../docs/PROJECT_REPORT.zh-CN.md#后续完整计划). The text below is the record of
> how the choice was made.

**v4 still has no final test.** Every v4 number on record is a development number. This
page says exactly what is blocking, so that the gap is a stated condition rather than
something a reader has to reconstruct from a footnote.

## Why nothing on disk can serve

LongMemEval-S is exhausted. Its 500 questions are partitioned across five disjoint sets
with **zero remaining**, and `longmemeval_oracle.json` carries the **same 500 question
ids** over a different haystack — verified, not assumed. A different haystack over
identical questions is not new data.

| set | n | exposure |
|---|---:|---|
| `dev50` | 50 | burned — read in full, prompts fitted on it |
| `heldout100` | 100 | spent as the v1 final |
| `train150` | 150 | v2/v3 development, read question-by-question |
| `dev100` | 100 | v2 validation; aggregate reads only, at most three v4 decisions |
| `test100` | 100 | v2 final; one shot, spent |

## The 87 held-out probes are not a substitute, and are reported separately

The v4 probe split holds back 87 of 229 synthesis probes (`count` 30, `duration` 30,
`comparison` 27), drawn before any v4 measurement, and the tool refuses to re-draw them.
They remain **unanswered**, and they are worth what they are worth:

- **What they control for:** fitting the probe set. A candidate tuned until the
  development probes improve has not been shown to generalise even to sibling probes
  from the same generator, and these detect that.
- **What they cannot control for:** anything else. Both halves come from the same store,
  the same extraction generation and the same generator, and their ground truth is SQL
  over stored facts, so it inherits every extraction error. Answering them measures
  synthesis *conditional on the store being right*.
- **Therefore:** a held-out probe result is an instrument reading. It may never be
  reported beside a LongMemEval accuracy figure, and it is not the final test this
  project owes. It is reported on its own page, under its own heading, with this
  paragraph attached.

## What a real candidate must prove

`tools/register_hidden_set.py` enforces this, and refuses on any of it. The check is run
**before** anything is frozen, because afterwards it is worthless.

1. **No question-id overlap** with `dev50`, `heldout100`, `train150`, `dev100`, `test100`.
2. **No verbatim question overlap** after normalisation — new ids over old text is the
   failure an id check alone cannot see.
3. **No near-duplicates**, at ≥ 80% content-word overlap against every question already
   read. A paraphrase is contamination for a memory benchmark in the same way a copy is.
4. **No repeated ids inside the candidate**, because `n` is the denominator of the
   reported result.
5. **Not already registered.** A set registered twice is a set someone reconsidered after
   seeing a result.

Verified working: pointed at LongMemEval-S itself, all three contamination checks fire.

**What it cannot check.** Whether the source corpus overlaps the *evaluated model's*
training data. Nothing in this repository can establish that, so a pass means "unseen by
this project", which is a weaker claim, and the manifest records it in those words.

Beyond the automated checks, a candidate also needs a licence that permits this use, a
conversational multi-session format the ingestion pipeline can read, and an ingestion cost
that has been estimated before it is started rather than after.

## Status: source direction chosen, specific corpus not yet adopted

The machinery is built and tested. The direction is set — **an external conversational
memory benchmark**, rather than a generated corpus or real user data. What remains is
choosing which one, and nothing below has been downloaded, ingested or registered.

### Shortlist, from a survey on 2026-09-12

| candidate | scale | licence | fit |
|---|---|---|---|
| **LoCoMo** (snap-research) | 10 conversations, ~300 turns each; QA pairs annotated by category | **CC BY-NC 4.0** — non-commercial | closest format to LongMemEval-S; sessions with timestamped turns map onto the existing ingest with an adapter |
| **BEAM** (ICLR 2026) | 100 conversations, 2,000 validated questions, up to 10M tokens | paper is CC BY 4.0; **dataset release URL not confirmed** | best statistical resolution of the three; ten memory abilities including contradiction resolution and temporal ordering |
| **PerLTQA / DialSim** | — | not checked | narrower: PerLTQA targets memory classification and retrieval; DialSim derives from television scripts, so topical diversity is limited |

### Two findings that should decide this

**LoCoMo carries a published third-party audit alleging substantial defects.** Recorded as
a claim, not as established fact — it is a third-party repository, not the authors — and it
must be verified against the data before adoption, not after. It reports 99 wrong answers
in 1,540 questions (6.4%, capping any score at 93.6%), an LLM judge that accepts about 63%
of deliberately wrong answers, category sizes from 96 to 841, and a 446-question category
left unevaluated by broken code.

**The category-size claim is this project's own problem restated.** An audit saying the
smallest LoCoMo category needs a 15-point gap to distinguish two systems is the same
finding as [`gate-protocol.md`](gate-protocol.md): a stratum too small to resolve the
effect being claimed. Adopting a benchmark without checking its per-category resolution
would repeat, on purchased data, the mistake that invalidated the v4 gates.

**Therefore BEAM is the better fit if its data is actually obtainable** — 2,000 questions
across 100 conversations is roughly an order of magnitude more resolution than LoCoMo, and
resolution is the binding constraint on every v4 decision. The open question is availability,
not suitability.

### What has to happen before anything is registered

1. Confirm BEAM's dataset release and licence. If unobtainable, fall back to LoCoMo.
2. Verify the audit's claims against the actual file, since a 6.4% ground-truth error rate
   changes what any score means and is measurable directly.
3. Compute the per-category minimum detectable effect with
   `tools/check_gate_is_resolvable.py` **before** adopting, and declare in advance which
   categories are large enough to decide anything.
4. Write the ingest adapter and run `tools/register_hidden_set.py`, which refuses the set
   if it overlaps anything already read.
5. Only then freeze, and answer it once.

Rejected for now, with reasons: a **generated corpus** tests the generator as much as the
system, and the generator is a model; **real user data** is the only option that measures
the actual product, but the product has no users, and it would need consent and a privacy
review first.

Until one is chosen and registered, the rule from the data protocol stands unchanged:
**every v4 number is a development number, and reporting one as a final result is the
single worst thing this protocol exists to prevent.**
