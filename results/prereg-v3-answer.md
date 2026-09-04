# Pre-registration — v3 answer reasoning

**Written after the v2 one-shot conclusion was sealed and before any v3 provider run.**

## Boundary

- v2 remains unchanged and is the control.
- v3 may use `train150`, synthetic mechanism fixtures and a newly created development set.
- v3 must not read test100 questions, answers, individual rows or judge reasons. The published
  aggregate failure counts (3 retrieval misses, 14 wrong after raw fallback and 11 wrong despite
  source context) are the only test100 diagnosis used to choose this work.
- The query router may use the user's question text and current date. It may not use benchmark
  question type, gold answer or answer-session ids.

## Hypothesis

When relevant evidence is already present, an explicit evidence → operation → checked answer
contract will reduce date arithmetic, cross-session counting and update-selection errors. It
should not change retrieval, stored memories or context size.

## Arms

| arm | retrieval/store | answer policy | purpose |
|---|---|---|---|
| `v2-control` | frozen v2 | `memory-aware-v2` | unchanged baseline |
| `v3-reasoned` | same as v2 | `memory-reasoned-v3` | isolated answer-policy change |

The later smart-context arm is not part of this comparison. It begins only after the answer-policy
gate, so a gain or loss can be attributed to one change.

## Required mechanism checks before API use

1. Existing temporal, aggregation and preference regression questions route from question text
   alone to the expected operation.
2. The structured answer contains a short evidence summary, optional calculation and confidence,
   while returning only the concise answer to the user.
3. v2 continues to use its original prompt and schema byte-for-byte at runtime.
4. Every result row records the actual v2 or v3 answer-prompt version.
5. Missing evidence still requests raw source or abstains; reasoning is not permission to guess.

## Development gate

Run paired arms on a train-only, stratified temporal/multi-session mechanism set, then on train150.
Use three repeats for comparisons that include temporal or multi-session questions.

| gate | requirement |
|---|---|
| target failures | at least +3 percentage points majority accuracy **or** 30% fewer answer-stage failures |
| ordinary questions | no more than 2 points majority-accuracy regression |
| abstention | no increase in unsupported confident answers |
| retrieval/context | identical selected context and no context-token increase |
| generation cost | answer output tokens no more than 1.5x v2 |

Failing a gate records a negative result; it does not trigger inspection of test100. Passing the
gate permits Step 3's new hidden reasoning set, not deployment.
