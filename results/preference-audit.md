# Hand audit: `single-session-preference` scores 0/3 for everything

**Audited 2026-08-14 from committed JSONL. No API quota spent.**

Every system scores 0/3, including `full_context`, which sees the entire history.
That rules out retrieval and memory as the cause. The audit below identifies what it
actually is.

## The finding

**The reference answers for this category are not answers. They are grading
rubrics.**

| Question | Reference answer (abridged) |
|---|---|
| "My phone battery is poor lately. Any tips?" | "The user **would prefer responses that** build upon their previous mention of purchasing a portable power bank…" |
| "Can you suggest a hotel for my trip to Miami?" | "The user **would prefer suggestions of** hotels with great views… may not prefer basic or budget hotels" |
| "My chocolate chip cookies need something extra. Any advice?" | "The user **would prefer responses that** build upon their previous experimentation with turbinado sugar…" |

The question is an ordinary user request. The gold is a description of what a good
reply would contain. The intended task is therefore: *produce a reply that reflects
the remembered preference*, and grade it against the rubric. The current judge
instead compares the reply to the gold **as if the gold were the expected output**.

## Per-question verdict

| Question | What happened | Class |
|---|---|---|
| `38146c39` cookies | `full_context` **did the task correctly** — it suggested turbinado, muscovado and demerara sugar. The judge marked it wrong because "the reference answer describes the user's preferences, whereas the candidate answer provides actual cookie-making suggestions." | **J — judge error** |
| `09d032c9` battery | `full_context` gave generic battery tips without referencing the power bank. Correctly marked wrong. `two_stage_hydrated` *did* retrieve the power bank and named it, but framed the reply as "I do not know what phone you use", so it declined instead of advising. | **A — answerer failure** (both) |
| `0edc2aef` Miami hotel | `full_context` answered "there is no information about a trip to Miami" despite seeing the whole history. `two_stage_hydrated` answered "I do not know". | **A — answerer failure**, possibly compounded by a reference/retrieval mismatch |

So of three: **one is a judge error against an answer that was actually right**, and
two are answerer-prompt failures where the model declined instead of advising.

Notably, on `09d032c9` LLTM **retrieved the right memory** — the reply names
the portable power bank and the wireless charging pad. The memory layer did its job;
the answering step then refused to use it.

## Why this is not a LLTM defect

`full_context` has the entire conversation in its prompt and still scores 0/3. No
change to extraction, retrieval, hydration, or ranking can move this category. Fixes
belong in two other places:

1. **Answerer prompt.** It currently optimises for factual recall and abstention —
   which is why abstention scores 100% — and that same disposition makes it decline
   on advice-shaped questions where the correct behaviour is to advise *using* the
   remembered preference.
2. **Judge prompt.** It needs to know that a rubric-style reference is a rubric:
   grade whether the reply satisfies the described preference, not whether it
   restates it.

## Why it still matters for the product

This is the one category that tests the thing the product is *for*: adjusting a
reply to remembered preferences ("given that you dislike raw fish…"). The capability
gap is real even though the measurement of it is broken. The demo walkthrough should
exercise exactly this, and the answerer prompt fix is on the path to it.

## Recommendation

Report this category with a footnote rather than folding it into the headline
average, and do not attempt to fix it by changing the memory system. Revisit after
the answerer and judge prompts are separated into "recall" and "advise" modes —
which is a P7/P8 concern, since the service will need both anyway.
