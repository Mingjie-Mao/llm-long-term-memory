# Anchoring memories to the right turn — v2 — offline pre-registration

Date: 2026-10-01 (Australia/Sydney). Written before the tool was run.

## Classification

**DEVELOPMENT, zero calls.** Measured on `train150`'s store and corpus. No answer is read;
no gold label is used — the reference is built from the memories and the raw turns.

## Why

A hand-read train150 miss had "The user completed an undergraduate degree in Computer
Science from UCLA." anchored to turn 1, where the user compares Stanford, Berkeley and
Carnegie Mellon, while "UCLA" is said only in turn 4
(`results/memory-as-index-offline-decision.md`). `provenance.source_span_for` scores a
sentence by the number of tokens it shares with the fact, plus 2 per shared number, and
breaks ties towards the earlier turn. "the", "user", "in" and "from" count as much as
"ucla". Zep avoids the problem by keeping each fact linked to the episode it was
extracted from; batched extraction here cannot, so the anchor has to be found.

## Reference (no gold label)

A memory is **checkable** when its content contains a specific — a number, a quantity,
or a multi-word proper name, as `ingest.fidelity._extract_facets` detects them — that
appears, case-insensitively, in exactly one turn of its source session. That turn is
the reference anchor. The reference is narrow by construction (it only covers memories
with a unique specific) and says nothing about memories without one.

## Arms

| arm | turn chosen |
|---|---|
| `stored` | the `source_turn_index` already in the store |
| `v1` | `source_span_for` recomputed — checks that `stored` is what v1 produces |
| `v2` | per sentence: Σ idf over shared content tokens (idf within the session's sentences, stop-words dropped), plus a bonus of 3 × idf for every specific of the fact the sentence contains; ties to the earlier turn |

## Decision rule

`provenance` switches to v2 for new ingests if, on the checkable memories:

1. v2's accuracy is at least **3 points** above v1's; and
2. of the memories v1 anchors correctly, v2 keeps at least **99%** correct.

Existing stores are not rewritten by this; re-anchoring a copy is a separate step.

## What this cannot show

That a better anchor changes any answer. Anchors matter to hydration and to using
memories as keys for turns; neither is measured here.

## Addendum after the first run — 2026-10-01

v2 passed both rules on train150 (87.8% -> 96.5%, keeping 99.1% of v1's correct
anchors) and on dev100 (87.8% -> 96.2%). Two things were then found, and are recorded
rather than folded in:

- **The reference flatters v2.** It picks the turn carrying a unique specific, and v2
  rewards exactly that. With v2's specifics bonus removed and the reference unchanged,
  accuracy is 93.6% (train150) and 93.4% (dev100): about two thirds of the gain does not
  depend on the bonus.
- **The motivating case was not fixed.** On the real UCLA memory v2 chose turn 11, an
  assistant's sample essay that repeats "UCLA", "Computer Science" and "undergrad";
  the user says it in turn 4. "UCLA" is in seven turns, so the reference cannot check
  this memory at all.

**v3**, defined after seeing that case: v2 restricted to the turns of the memory's own
speaker (`source_role`), falling back to every turn when that speaker shares nothing
with the fact — the provenance rule Zep enforces by linking a fact to its episode. v3 is
held to the same two rules against v1 on both stores, and it is reported next to v2 so
the post-hoc step is visible. New ingests use whichever passes; if both do, v3, because
it fixes the case that motivated the work.

**Result.** v3: 95.6% (train150) and 95.2% (dev100), keeping **97.9%** and **98.2%** of
v1's correct anchors — below the registered 99%. v3 fails; v2 passes. New ingests use
**v2** (`provenance-v2`). The UCLA memory is not fixed by the rule that passed, and that
stays recorded. `source_span_for_v2(..., role=)` is kept for measurement, unused by
ingestion.
