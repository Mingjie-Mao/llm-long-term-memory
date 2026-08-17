# Changing `since` to `on` fixed neither question it was aimed at

**Measured and not shipped.**

The fact-lineage audit found two dev50 failures where the store held the right
date, resolved correctly, and the answerer was shown it. The context read:

```
- The user upgraded their road bike pedals to Shimano Ultegra clipless
  pedals today. (since 2023-03-19)
- The user fixed a flat tire on their mountain bike on March 15, 2023.
  (since 2023-03-15)
```

The answerer reported zero days between the two events. The gold is four.

Two things about that rendering looked wrong, and one of them was testable for
free: `since` frames a point occurrence as the beginning of an ongoing state, and
`scope` already distinguishes them — both rows above carry `scope=event`, and the
resolver had already turned that "today" into 2023-03-19.

## The change

`render_memory` rendered `(on <date>)` for `scope=event` and left `(since <date>)`
for everything else. Superseded memories kept the two-ended window. Stores written
before `scope` existed were unaffected.

## The result

| | dev50 |
|---|---:|
| before | **36/50** |
| after | **33/50** |
| fixed | **0** |
| broke | 3 |
| exact McNemar | p = 0.25 |

**The aggregate is uninformative.** Re-running an unchanged configuration on this
setup has moved a variant between 48.0% and 54.0% — three questions
(engineering report §7). A three-question move is exactly the noise floor, so this
comparison cannot separate the change from a repeat run.

**The mechanism check is what settles it.** Both target questions answered exactly
as before, near word for word:

> You fixed your mountain bike and upgraded your road bike pedals on the exact
> same day, March 15, 2023, so 0 days passed between the two events.

The annotation now said `(on 2023-03-19)` and the answer was unchanged. The
answerer is not misreading the preposition. It is ignoring the annotation and
trusting the sentence.

So the change was reverted: a hypothesis falsified on its own target cases, with no
measurable benefit and a negative point estimate, is not worth the risk of
shipping.

## What this says about the remaining plan

The real defect is still there, one layer in: the prose says "today" while the
resolved date sits in the same row. The fix has to reach the content, not the
annotation — either by resolving deictics at extraction time or by substituting
them at render time. Neither was attempted here.

More importantly, **dev50 cannot measure changes of this size.** Anything worth
three questions or fewer is inside the repeat-run spread, and several items on the
current roadmap — rendering, retrieval weighting, a reasoning router for four
questions — are in that range. They have to be validated the way this one was: a
named mechanism check on the specific questions the change is supposed to affect,
with dev50 used only as a guard against large breakage, never as the evidence that
a small change worked.

The three questions this run "broke" are not analysed for the same reason. At this
noise floor they are not evidence of anything.
