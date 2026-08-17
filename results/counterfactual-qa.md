# Recovering the lost facts fixed two answers out of six

The batch-position work established that extraction at fifteen sessions per
request loses memories, causally and heavily. This asks the only question that
decides what to do about it: **does putting the lost facts back answer anything?**

Run by `scripts/counterfactual_qa.py` over `stores/cf-overlay.db`, which is the
clean P10 store with the memories eight gold sessions produced at batch 15 removed
and the ones they produce at batch 1 in their place. Same turns, same other
namespaces, same config, same retrieval, same answerer and judge. The production
store was copied, never modified.

## Result

| question | fact recovered at batch 1 | production | overlay | |
|---|---|---|---|---|
| `80ec1f4f` | the February gallery visit, with its date | ✗ | **✓** | **fixed** |
| `edced276` | the Hawaii trip lasted 10 days | ✗ | **✓** | **fixed** |
| `37f165cf` | the second novel was 416 pages | ✗ | ✗ | |
| `4adc0475` | the user had two assists | ✗ | ✗ | |
| `c9f37c46` | the user attended an open mic night | ✗ | ✗ | |
| `gpt4_7abb270c` | the sixth museum visit | ✗ | ✗ | |
| `73d42213` | **not recovered** — control | ✗ | ✗ | as expected |

**Six facts recovered, two answers fixed.** The control did not move, so the
overlay changed what it was meant to change and nothing else.

## Why the other four did not flip

Each for a different reason, and only two of them are about noise.

| question | what went wrong instead |
|---|---|
| `37f165cf` | recovering "the user recently finished a 416-page novel" also produced **three assistant recommendations for books that are themselves 416 pages** |
| `gpt4_7abb270c` | the sixth museum came back, buried in 68 memories of which 52 begin "The assistant recommended" |
| `4adc0475` | the assists came back and the goals degraded: production had "has scored 3 goals", batch 1 wrote "has scored **several** goals and two assists". One operand gained, the other lost |
| `c9f37c46` | every fact present and correct, and still answered wrong — a reasoning failure, not a supply one |

`4adc0475` is the one worth staring at. **Batch 1 is not uniformly better.** It
recovered a missing quantity and destroyed a present one in the same memory.
Coverage and fidelity are not the same axis, and extracting more does not
automatically extract more precisely.

## What the extra yield is made of

Across the eight sessions:

| | batch 15 | batch 1 | ratio |
|---|---:|---:|---:|
| memories | 19 | 154 | **8.1x** |
| of which "The assistant recommended …" | — | 107 (**69.5%**) | |
| **user facts** | 19 | **47** | **2.5x** |

The pilot's headline of 12.7 memories per session against 2.7 is inflated by
recommendation lists — art blog URLs, running shoe models, seventeen Japanese
restaurants. This also explains why the free quality floor saw nothing wrong:
those memories are genuinely grounded in the source and genuinely not duplicates
of each other. They are mostly just not things anyone asks about.

## What this changes

The remedy is not a smaller batch size.

Going from batch 15 to batch 5 costs 400 → 1,000 extraction requests for the
corpus, and batch 1 costs 4,800. The conversion measured here is 2 of 6, on
questions **selected because they were failing** — a random sample would convert
less. Meanwhile two of the four non-conversions were caused by material batch 1
added.

So the problem is not that extraction is too small. It is that extraction does not
distinguish between what is worth keeping and what is not, and a smaller batch
applies that same undiscriminating policy more thoroughly. Spending twelve times
the quota to execute an unexamined policy more completely is the wrong order of
operations.

> **A caution against the obvious fix**, since corrected. "Downweight assistant
> recommendations" was rejected here on the grounds that it would break this
> project's headline demo — the Mayo case is a question *about* an assistant
> recommendation, and `single-session-assistant` scores 6/6. Measuring it
> (`recommendation-utility.md`) inverted that: five of those six are answered
> from the **raw archive**, not from a recommendation memory, and one of them
> spent 85% of its context on recommendations before reporting `no_evidence`.
> The demo was never resting on them. Removing them still is not an accuracy
> fix — the freed slots fill with other irrelevant material — but the reason
> given here was wrong.

## Limits

Seven questions, chosen because they failed, on a development set. Two flips is
two questions. The overlay was built by re-extracting the same eight sessions a
second time and produced memory counts identical to the first run (1→8, 3→12,
12→17, 2→11, 0→19, 0→41, 1→27, 0→19), which says the extractor is close to
deterministic at batch 1 but says nothing about the other fifty-two sessions in
the corpus that were never touched.

`stores/cf-overlay.db` is a diagnostic artifact. Nothing in the product reads it,
and `sessions_per_request` is still 15.
