# Assistant recommendations are a third of the context and never carry an answer

**Measured. A context-economy argument, not an accuracy one — and not acted on.**

The counterfactual found that 69.5% of what batch-1 extraction adds begins "The
assistant recommended", and that two of its four non-conversions were caused by
material of that kind. The planned response was a memory taxonomy — `memory_kind`
alongside `source_role`, with recommendations downweighted or skipped at ingestion.

Before encoding that rule, it was measured.

## How much of the system is this

`scope=recommendation` already exists and 2,277 of its 2,278 rows come from the
assistant, so no new field is needed to ask the question.

| | share |
|---|---:|
| of the store | 2,278 of 6,233 — **36.5%** |
| of the answerer's retrieved context, across all fifty questions | 298 of 1,000 slots — **29.8%** |

Roughly a third of every answer's context budget.

## Do they carry answers

Eighteen dev50 questions have a gold answer made of words rather than a computed
figure, so "is the answer in this memory" is a question that can be asked of them.

| | |
|---|---:|
| questions where a supplied recommendation covers ≥50% of the gold vocabulary | **0 of 18** |

None. And they do not predict failure either — the recommendation share of context
is **30% on the questions answered correctly and 29% on the ones failed**. They are
neither carrying answers nor visibly crowding them out.

## The case that looked like a reason to keep them

The obvious objection was that this project's headline demo *is* a question about
an assistant recommendation, and `single-session-assistant` scores 6/6. Checking
how those six were actually answered inverts it:

| question | how it was answered | recommendations in its context |
|---|---|---:|
| `1b9b7252` | archive_wide | **17 / 20** |
| `41275add` (Mayo) | source_local | 14 / 20 |
| `8aef76bc` | archive_wide | 10 / 20 |
| `e8a79c70` | archive_wide | 11 / 20 |
| `fca762bc` | source_local | 14 / 20 |
| `4388e9dd` | from memory alone | 7 / 20 |

**Five of the six are answered from the raw archive, not from a recommendation
memory.** `1b9b7252` spent 85% of its context on recommendations, and the answerer
still reported `no_evidence` and went to the archive.

So recommendation memories flood the context on exactly the questions about
assistant recommendations, fail to answer them, and the fallback rescues them. The
earlier warning in `counterfactual-qa.md` — that downweighting them would break the
Mayo case — was wrong, and is corrected here.

## Would removing them help

Simulated for the failures by re-ranking with `scope=recommendation` excluded and
reading what enters the top 20. Twenty-nine memories are promoted across six
questions, and they are not the missing facts:

```
80ec1f4f  + [shared_context] The Canon 7s camera was produced from 1965 to 1967 ...
          + [shared_context] The 1 oz Platinum Maple Leaf coin had a mintage of 50,000 ...
4adc0475  + [shared_context] Open Grid Services Architecture (OGSA) provides security ...
73d42213  + [shared_context] Samantha and Jack are siblings visiting Sunnyside Naturist Camp.
```

Freeing the slots promotes different noise. The facts these questions need are not
ranked below the recommendations — they are **absent from the store** (the
batching problem) or **filtered before ranking** (`92a0aa75`'s supersession). Rank
is not the binding constraint.

## Decision

Not acted on. The evidence supports one claim and not the other:

* **True:** a third of every context is spent on memories that never answered a
  question on this set. That is a real cost and a reason to revisit what is worth
  storing.
* **Not shown:** that removing them improves accuracy. The simulation says the
  freed slots fill with other material that is equally irrelevant.

A `memory_kind` taxonomy, an ingestion filter and a re-ingest are a large change
to buy a context saving with no measured accuracy effect, on a set where the
measurement floor is three questions anyway. If it is done later it should be for
cost and store size, argued as such, and not presented as an accuracy fix.

## Limits

Fifty development questions, and "carries the answer" is vocabulary overlap against
a gold string — it works for lookup answers and is meaningless for computed ones,
which is why it was restricted to the eighteen that have text golds. The
crowding-out simulation re-ranks but does not re-run the answerer, so it shows what
would be *supplied*, not what would be *answered*.
