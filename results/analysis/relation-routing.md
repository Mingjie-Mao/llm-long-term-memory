# Can a question be routed to the relation an exhaustive scan needs?

**Zero provider calls.** Local encoder only, so this was answerable before v4.1 was
scoped rather than after. Tool:
[`tools/relation_routing_probe.py`](../../tools/relation_routing_probe.py); data:
[`relation-routing.json`](relation-routing.json).

## Why it had to be asked first

[`exhaustive-scan.json`](exhaustive-scan.json) measured the prize: scanning one relation
returns **100% of a set's members at 68.5 median tokens**, against top-20 similarity's
**58.3% at 439**. That is what would lift the `count` result, where 43% of probes cannot
be answered from the retrieved context at all.

But that scan arm was **told which relation to scan by the probe's ground truth**. The
routing a real system needs does not exist. Building v4.1 on a ceiling reached by an
oracle would have been the same mistake as `max_turns`: a fix aimed at a step that was
not the problem.

## Two arms, because the obvious one is rigged

The probe's own question is generated *from* the relation name — `visited` becomes
"places has the user visited" — so routing it back is close to matching a string against
itself. That arm is reported as an upper bound and nothing more.

### Templated arm — upper bound: **80.0%** (48/60)

| relation | n | correct |
|---|---:|---:|
| `practises_hobby` | 13 | 13 |
| **`owns`** | **10** | **0** |
| `attended` | 9 | 9 |
| `financial_activity` | 9 | 9 |
| `grows`, `read` | 4, 4 | 4, 4 |
| `visited`, `cooked`, `watched`, **`acquired`** | 2 each | 2, 2, 2, **0** |
| `completed`, `ate_at`, `knows_person` | 1 each | 1, 1, 1 |

**Every single error is one pair.** `owns → acquired` twelve times, `acquired → owns`
twice; nothing else is confused with anything. Merge those two and the upper bound is
60/60.

That is not a routing failure. Buying something makes you its owner, so `owns` and
`acquired` describe one relation that the vocabulary split in two — a **vocabulary design
defect**, found by trying to use the vocabulary rather than by reading it.

### Natural arm — real `train150` questions, and the answer is different

72 real aggregation questions, which nobody wrote with a relation vocabulary in mind.
There is no ground-truth relation for these, so accuracy here would be invented; what is
reportable is the margin between the best and second-best relation, and the routes
themselves.

| margin threshold | routed | abstained | share routed |
|---:|---:|---:|---:|
| 0.00 | 72 | 0 | 100% |
| 0.05 | 41 | 31 | 57% |
| 0.10 | 28 | 44 | 39% |
| 0.15 | 18 | 54 | **25%** |
| 0.20 | 9 | 63 | 12% |

**31 of 72 real questions route at a margin below 0.05** — the top two relations are
effectively tied, which is a coin flip wearing a decision.

## The finding that changes the plan

Reading the confident routes is what settles it. These all cleared a 0.15 margin:

    watched   m=0.26   How many hours did I spend watching documentaries last month?
    read      m=0.23   How many days had passed since I finished reading '...'?
    know_p    m=0.20   How many babies were born to friends and family recently?
    acquired  m=0.18   How many days ago did I buy a smoker?

Every one is **confidently routed and wrong about what is being counted**. The first
counts *hours*, not films. The second and fourth count *days*, not books or purchases —
they are duration questions the aggregation regex caught. The third counts babies, and
there is no `babies` relation in the vocabulary at all; `knows_person` is the nearest
thing that exists, which is exactly why it won.

So the ceiling is not blocked by the router's accuracy. It is blocked by two things the
router cannot fix:

1. **Coverage.** Real questions ask about things the 15 set relations do not name.
2. **Question shape.** "How many days since X" is a duration wearing a count's grammar,
   and no relation is the right answer for it.

## What this licenses, and what it forbids

**Forbidden:** wiring exhaustive scan behind an always-on router. A confident wrong route
is worse than top-20 similarity, because it returns a *complete* set of the *wrong*
relation — and completeness is exactly what would make the answer look trustworthy.

**Licensed, in this order:**

1. **Merge `owns` into `acquired`.** Free, removes 100% of the upper bound's errors, and
   it is the right call independent of routing.
2. **The router must be able to abstain**, and fall back to today's top-k when it does.
   At a 0.15 margin that is 25% routed and 75% abstaining — which is not a disappointment
   but the correct behaviour given coverage.
3. **Measure the scan gain only on the routed slice.** A number pooled over routed and
   abstained questions would be the ceiling diluted by questions the mechanism never
   claimed.
4. **Decide coverage separately.** Whether to grow the vocabulary past 15 set relations
   is a data-layer question, and this probe says nothing about how far it would have to
   grow.

## Limits

- The upper bound arm is 60 probes over 13 relations, and 5 of those relations have one
  or two probes each. It supports "one pair accounts for every error" and not a per-
  relation accuracy.
- The natural arm has **no labels**. Nothing here is a routing accuracy on real
  questions; the margins and the read samples are the evidence, and they are enough for
  the decision above but not for a number.
- The aggregation regex that selected the 72 questions also catches durations, which is
  itself part of the finding rather than a defect in the selection.

---

# Executed: merge, abstain, measure on the routed slice only

## 1. `owns` merged into `acquired` — at the routing layer, not in the map

The merge lives in
[`relation_router.py`](../../src/llm_long_term_memory/retrieve/relation_router.py), not in
`predicate-map.csv`. That is not a shortcut: the map is hashed into the probe set, which
is hashed into the held-out split, so rewriting the CSV would have silently invalidated
the only unseen check v4 has. A routing class that scans `("acquired", "owns")` as a
union achieves what the finding asked for and touches no stored data.

**Templated upper bound, development half, after the merge: 30/30 = 100%, no confusions
left.** Before the merge it was 80% with every error in that one pair.

## 2. The router abstains

`DEFAULT_MARGIN = 0.15`, chosen from the observed margin distribution rather than tuned
for a score. Below it, `Route.relations` is empty and the caller keeps today's retrieval.
An abstaining route still reports what it *would* have said, because a router that hides
its near-misses cannot be audited later.

## 3. The gain, measured only where the router claims the question

Development count probes only — the held-out half is not read.
[`routed-scan-gain.json`](routed-scan-gain.json).

| slice | n | share | route correct | top-k complete | scan complete |
|---|---:|---:|---:|---:|---:|
| **routed** | 16 | 53% | **100%** | 69% | **100%** |
| abstained | 14 | 47% | — | 29% | unchanged by construction |

On the slice the mechanism claims, it does exactly what the oracle arm promised: every
member of the set is returned, closing a 31-point completeness gap to nothing.

## The uncomfortable half of that table

Across all 30 probes, evidence completeness goes from **50.0% to 66.7%**. But look at
where the remaining incompleteness sits:

    routed      16 probes  ->  0 incomplete after the scan
    abstained   14 probes  -> 10 incomplete, unchanged

**The router abstains precisely where top-k is worst.** It claims questions that were
already 69% complete and declines the ones sitting at 29%. The mechanism helps where help
was least needed.

That is not an argument for lowering the margin. The confident-but-wrong routes read
earlier are what a lower margin buys, and a complete set of the wrong relation is worse
than an incomplete set of the right one. It is an argument that **coverage, not routing,
is the binding constraint** — the abstained questions are largely ones whose relation the
vocabulary does not name. Item 4 of the plan stands: whether to grow the vocabulary past
15 set relations is a data-layer decision this probe cannot make.

## Limits

- **16 and 14 probes.** Every percentage in that table is one or two probes wide. It
  supports the direction and the shape of the residue, not the size of either.
- "Complete" here means the evidence a correct count needs was returned. It is not an
  answer, and no answer was generated: nothing in this section cost a provider call.
- The upper-bound arm remains an upper bound. 100% on templated questions says the merge
  removed the confusion, not that routing works on real ones — the natural arm above is
  the evidence for that, and it says 25% at this margin.
