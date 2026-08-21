# The answerer disagrees with itself, and it was never measured

**Status: complete.** 5 questions × 3 context arms × 3 repeats = 45 answers, one
store, one config, `scripts/context_arms.py`, raw in
`results/raw/context-arms.json`.

This started as a question about context organisation and answered a different
one. The headline is the reproduction check that failed:

> **`flat20` reproduces the production configuration exactly — same store, same
> retrieval, same `top_k`, same answerer — and does not reproduce the held-out
> run's answers.**

`71315a70` was wrong in the held-out run and is right in 3 of 3 reruns.
`69fee5aa` was wrong there and is right in 1 of 3 here. `07741c45` answered
"under the bed", which no memory says, and that hallucination did not recur in
any of nine reruns of that question.

Nothing in this project has ever measured that variance, and every paired result
in it is a single run.

## What was run

| arm | context |
|---|---|
| `flat20` | production: `top_k` 20, retrieval rank order |
| `flatN` | same count as `coherent`, retrieval rank order, still scattered |
| `coherent` | the gold sessions' memories, event order, nothing else |

`flatN` exists to separate the two things
[the knowledge-update oracle](raw/ku-oracle.json) changed at once — session
coherence and a cut from twenty memories to five or seven.

## Results

Scored by reading. Each cell is correct-out-of-three.

| question | gold | `flat20` | `flatN` | `coherent` | held-out run |
|---|---|---:|---:|---:|---|
| `71315a70` | 10-12 hours | **3/3** | 1/3 | **3/3** | wrong |
| `69fee5aa` | 38 | 1/3 | 1/3 | **3/3** | wrong |
| `07741c45` | in a shoe rack in my closet | 0/3 | 0/3 | 0/3 † | wrong |
| `0977f2af` | Instant Pot | 0/3 | 0/3 | 0/3 | wrong |
| `031748ae_abs` | decline — false premise | 0/3 | 0/3 | 0/3 | wrong |

† `coherent` names the shoe rack in all three, but frames it as a plan: *"You
plan to store your old sneakers in a shoe rack once you organize your closet."*
The memory it has says the user **plans** to. Refusing to report a plan as a
current location is defensible, and the gold treats it as one. Counted wrong
here; worth flagging as a question whose gold is arguable rather than a system
failure.

### 1. Fewer distractors is not the mechanism

`flatN` is the cheap hypothesis — if trimming `top_k` produced the oracle's
flips, the fix costs nothing and needs no new machinery.

It does not. On `71315a70` **`flatN` is worse than `flat20`**, 1 of 3 against 3
of 3, at a third of the context. Cutting k while keeping the ranking scattered
made the answer less reliable, not more. The variable is not how many memories
are supplied.

This also reproduces, on a different question type, the dev50 finding that
re-answering `gpt4_e414231e` at k of 20, 10, 5 and 3 gave the same wrong answer
every time.

### 2. What `coherent` buys is consistency

| arm | cells where all three runs agreed |
|---|---:|
| `flat20` | 3 of 5 |
| `flatN` | 3 of 5 |
| **`coherent`** | **5 of 5** |

`coherent` never disagreed with itself. The flat arms disagreed on 2 of 5
questions each. On the two answerable questions `coherent` is 3/3 and 3/3, while
`flat20` is 3/3 and 1/3.

So the effect is real and it is **not** mainly an accuracy effect — it is a
variance effect. A scattered ranking makes the answerer's output depend on the
run; a coherent slice of the source sessions makes it depend on the evidence.
That is a stronger reason to build P6 than "it answers more questions", and it is
not the reason P6 was proposed.

Two caveats that keep this from being a finding about the product: two
answerable questions is not a sample, and `coherent` is an oracle — it uses the
gold session ids. A shippable version has to find those sessions from retrieval,
and whether it can is untested.

### 3. Two failures no context arrangement reaches

`0977f2af` and `031748ae_abs` are 0/3 in every arm with perfect agreement. The
first needs a fact extraction never produced — the Instant Pot appears only as
"plans to make Jjimdak in their Instant Pot", never as a purchase. The second
needs the answerer to notice that the question presupposes a role the user never
held; it answers "You lead 4 engineers" nine times out of nine.

Stable failures are useful: they are the ones a single run measures correctly.

## What this does to everything else

Every headline in this project is a single run.

| result | what the variance does to it |
|---|---|
| held-out **70.0%** vs dev50 **72.0%** | the -2.0pp gap is 2 questions. At the flip rates seen here, one run of 100 questions carries a run component nobody has measured |
| fallback **p = 0.022** | 9W-1L, single run. One flipped question moves it to 8W-2L, p = 0.109 |
| oracle A's **4 of 9**, the counterfactual's **2 of 6** | single run each, on questions selected for failing — the population most likely to be unstable |
| the [batch-size pre-registration](prereg-batch-size.md) | its power table models sampling variance only. True power is lower than stated, and the case for coverage as the primary endpoint is stronger than it was written |

The project had noticed without quantifying: `record_golden.py` refuses to record
a demo whose runs disagree about whether the archive was needed, and
`live-regression-v2` reports "identical across three runs". Both treat agreement
as a property worth checking. Neither turned it into a number.

## What follows

- **Repeats are not optional for small comparisons.** Any paired result on tens
  of questions needs k runs and a stated agreement rate, or it reports noise.
- **The cheapest fix to the batch experiment** is to run its QA endpoints three
  times and vote, which costs 3x the answerer quota it was already going to
  spend and nothing in extractor quota — the expensive half.
- **Measure the variance at scale before trusting any of this.** Three repeats of
  the full held-out hundred would say what the ±band on 70.0% actually is. That
  is ~600 answerer calls and needs no new store.
- **P6's justification changes** from "coherent context answers more" to
  "coherent context answers the same thing twice". Whether retrieval can find the
  right sessions without the gold labels is the experiment that decides it.
