# Nine of fourteen failures are lost at extraction, and none at retrieval

Four planned fixes in a row were aimed at the wrong layer: the renderer hypothesis
was falsified on its own target questions, the typed-value work turned out to
address 0.14% of the store, filtering assistant recommendations frees a third of
the context and promotes different noise, and a supersession fix stopped a real
loss without answering the question that motivated it. Each time the shape was the
same — see a symptom, name a module, change it.

So this assigns each remaining failure to the **first** stage that loses it, and
uses those counts as a ceiling on what each module could be worth. First and not
every-stage-that-looks-wrong: "the number is not in memory" is equally consistent
with never extracting it and with extracting then mangling it, and separating
those took three measurements last time.

Produced by `scripts/stage_oracle.py`; assignments are by hand from
[the lineage audit](fact-lineage-dev50.md), because every automated classification
attempted in this project has been confidently wrong.

## Where they break

| stage | n | |
|---|---:|---|
| S0 source | 0 | the fact is not in the conversation |
| **S1 extraction** | **9** | it never became a memory |
| S2 lifecycle | 1 | it became one, then was merged or superseded away |
| S3 eligibility | 0 | filtered before ranking |
| **S4 retrieval** | **0** | eligible, never reached the context |
| S5 reasoning | 4 | everything supplied and correct, answer still wrong |

| question | stage | what is lost |
|---|---|---|
| `edced276` | S1 | the Hawaii trip's ten-day duration |
| `4adc0475` | S1 | the two assists |
| `37f165cf` | S1 | the 416-page figure |
| `73d42213` | S1 | the two-hour journey — at any batch size |
| `c9f37c46` | S1 | the open mic night; its gold session yielded nothing |
| `gpt4_7abb270c` | S1 | the sixth museum |
| `80ec1f4f` | S1 | the February gallery visit; its gold session yielded nothing |
| `gpt4_fa19884d` | S1 | the user's own statement — only the replies to it were kept |
| `0edc2aef` | S1 | the user's transferable preference |
| `92a0aa75` | S2 | extracted, then folded away as a restatement |
| `gpt4_e414231e` | S5 | both dates supplied and annotated; reported as the same day |
| `gpt4_d6585ce8` | S5 | every event and date supplied; ordering still wrong |
| `9a707b81` | S5 | the date supplied and correct; anchored on the wrong day |
| `71017277` | S5 | the fact supplied and correct; not accepted as jewellery |

## Oracle ceilings

The most each module could be worth, if it were perfect:

| module | ceiling | |
|---|---:|---|
| extraction | **9** | every S1, and only if the fact alone is enough |
| **retrieval** | **0** | no failure is lost at S3 or S4 |
| reasoning | 4 | every S5 |

**Hybrid retrieval has a ceiling of zero on this set.** Not "small" — zero. Nothing
is currently ranked out of reach; the facts these questions need are absent or
were removed before ranking. That settles the item without an experiment and
without appealing to what other systems do. It also disposes of the observation
that four of five retrieval signals are weighted at zero
(`docs/ENGINEERING_REPORT.md` §5): the weighting is a real inaccuracy in the
documentation and not a cause of any failure here.

The extraction ceiling of 9 is an upper bound and is known not to be reached. The
[counterfactual](counterfactual-qa.md) recovered six real facts and fixed two
answers — a conversion of 2 in 6 — because recovering a fact does not guarantee
retrieval surfaces it or the answerer uses it.

## What the headline accuracy is made of

| | | |
|---|---:|---:|
| final accuracy | 36/50 | **72.0%** |
| answered from memory alone | 27/50 | **54.0%** |
| rescued by the raw archive | 9/50 | **18.0%** |

The fallback fired 16 times and produced the right answer 9 of those (56.2%).

This split matters more than it looks. **Structured memory on its own scores 54.0%,
which is exactly `naive_rag`'s 54.0% and below `full_context`'s 56.0%.** What puts
the product ahead of both is the raw-conversation archive, not the memory
representation. Reporting 72% without the split invites the reading that the
memory system answered 72% of the questions, and for a quarter of the correct ones
that is not what happened.

It is also consistent with the audit: if nine of fourteen failures are facts that
never became memories, the memory layer is doing less work than the headline
suggests, and the archive is absorbing the difference.

## What this settles, and what it does not

**Settled.** Retrieval work is not justified on this evidence — hybrid ranking,
reranking, a larger `top_k`, entity or date boosting. Their shared ceiling is zero
questions. Anything spent there is spent on a stage that is not losing anything.

**Not settled.** Whether the extraction ceiling of nine is reachable, and at what
cost. The one measurement available says two in six, and it was taken on questions
chosen for failing, which flatters it rather than the reverse.

**Open.** Two oracles remain unrun and would sharpen both numbers: injecting each
missing fact as a memory and running the real pipeline (the true extraction
ceiling), and handing the answerer the gold evidence alone (whether the four S5
failures are really reasoning). Oracle B — a perfect retriever — is deliberately
not run: a ceiling of zero needs no experiment, and spending quota to confirm it
would measure the answerer's run-to-run noise rather than retrieval.

## Limits

Fourteen questions on a development set, hand-classified by one reader, using a
`STAGE` table checked into the script so it can be argued with. The stage
boundaries are not always sharp — `0edc2aef` is filed S1 because the user's
preference was never extracted, but it is equally a failure to *apply* a
preference the answerer could see. And "first failing stage" hides later ones: a
question fixed at S1 may then fail at S5, which is exactly what the 2-in-6
conversion suggests is happening.
