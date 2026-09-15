# Ten of fourteen failures are lost at extraction, and none at retrieval

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
| **S1 extraction** | **10** | it never became a memory |
| S2 lifecycle | 1 | it became one, then was merged or superseded away |
| S3 eligibility | 0 | filtered before ranking |
| **S4 retrieval** | **0** | eligible, never reached the context |
| S4b composition | 1 | in context and top-ranked, still not used |
| S5 reasoning | 2 | everything supplied and correct, answer still wrong |

Two of the four originally filed S5 moved once the oracles read what was actually
supplied — see below. First-stage classification by inspection was wrong twice out
of fourteen, which is worth knowing about every other table in this project.

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
| `gpt4_d6585ce8` | S1 | the free outdoor concert — only the replies to it were kept |
| `gpt4_e414231e` | S4b | both dates ranked 1 and 2, and still read as the same day |
| `9a707b81` | S5 | the date supplied and correct; anchored on the wrong day |
| `71017277` | S5 | the fact supplied and correct; not accepted as jewellery |

## Oracle ceilings

The most each module could be worth, if it were perfect:

| module | ceiling | measured | |
|---|---:|---:|---|
| extraction | 10 | **4 fixed of 9 tried** | oracle A: inject the fact, run the real pipeline |
| **retrieval** | **0** | not run | no failure is lost at S3 or S4 |
| composition | 1 | 3 of 4 runs | oracle C on `gpt4_e414231e` |
| reasoning | 2 | 0 | oracle C on the rest |

**Oracle A fixed four of nine.** `4adc0475`, `80ec1f4f`, `c9f37c46` and `edced276`
came back when their missing fact was injected, correctly worded, and the pipeline
was left otherwise untouched. `73d42213`, `37f165cf`, `0edc2aef`,
`gpt4_7abb270c` and `gpt4_fa19884d` did not — so five of the S1 questions have a
second problem behind the missing fact, and perfect extraction alone would not
reach them. A conversion of 4 in 9 against the counterfactual's 2 in 6, both on
questions chosen for failing.

**Oracle C produced one surprise and one correction.** `gpt4_e414231e` — the
question the renderer change was built for and failed to move — answers correctly
in 3 of 4 runs when given the two gold sessions' memories:

> You fixed your mountain bike on March 15, 2023, and upgraded your road bike's
> pedals on **March 19, 2023, which is 4 days later**.

The answerer can do the arithmetic. It fails in production with the *same two
memories ranked first and second*, which is why this is filed S4b rather than S5:
not a reasoning failure, and not a retrieval failure either.

The obvious explanation — too much context — was tested and is wrong. Re-answering
at `top_k` of 20, 10, 5 and 3 gives the same wrong "same day, March 15" every
time, and at k=3 the context is exactly the two needed memories plus one. Oracle
C's six-memory set is a *superset* of that and does better. Fewer is not the
variable. What differs is that oracle C supplies a coherent slice of two
conversations while `top_k` supplies a scattered ranking across many — a
hypothesis, not a finding, and untested.

`gpt4_d6585ce8` was misfiled. Oracle C showed no memory for the "free outdoor
concert series in the park" that the gold answer lists; the user states it in the
raw turns and only the assistant's replies were extracted. It is S1.

**Hybrid retrieval has a ceiling of zero on this set.** Not "small" — zero. Nothing
is currently ranked out of reach; the facts these questions need are absent or
were removed before ranking. That settles the item without an experiment and
without appealing to what other systems do. It also disposes of the observation
that four of five retrieval signals are weighted at zero
(`docs/history/ENGINEERING_REPORT.md` §5): the weighting is a real inaccuracy in the
documentation and not a cause of any failure here.

The extraction ceiling of 10 is an upper bound and is known not to be reached.
Oracle A converts 4 of 9 and the [counterfactual](counterfactual-qa.md) converted
2 of 6, because recovering a fact does not guarantee retrieval surfaces it or the
answerer uses it.

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

**Not settled.** Whether the extraction ceiling of ten is reachable, and at what
cost. Two measurements now exist, 4 of 9 and 2 of 6, both on questions chosen for
failing, which flatters them rather than the reverse. Five S1 questions survived
having their missing fact handed to them, so they carry a second defect behind the
first.

**Open.** What makes oracle C's context work where a top-ranked one does not.
Fewer memories is ruled out; a coherent slice of two conversations against a
scattered ranking is the remaining guess and has not been tested. Oracle B — a
perfect retriever — is deliberately not run: a ceiling of zero needs no
experiment, and spending quota to confirm it
would measure the answerer's run-to-run noise rather than retrieval.

## Limits

Fourteen questions on a development set, hand-classified by one reader, using a
`STAGE` table checked into the script so it can be argued with. The stage
boundaries are not always sharp — `0edc2aef` is filed S1 because the user's
preference was never extracted, but it is equally a failure to *apply* a
preference the answerer could see. And "first failing stage" hides later ones: a
question fixed at S1 may then fail at S5, which is exactly what the 2-in-6
conversion suggests is happening.


## The second defect behind the five oracle A did not fix

All five had the injected fact in their context and failed anyway, so the audit's
"first failing stage" was hiding a second one in each.

| question | what stopped it the second time |
|---|---|
| `73d42213` | **a second missing fact** — the 9:00 arrival needs a 7:00 departure too, and the injection supplied only the two-hour journey. Still S1 |
| `37f165cf` | **S5** — 341, 440 and the injected 416 were all in context and the answer was "I do not have a record of the specific books or page counts" |
| `0edc2aef` | **S5** — the preference was supplied and the answerer declined to transfer it, replying that the trip was to Seattle |
| `gpt4_7abb270c` | **S5** — six museums listed in order, with "Modern Art Gallery" substituted for the Museum of Contemporary Art |
| `gpt4_fa19884d` | the gold is "a bluegrass band that features a banjo player", which names no artist. Likely a benchmark limit |

**So the reasoning burden is larger than S5 = 2.** The audit records only the first
stage that loses a question; behind four of these five there is an S5 waiting. A
perfect extractor converts four of nine and then hands four more to a wall it
cannot help with, which is the honest reading of the extraction ceiling: not ten
questions, closer to five.

## Coverage-aware extraction has nothing to select on

The plan after this was adaptive extraction — ingest at batch 15, detect sessions
at risk of having lost something, and re-extract only those at batch 1. The pilot's
matched cohort already has the data to test whether such a detector could exist.

It cannot. Every stratum loses at roughly the same rate:

| stratum | batch 1 / batch 15 |
|---|---:|
| under 8 turns | 7.0x |
| 8–11 turns | 5.5x |
| 12–15 turns | 4.5x |
| 16+ turns | 5.2x |
| historically zero-yield | 6.0x |
| historically normal | 4.6x |

And the signal a detector would most obviously use runs the wrong way:

| yield at batch 15 | yield at batch 1 |
|---:|---:|
| 0 memories | 11 |
| 1–2 | 10 |
| **5 or more** | **17** |

**The sessions that did best at batch 15 gain the most at batch 1.** A detector
that re-extracts the suspiciously quiet sessions would skip exactly the ones with
the most to recover. The loss is not concentrated in an identifiable subset — it is
roughly five-fold everywhere — so there is no risk signal to route on, and
"selective retry" reduces to "re-extract everything", which is the 4,800-request
option already rejected.


## The reasoning module's oracle is smaller than it looks

Handing the answerer only the operands, with nothing else in context:

| question | result | |
|---|---|---|
| `gpt4_7abb270c` | **correct order** | Science Museum, Museum of Contemporary Art, Metropolitan, … |
| `37f165cf` | "440 pages and 416 pages" | listed, not summed; gold is 856 |
| `9a707b81` | "26 days ago, on March 20, 2022" | anchored on the question date, not the referenced event |

The model can order six dated events and can subtract dates. It fails at choosing
*which* operands and *which* anchor. So a router dispatching to deterministic
sum/duration/ordering operators addresses the half that already works: ordering
needs no operator, the date question needs the right anchor before any subtraction
helps, and only the sum is a case where a deterministic operator would clearly fix
the answer.

That is one question of the six, and it arrives after the extraction work that
would have to supply the operands in the first place.
