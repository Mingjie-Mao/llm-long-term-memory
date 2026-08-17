# The extractor does not soften numbers, so typed exact-value fields were dropped

**Measured and not built.**

Six of the fourteen dev50 failures were classified as losing an identifying detail
— ten days in Hawaii, two assists, a 416-page novel, two hours to the clinic. The
planned remedy was typed slots: store `value`, `value_raw` and `value_type`
alongside the prose so that a number, date or proper noun survives verbatim, on
the principle that an LLM may summarise meaning but not identifiers.

That assumed the extractor was rewriting figures it had already found. It is not.

## What was measured

Every memory carries a provenance span — the exact character range of the source
turn it came from — so the source text for each memory is known rather than
guessed at. Three measurements, each correcting the previous one.

| measurement | result | why it was wrong |
|---|---:|---|
| numbers in the span that the memory omits | 47.6% | the span is often a whole table or list, and one memory covers one fact of it |
| same, restricted to spans under 120 chars | 36.6% | still per-memory, and one span routinely produces several memories |
| same, unioned across every memory from the span | 32.7% | counts facts never extracted at all, which is coverage, not fidelity |

The third number is real but it answers a different question. Reading the cases
shows what it is made of: list numerals (`1.` in "here are ten examples"), spans
truncated mid-figure (`and 1.` where the source said 1.5%), and figures belonging
to a *different* fact in the same sentence — "clean driving record for the past 5
years" sitting beside "a 2018 silver Honda Civic", where the memory records the
car and nobody extracted the driving record. That last kind is a missing fact, not
a mangled one, and no typed field would have kept it.

**Absent numbers and softened numbers look identical to that measurement.** So it
cannot size the problem the typed fields were for.

## The measurement that can

The signature of the failure typed fields would prevent is different: the memory
*does* cover the fact and replaces an exact quantity with a vague one. That is
detectable — a hedging quantifier in the content whose source span held a figure
the memory did not keep.

| | |
|---|---:|
| memories using a vague quantifier at all | 261 of 6,233 (4.2%) |
| of those, ones whose span held an exact figure they dropped | **9** |
| as a share of the store | **0.14%** |

And most of the nine are false positives: "about" as a preposition, not as a
hedge — *"a story **about** a basketball team"*, *"a dream **about** being a human
in a giant-sized world"*. The true rate is near zero.

## Where the one real case came from

`4adc0475` is the case that motivated this, and it points the other way:

| | |
|---|---|
| production, batch 15 | "The user plays in a recreational indoor soccer league and has scored **3 goals**." |
| batch 1 re-extraction | "The user has scored **several goals** and two assists in their indoor soccer league." |

The shipped configuration kept the figure. The smaller batch softened it while
recovering the assists — one operand gained, another degraded, in the same
sentence.

## What this changes

**Typed exact-value fields are not worth building.** They would address something
that happens in roughly one memory in seven hundred, at the cost of a schema
change, a prompt change and a full re-ingest to evaluate.

**The six "dropped identifier" failures are coverage failures.** The fact was
never extracted, not extracted and then mangled. That is the batching problem
already measured, and it has a different remedy — one that the counterfactual
showed converts at two in six.

**And it is one more reason not to move to batch 1.** The only observed
number-softening in this project came from that arm. Coverage and fidelity are not
the same axis, and the cheaper-per-session configuration is the one that got a
figure wrong.

## Limits

Hedging is detected by a fixed word list against the provenance span, so a
paraphrase that keeps a number but changes it — "3 goals" becoming "4 goals" —
would not be caught here at all, and nothing in this project has measured that.
The 0.14% is a floor on one specific failure mode, not a fidelity score.
