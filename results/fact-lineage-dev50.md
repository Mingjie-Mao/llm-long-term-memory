# Where the fourteen dev50 failures actually lose their answer

Every one of the fourteen questions the product arm gets wrong has
`source_session_recalled` set. The statistic says the folder holding the answer was
opened; it does not say the right sheet came out. This traces each required fact
through four layers — raw turns, extracted memory, the context the answerer
received, the answer — and finds where it disappears.

Produced with `scripts/fact_lineage.py`, annotated by hand. Three earlier attempts
to automate the classification were wrong and are recorded at the end, because each
failed in a way worth not repeating.

## The table

| question | the fact the answer needed | where it was lost | class |
|---|---|---|---|
| `edced276` | Hawaii trip = **10 days** | raw says "the 10-day"; the session's one memory says "an island-hopping trip to Hawaii with their family" | **F** |
| `4adc0475` | **2 assists** | raw says "I've had two assists in the league so far"; memory says "The user plays indoor soccer with colleagues" | **F** |
| `37f165cf` | the **416-page** novel | both counts in raw; 440 survived, 416 did not, and the gold is their sum | **F** |
| `73d42213` | **2 hours** travel to the clinic | raw says "it took me two hours to get to the clinic last time"; memory says "looking for a clinic within an hour's drive" — a different number about a different thing | **F** |
| `gpt4_fa19884d` | the user took up **bluegrass** | raw is the user saying so; the memories are the assistant's recommendation lists | **F** |
| `0edc2aef` | the user's preference for **views / a rooftop pool** | the memory is the assistant's Space Needle package, not the user's transferable preference | **F** |
| `80ec1f4f` | the **second February** gallery, with its date | one gold session yielded **zero** memories; the visit survives via another session without its date, so it cannot be counted as February | **E + F** |
| `c9f37c46` | the **open mic night** date | the gold session containing it yielded **zero** memories | **E** |
| `gpt4_7abb270c` | the **Modern Art Museum** visit | five of six museums are in memory and supplied; the sixth is not, and one gold session yielded **zero** memories | **E** |
| `92a0aa75` | total experience **3 y 9 m** | extracted correctly, then **not retrieved** — the other operand was supplied and this one was not | **R** |
| `gpt4_e414231e` | the pedal upgrade **date** | present and supplied; see below | **A** |
| `gpt4_d6585ce8` | **dates** on the concert events | present and supplied; see below | **A** |
| `9a707b81` | the baking class date | present, correct, supplied. The question anchors on the day the cake was baked; the answer anchored on the day the question was asked | **A** |
| `71017277` | the chandelier came **from the aunt** | present, correct, supplied. The answerer did not accept a crystal chandelier as "a piece of jewelry" | **A** |

| class | | n |
|---|---|---:|
| **F** | extracted, but the identifying detail was dropped or replaced | **6** |
| **E** | not extracted at all | **2** (+1 shared with F) |
| **R** | extracted, not retrieved | **1** |
| **A** | supplied and correct; the answer was still wrong | **4** |

**Nine of fourteen never reach the answerer with what they need**, and eight of
those nine are lost at extraction rather than retrieval.

## The dominant sub-mode is a dropped number or date

Seven of the fourteen turn on one missing quantity or date: 10 days, 2 assists,
416 pages, 2 hours, a February date, and two event dates. The memory around it is
correct — the gist survives and the identifier does not. It is the shape the Mayo
case had, at scale:

```
raw     "we had to plan everything out for the 10-day ..."
memory  "The user recently returned from an island-hopping trip to Hawaii
         with their family."
answer  "9 days total: 5 days in New York and 4 days in Hawaii"
```

The 4 was borrowed from a neighbouring memory about spending four days in Paris.
An extractor that drops a number does not produce an abstention downstream; it
produces a confident wrong number.

## The two date failures are not extraction failures

`gpt4_e414231e` and `gpt4_d6585ce8` looked like dropped dates and are not. The
store holds them, resolved correctly:

| memory content | `event_time` |
|---|---|
| "upgraded their road bike pedals … **today**" | **2023-03-19** |
| "attended a music festival in Brooklyn" | **2023-04-01** |

The temporal resolver did its job. And the answerer *was* shown them — with
`temporal` on, `render_memory` appends a window, so the context read:

```
- The user upgraded their road bike pedals to Shimano Ultegra clipless
  pedals today. (since 2023-03-19)
- The user fixed a flat tire on their mountain bike on March 15, 2023.
  (since 2023-03-15)
```

The answerer took the in-prose "March 15" for both and reported zero days between
them. The gold is four.

Two things about that rendering are worth fixing before anything more expensive is
tried, and neither needs a re-ingest:

* **`since` is the wrong word for an event.** It frames a point in time as the
  start of an ongoing state. Both memories above are `type=semantic` carrying a
  one-off event; "on 2023-03-19" says what the field means.
* **The prose and the annotation disagree.** One memory says "today" and is
  annotated 2023-03-19. Leaving an unresolved deictic in the content when the
  resolved value is already in the row invites exactly the reading that happened.

## What this changes

Exact-detail fidelity moves to the top. It is not "objectively real but without
direct evidence" any more: six of fourteen failures are a dropped identifier, and
a seventh is a dropped date compounded by a zero-yield session.

The targeted `batch=1` diagnostic now has named facts to check rather than an
average to compare. The question is no longer "does batch 1 produce more
memories", it is whether it produces **these**:

| question | does batch 1 recover it? |
|---|---|
| `edced276` | "the 10-day" Hawaii trip |
| `4adc0475` | "two assists" |
| `37f165cf` | the 416-page novel |
| `73d42213` | "two hours to get to the clinic" |
| `c9f37c46` | the open mic night, from a session that yielded nothing |
| `gpt4_7abb270c` | the sixth museum, likewise |
| `80ec1f4f` | the February gallery visit, with its date |

Seven questions, and only the gold evidence sessions behind them — a far smaller
run than re-extracting all fourteen.

Two failures need no model call at all. `92a0aa75` is a retrieval miss on a fact
already in the store, and the two date-rendering failures above are a formatting
question.

## Three automated classifications that were wrong

Recorded because each looked reasonable and each produced a confident false answer.

1. **Matching the gold answer's wording against the context.** Worthless for a
   computed answer: the gold "15 days" appears nowhere in a conversation about a
   five-day trip and a ten-day one. It classified every arithmetic question as an
   evidence gap.
2. **Matching quantities as bare strings.** Scored the Hawaii duration as having
   survived, because an unrelated memory said "stay in Europe for 7-10 days". A
   number is not a fact until it is attached to its subject.
3. **`source_session_recalled`, at 94%.** Session-level recall, read as though it
   were fact-level. It is the same mistake the 14.5% zero-yield rate encouraged —
   a coarse statistic that looks healthy while the thing it aggregates is not.

The lesson each time was the same: fourteen is small enough to read.

## Limits

Fourteen questions on a development set, classified by one reader. The class
boundaries are not always sharp — `0edc2aef` is filed as **F** because the memory
stored the assistant's recommendation instead of the user's preference, but it is
equally a failure to *apply* a preference that was in front of the answerer.
`71017277` depends on accepting the benchmark's view that a crystal chandelier is
a piece of jewelry.

This is a benchmark diagnostic. It cannot run in production, where nothing knows
which facts an answer required.
