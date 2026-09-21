# v5.0 / v5.1 offline gate (zero model calls)

> Rows: `results/raw/two_stage_v2c.reasoning48-v2c8.jsonl` · store: `stores/v2c-reasoning48.db`. Derived from the
> committed v2c run; no question is re-answered and no gold is read beyond the
> string matching already used by the extraction-coverage diagnostic.

- questions: **48**, correct **37**
- questions where no raw turn ever reached the reader: **38/48**
- wrong answers: **11** (4 string-matchable, 7 not: abstention, multi-session or preference golds)
- how the matchable ones match against the whole store: {'none': 2, 'numeric': 2} — `exact` is a real phrase hit, `numeric` only means the digits occur somewhere, `tokens` only that 80% of the words do

## Attribution of every wrong answer (hand audit)

At eleven failures the automated ladder below is supporting evidence, not the
instrument. Reading the rows overturned it twice, so each failure was checked
against the store and its raw turns by hand.

| cause | n | what it means |
|---|---:|---|
| `reader` | 6 | the evidence was in the selected context and the answer still failed |
| `local_raw` | 3 | a sentence the extractor dropped, in the raw turns of an already-selected session — **v5.0** |
| `repack` | 1 | the fact was in the store and not selected — **v5.1** |
| `retrieval` | 1 | the gold session was never found |

| question | cause | note |
|---|---|---|
| `0ddfec37_abs` | `reader` | answered with a count instead of declining a false premise |
| `58ef2f1c` | `local_raw` | structured kept 'February 2023'; the raw turn says the dinner was "back on Valentine's Day", which the gold states as February 14th. The evidence is a raw turn of a selected session, but reaching the gold from it also needs the reader to know the date of Valentine's Day |
| `a4996e51` | `local_raw` | 'up to 50 hours per week' is an assistant turn; the surviving memory says 40-45 and was not selected |
| `a9f6b44c` | `reader` | listed the qualifying events and never produced the count |
| `d905b33f` | `reader` | $24 and the original price were both in context; read it as 30 -> 30 |
| `dd2973ad` | `reader` | the 2 AM memory was selected; the answer still said it did not know |
| `e48988bc` | `retrieval` | the gold session was never found |
| `gpt4_2f8be40d` | `reader` | counted four weddings where the gold names three |
| `gpt4_70e84552_abs` | `reader` | answered instead of declining a false premise |
| `gpt4_74aed68e` | `repack` | the dated 'February 14' memory was in the store and unselected; the undated restatement carried the session date instead |
| `gpt4_7a0daae1` | `local_raw` | 'I just received my new tennis racket today' is a raw turn of a session that was selected; only assistant recommendations were extracted from it |

## Where a string-matchable gold is, by exact match only

Supporting only. Seven of the eleven golds cannot be located this way: two are
abstention sentences whose words come from the conversation, two are derived
values that no turn ever spells out, and the rest are multi-session or
preference answers the coverage diagnostic already treats as unmeasurable.

| location | n | reachable by | questions |
|---|---:|---|---|
| already in the selected memories | 0 | neither mechanism — the reader had it | — |
| in the store, not selected | 0 | v5.1 repacking | — |
| only in raw turns of located sessions | 0 | v5.0 source-local window | — |
| only in raw turns elsewhere | 2 | archive-wide retrieval | `e48988bc`, `gpt4_7a0daae1` |
| nowhere | 2 | neither — extraction or gold matching | `58ef2f1c`, `gpt4_74aed68e` |

## How much of a located conversation the reader sees

- sessions represented in a context, summed over questions: **528**
- memories from them shown: **958** of **2949** stored
- mean fraction of a located session shown: **44.7%**
- sessions shown at most a quarter of: **200**

## Decision

**v5.0 is worth running: 3 of 11 failures are a sentence the extractor dropped from a session retrieval already found.** All three have the same shape, which is the shape the mechanism addresses, and in 38 of 48 questions no raw turn reached the reader at all.

**v5.1 repacking alone reaches 1**, and that one is compounded by a cheaper defect: the dated memory was unselected while an undated restatement carried the session date as its event time. Splitting event time from observation time may fix it without touching the context shape, so v5.1 is not registered on this evidence.

**7 of 11 are out of reach of either mechanism** — six where the evidence was in front of the reader, one genuine retrieval miss. That is the dominant bucket and it is a reader question, not a context-shape question.
