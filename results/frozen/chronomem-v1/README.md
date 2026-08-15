# llm-long-term-memory v1 — frozen

Do not overwrite, regenerate, or "fix" anything in this directory. It is the record
of the first end-to-end measurement of the memory system, including the extraction
prompt that produced it (`extraction_prompt.txt`, sha256 `e6e7a4e4…`).

## What it shows

| variant | accuracy | temporal | know-update | evid. recall | ctx tokens |
|---|---|---|---|---|---|
| `full_context` | 56.0% | 23.1% | 87.5% | — | 109,260 |
| `naive_rag` | 54.0% | 46.2% | 75.0% | 94.0% | 13,057 |
| `chronomem_no_temporal` | 26.0% | 7.7% | 37.5% | 80.0% | 331 |
| `chronomem` | 26.0% | 7.7% | 62.5% | 80.0% | 465 |

**Aggressive memory compression cut context by roughly 28x and cost half the
accuracy.** Extraction reduced ~13,000 tokens of retrieved session text to ~350
tokens of facts, and the details the questions ask about — durations, counts,
relative dates — did not survive the reduction.

The failure is localised. Evidence memories were recalled for 40 of 50 questions;
only 12 of those 40 were answered correctly, and 28 of 50 answers were "I do not
know". Retrieval found the right material. The material no longer contained the
answer.

## Why it is kept

This is the measurement the next phase is designed against. A memory system that
compresses an order of magnitude and holds accuracy is a result; one that
compresses an order of magnitude and halves accuracy is a different result, and
which of the two happened is only knowable because this was run and recorded rather
than tuned until it looked better.

Deleting it would also delete the reason for everything that follows. The v2
extraction work exists because of these numbers, and a reader who cannot see them
has to take the diagnosis on trust.

## Provenance

- LongMemEval-S, 50-question stratified subset, seed 0
- answerer `gemini-3.5-flash-lite`, judge `gemma-4-31b-it`, both pinned
- store: 2,400 sessions → 2,007 memories, 15 sessions per extraction request
- judge/human agreement on this configuration: 94% (n=50)

The 50 questions used here are **dev**, not test: batch size, the predicate arity
list, the v2 extraction prompt, and the decision to build P4 before P3 were all
chosen by looking at them. See `docs/DECISIONS.md` D27.
