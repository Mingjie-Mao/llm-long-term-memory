[![English](https://img.shields.io/badge/English-2962FF?style=for-the-badge)](README.md)
[![中文](https://img.shields.io/badge/%E4%B8%AD%E6%96%87-555555?style=for-the-badge)](README.zh-CN.md)

# ChronoMem

An adaptive long-term memory engine for LLM agents — temporal fact resolution,
memory consolidation, decay-aware hybrid retrieval, and token-budgeted context
assembly, evaluated on [LongMemEval](https://github.com/xiaowu0162/LongMemEval).

> **Status: a negative result localized the representation failure.** On a
> 50-question development subset, v1 ChronoMem scores 26% against naive RAG's 54%.
> Source-session retrieval often succeeds, but the structured representation drops
> answer-bearing details. The next experiment is provenance-backed evidence
> hydration, not ranking tuning. See [the evaluation protocol](docs/EVALUATION.md).

## Why

A vector store plus top-k retrieval answers "what did the user say?" but not
"what is *currently* true?". Given

```
Jan   I use TensorFlow.
Mar   I'm learning PyTorch.
Aug   I've switched completely to PyTorch.
```

naive RAG surfaces all three and lets the model guess. ChronoMem resolves them into
one active fact with a validity window and one superseded fact, then decides which
of them is worth spending context-window tokens on.

## Results

LongMemEval-S · 50-question stratified subset (seed 0) · answerer
`gemini-3.5-flash-lite` · judge `gemma-4-31b-it`, both pinned across every row.
The subset is for development and failure localization, not a final benchmark
claim. p95 is observed answer-provider API latency and excludes free-tier rate-limit
queueing. Every number is regenerated from the JSONL artifacts in `results/raw/`.

| Variant | n | Accuracy | Temporal | Know-update | Abstention | Source-session recall | Ctx tokens | p95 API |
|---|---|---|---|---|---|---|---|---|
| `full_context` | 50 | **56.0%** | 23.1% | 87.5% | 50.0% | — | 109,260 | 7.9s |
| `naive_rag` | 50 | **54.0%** | 46.2% | 75.0% | 100% | 94.0% | 13,057 | 2.1s |
| `chronomem_no_temporal` | 50 | **26.0%** | 7.7% | 37.5% | 100% | 80.0% | 331 | 7.0s |
| `chronomem` | 50 | **26.0%** | 7.7% | 62.5% | 100% | 80.0% | 465 | 1.2s |

**The memory system loses, and the root cause is upstream of temporal resolution.**
Temporal filtering has 3 wins and 3 losses (`p = 1.000`): no detectable difference
at n=50, not evidence that the variants are equivalent.

**40 of 50 source sessions have a selected memory**, but only **12 of those 40**
answers are correct. That is source-session recall, not answer-support recall: the
selected memory can cite the right session while dropping the duration, date, or
number the answer needs. **28 of 50** answers are "I do not know".

One case, traced end to end — *"How long have I been collecting vintage cameras?"*,
gold `three months`. The evidence session produced three memories, ranked first:

```
- The user owns 17 vintage cameras, including a Brownie Hawkeye acquired in May 2023.
- The user owns a rare 1978 pressing of Fleetwood Mac's Rumours.
- The user owns a Mondo poster featuring Hogwarts castle.
```

The duration was never extracted. The next question (`25` postcards) failed the same
way, and the model answered `17` — the nearest number in context.

The v1 literal-coverage gate was a regression detector, not an accuracy ceiling.
It nevertheless exposed the same failure: a lossy structured memory is not enough
when the question depends on local wording or temporal detail. New stores retain raw
turns and memory-to-span anchors; the `two_stage_hydrated` ablation restores only
bounded source evidence after structured retrieval.


Regenerate with `chronomem eval report`; the full table including
single-session-assistant and preference splits is in
[results/table.md](results/table.md).

### Read the paired test, not the accuracy column

The same configuration re-run unchanged scored **48.0%** and **54.0%**. `temperature=0`
does not make a hosted model deterministic, and the judge's borderline calls move
too; four flips out of fifty is eight points. **Any gap smaller than that is not
evidence**, which rules out comparing headline accuracies at this sample size.

Both variants answer the same questions, so the runs are paired and the noise they
share can be thrown away. `chronomem eval compare` runs an exact McNemar test over
the disagreements only:

```
full_context vs naive_rag
  both right   19        naive_rag wins    8
  both wrong   14        naive_rag losses  9
  17 disagreements, p = 1.000  ->  no detectable difference
```

**The supported finding is narrower than "more context buys nothing"**: on this
50-question subset, 8.4x more context produced no detectable accuracy improvement
over naive RAG. The two systems disagree on 17 questions in both directions.

Both systems sit at 54–56%. The temporal split (23.1% vs 46.2%) is a directional
lead for follow-up, not a claim that it survives sampling noise: its denominator is
small and it needs repeated and held-out evaluation.

Two category-level splits survive the noise floor as leads worth pulling on:

- **Abstention, 100% vs 50%.** With sparse retrieved context the model reliably says
  it does not know; handed the whole history it confabulates half the time. Any
  variant that packs in more relevant material risks trading this away, so the
  column stays visible rather than folded into an average.
- **Single-session-user, 100% vs 71.4%.** This is a follow-up lead, not a retrieval
  conclusion: 94% source-session recall does not measure answer-support recall.

These compare ChronoMem's own variants against two baselines. They are **not** a
claim about any third-party system: cross-system memory numbers are only comparable
under an identical judge and prompt, which is not the case across published results.

**Judge reliability is not assumed.** The free tier offers no model stronger than
the answerer to grade with, so the judge is cross-checked against an independent
labelling of all 50 questions: **94% agreement (n=50)** — 1 case where the judge was
more lenient than the label, 2 where it was stricter. No directional bias was
observed in those three disagreements; this does not prove the judge is unbiased.

Two caveats are stated rather than buried. First, the labelling was done by an LLM,
not a person, so it establishes that the rubric is reproducible, not that the judge
is right. Second, the first pass scored 92% and one of those disagreements was the
*labeller's* error: it read a truncated 220-character view of the answer and missed
a conclusion in the last sentence. Labelling now reads full text. The remaining
three disagreements are genuine judgement calls — a misspelt app name (`Memorse`
for `Memrise`), a count that is numerically right but includes a planned purchase,
and an answer naming a specific artist where the gold describes an unnamed band.

**Every number here is regenerated from the JSONL artifacts**, not from console
output. `run_eval` refuses to return a result when the two disagree: an earlier run
printed `50 questions, 56.0%` over a file holding 30, and both numbers looked
reasonable ([D25](docs/DECISIONS.md)).

Detailed metric definitions, failure taxonomy, repeat-run protocol, and latency
interpretation are in [docs/EVALUATION.md](docs/EVALUATION.md).

## Extraction

The write path turns sessions into typed, triple-formed memories with validity
windows. Two gates protect it, both of which have already caught real defects:

**Answer coverage** (`chronomem ingest coverage`) extracts from the evidence-only
split and checks whether the gold answer survives into memory. It found the first
extraction prompt generalising away exactly the specifics the benchmark asks about
— `The Glass Menagerie` becoming "interested in acting" — and rewriting the prompt
took measurable coverage from **26.3% to 50.0%**, with `single-session-user` going
1/3 → 3/3. The metric is a strict lower bound and is documented as one: most
LongMemEval answers are *computed* from stored facts rather than stated in them.

**Dedup arity.** Embedding similarity is only a recall filter; an LLM makes the
DUPLICATE / UPDATE / DISTINCT call, because "likes Python" and "does not like
Python" sit at ~0.95 cosine and no threshold separates them. Restricting
`(subject, predicate)` collisions to single-valued predicates cut adjudication
calls **72 → 3** per 60 sessions while *increasing* duplicates caught.

## Temporal resolution

Every fact carries a validity window. For each `(subject, predicate)` key the
resolver sorts by event time and rewrites the whole chain, so `TensorFlow` ends up
closed at the date `PyTorch` began and only one value is left in force.

Rebuilding the timeline rather than comparing pairs is the load-bearing choice.
Sessions are ingested in arbitrary order, so "the memory that just arrived
supersedes the one already there" would let a late-arriving *January* fact become
current — and once a memory is superseded it is no longer the head, so a fact
landing between two existing ones could never rewire the link that now points past
it. Rebuilding is idempotent, order-independent, and costs no LLM calls.

Two refinements that came out of writing the tests:

- **Restatements are not moves.** "I live in Canberra" in March and again in June
  collapses to one interval owned by *March*, so "when did you move?" answers
  correctly. Counted as `restatements`, not `superseded`.
- **Arity is the default rule, not the whole rule.** `uses_tool` is multi-valued —
  using PyTorch does not stop you using NumPy — so a static predicate list leaves
  the headline example unresolved. Extraction therefore also emits
  `replaces_previous` when the user says "I switched to X", at no extra request
  cost, and either signal is enough to resolve a key.

The first real ingest immediately proved the list wrong. `lives_in` produced
`Tokyo → South Bay → Las Vegas`, which is right; `scheduled` produced
`layover in London → cooking class → the 9:15 train → Friday game nights`, which is
not a sequence of competing values at all. `scheduled` and `has_goal` were removed.

That mistake cost seconds rather than a day, and deliberately so: resolution reads
stored data and calls no model, so `chronomem resolve` rebuilds every timeline
in-place. Arity is a property of the predicate, not of the data, so keeping it out
of the ingested artifact is what makes it cheap to be wrong about.

## Quickstart

```bash
uv sync --group dev
uv run chronomem data download --variant s
uv run chronomem data stats --variant s
uv run chronomem data plan --variant s

uv run chronomem doctor
uv run chronomem ingest coverage --n 20      # is extraction keeping the answers?
uv run chronomem ingest run                  # build the store (resumable)
uv run chronomem eval run naive_rag
uv run chronomem eval report
```

`data plan` reports how many days a full ingestion takes at each batch size given
your API quota — on a request-capped free tier that is the schedule, not the cost.

## Design

Written up in [docs/DECISIONS.md](docs/DECISIONS.md) — what was chosen, what was
rejected, and what the measurements said.

| Layer | Choice |
|---|---|
| Store | SQLite (WAL) + FTS5 for BM25 — one file, no service dependency |
| Vectors | Exact flat inner-product search over numpy |
| Embeddings | `all-MiniLM-L6-v2`, local, MPS-accelerated |
| LLM | Gemini via `google-genai`, behind a three-role config (extractor / answerer / judge) |
| Benchmark | LongMemEval-S via a checkpointed, quota-aware ingestion pipeline |

## Layout

```
src/chronomem/
  store/       schema.sql, SQLiteMemoryStore, NumpyFlatIndex
  llm/         quota-aware rate limiter (RPM / TPM / RPD), retrying client
  ingest/      extraction, dedup, checkpointed pipeline, coverage gate
  evaluation/  benchmark loaders, runners, judge, reporting
  config.py    one YAML per ablation variant
  cli.py
tests/
docs/DECISIONS.md
```

## License

MIT
