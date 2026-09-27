# Memory as an index — offline gate v1 — decision

Date: 2026-09-28 (Australia/Sydney). Registered in
`results/prereg-memory-as-index-offline-v1.md`. **DEVELOPMENT, zero calls**, `train150`.
Artifacts: `results/analysis/memory-as-index-offline-v1.{json,md}`.

## Verdict: STOP — memories are not a reliable index into the conversation

Using extracted memories only to locate source turns reaches the gold evidence far less
often than searching the raw archive with the question itself, at every budget:

| all-gold coverage | 1,000 | 2,000 | 4,000 | 8,000 |
|---|---:|---:|---:|---:|
| `mem_turn` (memory → its source turn) | 41.8% | 45.2% | 45.9% | 47.3% |
| `mem_window` (… ± 1 turn) | 19.9% | 38.4% | **50.0%** | 56.8% |
| `mem_session_bm25` (memory → session, question → turns) | 56.2% | 66.4% | 70.5% | 77.4% |
| `bm25_turns` (question → any turn) | 62.3% | 71.9% | **78.1%** | 85.6% |
| `mem_session_whole` | 5.5% | 8.9% | 25.3% | 63.0% |

Both registered conditions fail at 4,000 tokens: the best memory arm covers 50.0%
(needs 80%), and it loses to `bm25_turns` 9 wins to 50 losses. No answer experiment is
paid for on this design.

## First failing layer: extraction, not retrieval

Of 146 questions with flagged gold turns, a top-20 memory is anchored to a gold turn in
107 (73.3%). Top 50 adds none. For the 39 others:

| cause | questions |
|---|---:|
| no memory in the store is anchored to any gold turn | **38** |
| anchored memory exists but is superseded | 1 |
| anchored, active, and ranked out of the top 50 | **0** |

Retrieval never lost a gold-anchored memory. What it cannot rank is what extraction did
not write — or wrote under the wrong turn. Only **50.0%** of questions have a memory
anchored to *every* gold turn, which caps any memory-as-index design on this store.

The losses concentrate where v2 lost to full history on `test100`:
single-session-user 10 of 20, single-session-assistant 8 of 17, single-session-preference
6 of 9 have no memory from the gold turn.

Four missed cases read by hand (train150 permits it):

- three are extraction loss — one summarising memory per session, and the detail asked
  for ("38 subjects", "28. Kg3", the playlist "Summer Vibes") was never written;
- one is **anchor error**: "completed an undergraduate degree in Computer Science from
  UCLA" was extracted but anchored to turn 1, while the user said it in turn 4. v2 could
  answer it; an index would point at the wrong turn.

The store was built at batch 15. Batch 8 plus the repair raises specific recall
(47.5% → 60.5% on `v2b-gate16`) and would raise the index hit rate, but not to where the
raw archive already is.

## What this changes

The proposal was "memories as the index, source turns as the answer context". The data
says the memory layer is a good **state** layer and a poor **locator**: it knows which
facts are current, and it misses half the turns a question needs. The raw archive,
searched with the question, reaches 78% of all-gold evidence in 4,000 tokens — seven
times v2's median context, still 27 times less than full history.

That points to a different design, not tested here: **raw-turn retrieval as the primary
evidence, extracted memories alongside it for temporal state** — the "artifacts
alongside chunks" configuration that preserved accuracy in the external ablation. It
needs its own registration. Two cautions carried into it:

- LongMemEval questions often reuse the gold turn's wording, which flatters BM25
  (`results/raw-recall-diagnostic.md`: natural-question R@5 74%, keyword 97%). The
  retention paraphrase check found source sessions still recovered on 16 items, but did
  not score turns or answers.
- v5.0 already added question-driven raw turns from memory-located sessions
  (`mem_session_bm25` here, at a far smaller budget) and returned net 0 on
  reasoning-48, below that set's resolution. Reach is not use.

## Reproduction

```bash
python tools/index_offline_gate.py \
  --json-out results/analysis/memory-as-index-offline-v1.json \
  --md-out results/analysis/memory-as-index-offline-v1.md
```

Local `all-MiniLM-L6-v2` encoder, `configs/v2.yaml` retrieval, `stores/train150.db`,
2 min 43 s on the development machine. Gold-turn text matched the store for every
flagged turn (0 mismatches).
