# Raw-primary v4 against v1 — dev100, three runs per arm — pre-registration

Date: 2026-10-01 (Australia/Sydney). Written before any answer or judge call of this
experiment. Supersedes `results/prereg-raw-primary-dev100-v3.md`, withdrawn unrun.

## Classification

**A development comparison, recorded as regression evidence.** `dev100` was used once,
in aggregate, for a v2 arm decision; no per-question failure of it has been read. Its
legacy manifest lets the CLI record runs only as `--experiment-class regression`.
Nothing here is an unseen result or an accuracy figure.

## What v4 is

v1 (`two_stage_raw_primary`, passed on train150, net +29) plus five changes, each taken
from how comparable systems handle a failure this project measured:

| change | from | evidence before this run (zero calls) |
|---|---|---|
| BM25 + dense turn fusion | LongMemEval, Emergence: dense turn matching | 4k all-gold 78.1% -> 83.6% train150, 86.0% -> 88.2% heldout100 |
| turns keyed by text + anchored facts (lexical and dense) | LongMemEval: fact-augmented key expansion | 77.4% -> 81.5% train150 (7-1), 84.9% -> 89.2% heldout100 (4-0) |
| turns the retrieved memories point at, as a third ranking | Zep/Mastra: a layer that holds as history grows | at 4x history 74.7% -> 80.8%; at 1x 83.6% -> 85.6% |
| conversation dates stated relative to the question | Mastra: relative dates computed ahead of the reader | none — answering only |
| evidence listed before deciding (`v2_notes`, 1024 output tokens) | LongMemEval: Chain-of-Note reading | none — answering only |

Anchors in the store are the stored ones; new ingests anchor with `provenance-v2`
(`results/prereg-anchor-v2-offline-v1.md`), which on heldout100 made no difference to
fact-keyed reach (90.3% stored vs 89.2% recomputed).

Not included, with the reason: a pinned preference block (Letta, Mastra) — on all 21
preference questions across three sets, no preference or profile memory is anchored to a
gold turn, so a pinned block would carry none of the evidence; the remedy is at
extraction and needs its own paid probe. A stronger answerer — registered separately as
a diagnostic (`results/prereg-strong-answerer-probe-v1.md`).

**Five changes in one arm, on purpose.** Each is worth a few questions per hundred at
most, below what dev100 resolves for any one of them. A pass says the bundle helps and
cannot say which part did; a failure cannot say which part failed.

## Data and system

Store `stores/dev100.db` (12,436 memories, 49,590 turns), fact-keyed turn index
`stores/dev100-turn-key-index` (`lltm lifecycle build-turn-index --fact-keys`, 49,590
rows), config `configs/fallback.yaml`, manifest `results/manifests/dev100.json`, answerer
`gemini-3.5-flash-lite`, judge `gemma-4-31b-it`; answer prompts `memory-aware-v2` (v1) and
`memory-aware-v2-notes` (v4).

## Pre-run estimate of context (zero calls, `tools/context_precheck.py`)

| arm | median | p90 | max |
|---|---:|---:|---:|
| v1 | 5,689 | 5,806 | 5,873 |
| v4 | 5,811 | 5,922 | 6,105 |

## Resolution and runs

As in v3: dev100 resolves a net of 7 with one run per arm (`results/analysis/resolution-dev100.md`,
conservative; pairwise gives 6). Each arm is run **three times**; arms are compared on
each question's mean correctness, which lowers the floor to **4**.

## Decision rules

v4 passes if **all** hold:

1. mean net — Σ over questions of (v4's mean correctness − v1's) — **≥ +4.0**;
2. no question type's mean net below **−2.0**;
3. median total answer context of v4 over its three runs **≤ 6,000** tokens.

Reported, not deciding: per-run scores, per-type mean nets, median output tokens, fallback
second calls per run, an exact sign test over questions whose mean moved (no significance
claimed). Gate: `tools/raw_primary_gate_v4.py`.

**A pass** makes v4 the candidate for a comparison on a new, unseen set. **A failure**
leaves v1 the candidate. Nothing further is tuned on dev100 against these runs.

## Cost

About 340 answer calls per arm over three runs, 680 in all, and 600 judge calls: two
answerer quota days before provider failures. Runs resume from their result files; if all
six are not complete by **2026-10-08**, the experiment is recorded as incomplete.

## Reproduction

```bash
lltm lifecycle build-turn-index --config configs/fallback.yaml --store-name dev100 --fact-keys
for rep in 1 2 3; do
  lltm eval run two_stage_raw_primary    --config configs/fallback.yaml --store-name dev100 \
    --questions results/manifests/dev100.json --label dev100-v4-rep$rep
  lltm eval run two_stage_raw_primary_v4 --config configs/fallback.yaml --store-name dev100 \
    --questions results/manifests/dev100.json --label dev100-v4-rep$rep
done
python tools/raw_primary_gate_v4.py
```

## Withdrawn before any call — 2026-10-05

No run of this registration was started. Since it was written, a separate line of work
registered `paged-paired-dev100-v18-v1` (`results/prereg-paged-paired-dev100-v18-v1.md`) on
the same `dev100` and has 356 of its 600 answers saved. Running v4 on `dev100` as well would
spend the set twice and make two families of comparisons on it. The project keeps one main
line: v18 completes `dev100`; v4's changes are carried forward as a follow-up to be
registered on another set once v18's result is known. The v4 implementation (fused and
fact-keyed turn retrieval, memory-led turns, dated headers, evidence-first answering) stays
in the code, off by default.
