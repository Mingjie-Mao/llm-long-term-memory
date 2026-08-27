# Five subsystems that are built, wired and switched off

**Inventory taken 2026-08-25.** None of this affects any published number — every item
below is off in the shipped configuration and off in `configs/v2.yaml`. It is recorded
because "there is code here that nothing runs" is worth knowing before someone reads the
repository and assumes otherwise.

| Subsystem | Code | Switch | Why it is off |
|---|---:|---|---|
| Utility predictor for budget packing | `influence/` — 697 lines | `pack.enabled: false` | **Measured and rejected.** Held-out RMSE 0.310 against a mean-baseline 0.263 — it lost to predicting the mean (`results/p6-pilot.md`) |
| Budget packer | `pack/` — 130 lines | `pack.enabled: false` | Depends on the predictor above |
| Cross-encoder reranking | `retrieve/rerank.py` | `retrieval.rerank.enabled: false` | **Measured and rejected.** At k=10 it changed zero answers and lowered source recall at every k (`results/rerank-pareto.md`) |
| Always-on evidence hydration | `retrieve/hydrate.py` | not selected by any shipped variant | **Measured and demoted.** +3.2pp (2W-1L, p = 1.000) at 3x the context; replaced by conditional fallback |
| Memory consolidation | `consolidate/` — 174 lines, plus the `evidence` table | `consolidation: false` | **Never measured.** The only item here with no experiment behind it |
| Decay / reinforcement | `strength`, `access_count`, `last_accessed_at`, `strength_updated_at` in the schema; `use_strength` in the retriever | `decay.enabled: false`, `use_strength=False` | **Never measured.** Four schema columns carried but not exercised |

Roughly 1,200 lines of Python and four schema columns.

## The distinction that matters

Four of the six are **measured rejections**. Keeping them, off by default and behind their
own extras, is deliberate: the result is reproducible by anyone who flips the switch, and a
negative result that cannot be re-run is only an assertion. The reranker in particular sits
behind an optional `rerank` extra so that running the product does not pull torch and a
model nobody asked for — the dependency is made deliberate rather than incidental.

Two are **not measured at all**: consolidation and decay. They are dead weight by a
different definition — nothing establishes that they should be off, only that nobody turned
them on. Decay also costs four columns in a schema that every store carries.

## What to do about it, and when

Nothing before `test100`. Deleting or enabling any of it changes `src/`, which is inside
the freeze's hashed source inventory, and would break the dev100 lineage exactly as an
earlier one-line change already did.

After the experimental conclusion is fixed, three honest options per item:

1. **Measure it** — the only route that turns consolidation or decay into a real decision.
2. **Delete it** — with the reason recorded in `DECISIONS.md`, which is the current fate
   this inventory recommends for decay: four columns in every store for a mechanism nobody
   has justified.
3. **Keep it, off, with its measurement cited** — the correct outcome for the four rejected
   items, and what already happens.

The one thing not to do is leave the two unmeasured subsystems in the state they are in,
where "off" reads as a decision and is in fact an absence of one.
