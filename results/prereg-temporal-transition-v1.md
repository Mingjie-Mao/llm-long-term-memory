# Temporal transition replay v1 — preregistration

## Classification

Regression analysis over an already inspected development store. This is not an unseen
benchmark and produces no QA or statistical-significance claim.

## Inputs and cost

- store: `stores/v2b-gate16-repair.db`
- manifest lineage: `results/manifests/v2b-gate16.json`
- baseline: states already persisted by the previous resolver
- candidate: explicit REPLACE/REMOVES handling with `target_object`
- models/prompts: no model is called; existing extracted rows are replayed
- request, token and quota cost: zero

The source store is copied to a temporary database. Frozen, sealed and archived artifacts
are not opened for writing.

## Questions and gates

1. A removal/termination row must not remain `active` after replay.
2. The known Nike termination must not supersede the current Adidas value.
3. Ambiguous transitions must not close a value by row order.
4. Replaying the candidate twice must produce identical memory states.
5. Every changed state and ambiguity is retained in the generated report.

This gate measures lifecycle consistency only. It cannot establish extraction recall,
retrieval recall or end-to-end QA gain.
