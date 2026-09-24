---
name: experiment-guard
description: Plan or run reproducible memory experiments while protecting split validity, frozen artifacts, namespaces, and quota. Use for new benchmarks, prompt/extractor/retrieval/answerer comparisons, ablations, regressions, or new reported numbers; do not use for purely local unit tests with no experimental claim.
---

# Experiment guard

Before execution, state the dataset, exact split/manifest, whether it has been inspected,
experiment class (development, ablation, regression, or unseen final), baseline,
candidate, config, models, prompt/extractor versions, and estimated request/token/quota
cost. Prefer zero-call artifact analysis and replay before paid calls.

**State what the question set can detect, before deciding to run.** Run
`tools/resolution.py <rows>` on a set with the target's composition: it reads the
repeat runs already on disk, measures how often an unchanged configuration disagrees
with itself per question type, and reports the smallest net difference that set can
distinguish from noise. If the mechanism's plausible effect is smaller than that
number, the choice is to enrich the set toward the failure it targets, repeat each arm,
or not run it — made before the quota is spent, not after. Two experiments were
registered against a 48-question set whose floor is 4 and returned nets of +0, +0 and
-1; the repeat data that predicted it predated all three.

Never overwrite `results/frozen/`, `results/sealed/`, or archived history. Create a new
config/result namespace, preregister selection and pass/fail rules before reading new
outcomes, and retain failures and negative results. An exposed final split is development
or regression evidence thereafter and must not be used repeatedly for model selection.

Record git SHA, config, models, prompt versions, dataset/split/manifest, sample count,
seed, request/token counts, environment/version information, and the exact output paths.
Use existing protocol and verifier scripts under `scripts/` and `tools/` where they fit;
do not hand-edit their generated artifacts.

Label development-only results explicitly. Do not claim significance without a suitable
test, and do not compare figures from different experimental protocols as if paired.
