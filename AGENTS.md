# Repository rules for Codex

These rules apply to all work in this repository. More specific instructions may add
constraints, but must not weaken the experiment and tenant-isolation boundaries below.

## Project boundary

- This repository implements long-term memory for LLMs and agents.
- Do not add working memory or current-session short-term memory. The caller owns the
  active conversation context.
- Enterprise RAG and Agentic RAG belong to another project. Do not merge those scopes
  here unless the user explicitly requests a scope change.
- Prefer the smallest change that advances the stated long-term-memory goal.

## Evidence before architecture

- Claims of improvement require reproducible measurements.
- Label evidence accurately as historical, development, regression, or unseen-final.
  Never present development results as final results.
- Preserve failures and negative results. Prefer existing measurements to architectural
  intuition.
- Diagnose the first failing layer before changing the system:

  `raw conversation -> extraction -> temporal keying -> deduplication -> temporal lifecycle -> storage -> retrieval -> raw fallback -> answer synthesis`

- If the required evidence reached the answer context, do not blame retrieval. If the
  fact was never extracted, do not blame retrieval either.
- Do not add rerankers, Graph RAG, query expansion, a new vector database, or a complex
  retrieval pipeline without evidence that retrieval is the bottleneck.
- Keep SQLite + NumPy unless a measured product requirement justifies migration, such as
  multi-process writes, multi-instance serving, high QPS, or millions of memories.

## Frozen and sealed experiments

- Treat v2 frozen/sealed artifacts and all other `results/frozen/`, `results/sealed/`,
  and archived evidence as immutable history. Do not overwrite, regenerate, edit, or
  delete them.
- Do not change benchmark numbers to make documents appear consistent.
- Give every new experiment a new config/result namespace. Keep failed experiments.
- Once a test split has been inspected or used, it is development/regression evidence,
  not unseen final evidence. Never reuse it for model selection while calling it unseen.
- Before paid calls, check whether existing artifacts or an offline replay can answer the
  question. Record dataset, split exposure, experiment class, baseline, candidate,
  config, models, prompt/extractor versions, sample count, seed, request/token usage,
  environment, and git SHA where applicable.
- Do not imply statistical significance without an appropriate test.

## Temporal correctness

- Keep conversation/session time, fact/event time, `valid_from`, `valid_to`, current
  state, historical state, and `as_of(t)` distinct.
- Keep ADD, REPLACE, TERMINATE, and COEXIST conceptually distinct. TERMINATE must not be
  represented as a fabricated replacement value.
- Out-of-order ingestion must converge to the same timeline as chronological ingestion.
- Preserve raw relative-date expressions and their provenance when resolving them.

## Tenant isolation

- User/tenant isolation is a correctness boundary, not an optional security enhancement.
- Any change to deduplication, retrieval, vector indexes, raw fallback, API auth, storage,
  export, or deletion must test cross-user isolation.
- Never deduplicate or retrieve across users.

## Code-change workflow

Before editing code:

1. Identify the requirement or first failing layer.
2. Read the relevant existing tests.
3. Read the relevant experiment evidence.
4. Make the smallest change that addresses the failure.

After editing code:

1. Run subsystem-targeted tests first.
2. Expand testing in proportion to the blast radius.
3. Run lint and formatting checks.
4. Report exactly which checks ran and their outcomes.
5. Never say "all tests pass" unless the full suite actually ran. Distinguish passed,
   failed, skipped, and environment-blocked tests.

## Results and documentation

- Do not casually modify `results/`. Use its generators when artifacts are generated,
  and keep source code, evaluation tooling, and generated evidence separate.
- Before changing a benchmark or release number, trace it to an authoritative artifact
  and check `README.md`, `docs/PROJECT_REPORT.md`,
  `docs/EVALUATION*`, configs, `public-demo/site/release.json`, and the related
  frozen/sealed artifacts.
- Accuracy, context, p-values, test counts, and coverage figures must refer to the same
  experiment and remain traceable to their source.

## Default search scope

- For ordinary implementation and bug-fix work, search `src/` and the corresponding
  files under `tests/` first. Read `configs/` and `docs/ARCHITECTURE.md` only when the
  task needs them.
- Read `results/` by default only for benchmark analysis, failure analysis, experiment
  reproduction, release audits, or when the user explicitly requests historical results.
- Do not scan all of `results/` for an ordinary code change. Locate large artifacts by
  question ID, experiment ID, or a specific key and read only the relevant slice.
- Prefer a summary or report before raw JSON/JSONL. Never put an entire historical
  experiment directory into model context.
- Treat `results/frozen/` and `results/sealed/` as read-only. Limit searches to the
  subsystem being changed rather than searching the whole repository by default.

## Repository skills

Use the focused skills in `.agents/skills/` when their descriptions match. In
particular, use `experiment-guard` before experiments and
`code-change-verification` after behavior changes. Use diagnostic skills to locate the
first loss before proposing a new subsystem.
