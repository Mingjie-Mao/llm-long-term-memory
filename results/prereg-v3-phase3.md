# Pre-registration — v3 phase 3 reasoning validation

**Written after the aggregate-only pilot passed and before inspecting or running the
new `v3-reasoning-dev60` questions.**

## Data boundary

- `v3-reasoning48` is completed development evidence and will not be rerun.
- `v3-reasoning-tune42` may be inspected and reused while improving time and
  multi-session reasoning.
- `v3-reasoning-dev60` is a sealed development check. Its question text, answers,
  model rows and judge reasons must not be inspected before the candidate is frozen.
- All three manifests are disjoint and together partition `train150`.
- No individual `test100` content or rows may be read or used. Step 5 requires a
  genuinely new final set; `dev60` is not represented as that final set.

## Workflow

1. Add deterministic local tests using only synthetic cases and `tune42`.
2. If provider calls on `tune42` are needed, obtain separate user permission first.
3. Freeze one candidate answer policy before any `dev60` provider call.
4. Compare the unchanged v2 answer policy with that candidate on the same 60
   questions, three repeats per arm. Publish aggregate-only results.
5. Do not change the candidate or rerun `dev60` after seeing its aggregate.

## Registered dev60 gate

| check | requirement |
|---|---|
| combined time + multi-session | candidate gains at least 5 percentage points in majority accuracy |
| time questions | candidate majority-correct count is not lower than v2 |
| multi-session questions | candidate majority-correct count is not lower than v2 |
| knowledge updates | candidate majority-correct count is not lower than v2 |
| ordinary controls | candidate majority-correct count is not lower than v2 |
| confident mistakes | no increase in high-confidence majority-wrong answers |
| retrieval/context | same store/retrieval/fallback and median selected context no larger than v2 |
| generation cost | answer output tokens no more than 1.5x v2 |

All checks must pass. Failure records a negative result and sends the work back to a
new tune/development cycle; it does not permit inspection of test100 or a second dev60 run.
