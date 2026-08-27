# v2 execution runbook

This is the exact order of operations.  Commands without `--run` are preflights and
make no API calls.  Quota pauses always resume the same command.  Never add
`--fresh`, never delete a one-shot ledger, and do not commit or push.

| step | action | completion evidence |
|---:|---|---|
| ~~1~~ | ~~finish train150~~ | **done 2026-08-25** — 7,180/7,180, 18,519 memories |
| ~~2–3~~ | ~~final zero audit + fixed train grid + write v2 config~~ | **done 2026-08-25** — audit, selection record and `configs/v2.yaml` (`mean` / radius 1 / cap 30) |
| ~~5a~~ | ~~freeze dev ingestion inputs~~ | **done 2026-08-26** — `v2-candidate-preingest`, verifies PASS. A first attempt on 2026-08-25 was discarded with its 37%-complete store; see `prereg-context-shape.md` amendment 2 |
| 5b | ingest dev100 | **in progress — 2,183/4,791 (45.6%)**; resume the same command each quota day |
| 5c | run aggregate-only dev experiment | five arms x three repeats and `dev100-aggregate.md` |
| 6a | choose v2 by registered dev rule | decision recorded without individual dev rows |
| 6b | freeze and ingest test100 | pre- and post-ingest hashes |
| 6c | spend test100 once | durable ledger + one aggregate report for v2 and baselines |
| 7 | product hardening | only after experimental conclusion is fixed; follow `docs/PRODUCTIZATION_V2_PLAN.md` |

## 1. Resume train150

Before and after every provider run, use the read-only state checker:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/check_ingest_state.py
```

It must print `SAFE TO RESUME` while work remains, or `COMPLETE` at 7,180/7,180.
It verifies the checkpoint, raw archives, SQLite integrity, memory/index/vector
counts and extractor fingerprint together. One raw-only pending batch is safe: it
is the interrupted batch that the next run deliberately replays.

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m llm_long_term_memory.cli ingest run \
  --config configs/fallback.yaml \
  --store-name train150 \
  --questions results/manifests/train150.json
```

Exit 2 means a normal quota pause with unfinished work.  Run the identical command
after the next Pacific-midnight reset.

## 2–3. Finalize train150 without API calls

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/finalize_train150.py
```

It refuses anything below 7,180/7,180.  On success it writes:

- `results/analysis/train150-ingest-state.final.json`, proving every session is
  successful, explicitly empty, or content-blocked and that all store artifacts agree;
- `results/analysis/train150-zero-yield.final.json`;
- `results/analysis/train150-zero-yield.final.md`, including the validated randomized
  batch-position evidence and an explicit unresolved bucket;
- the seven fixed context-grid artifacts;
- `results/analysis/train150-context-selection.final.json`;
- `configs/v2.yaml`;
- a second selected-config verification artifact.

If Top-3, assembled recall, or context gates fail, it records `STOP` and writes no
candidate config.

## 5. dev100 validation

The frozen variant order is the pre-registration's flat baseline, coherent
candidate, oracle ceiling, reported retrieval baseline, then memory-only
ablation.  Keep the same five variants, in this order, for every dev100 freeze
and ingest resume.

```bash
.venv/bin/python scripts/freeze_v2.py --capture \
  --name v2-candidate-preingest \
  --config configs/v2.yaml \
  --manifest results/manifests/dev100.json \
  --pre-ingest-store-name dev100 \
  --variant two_stage_fallback \
  --variant two_stage_coherent \
  --variant two_stage_coherent_oracle \
  --variant naive_rag \
  --variant two_stage_memory_only
```

```bash
.venv/bin/python scripts/run_frozen_ingest.py --run \
  --freeze-name v2-candidate-preingest \
  --config configs/v2.yaml \
  --manifest results/manifests/dev100.json \
  --store-name dev100 \
  --variant two_stage_fallback \
  --variant two_stage_coherent \
  --variant two_stage_coherent_oracle \
  --variant naive_rag \
  --variant two_stage_memory_only
```

Repeat the second command after quota resets until complete, then freeze the actual
store:

```bash
.venv/bin/python scripts/freeze_v2.py --capture \
  --name v2-candidate \
  --config configs/v2.yaml \
  --manifest results/manifests/dev100.json \
  --store-name dev100 \
  --variant two_stage_fallback \
  --variant two_stage_coherent \
  --variant two_stage_coherent_oracle \
  --variant naive_rag \
  --variant two_stage_memory_only
```

The post-ingest freeze is accepted only if it extends the original pre-ingest
freeze: code, configuration, data and pre-registered rules must still have the
same hashes. The complete train150 zero-yield, seven-arm grid, selected-config and
verification artifacts are part of that lock. Only the completed store is allowed
to be newly added.

Preflight once without `--run`, then start/resume validation:

```bash
.venv/bin/python scripts/run_validation.py --run \
  --freeze-name v2-candidate \
  --config configs/v2.yaml \
  --manifest results/manifests/dev100.json \
  --store-name dev100 \
  --arm flat20=two_stage_fallback \
  --arm coherent-auto=two_stage_coherent \
  --arm coherent-oracle=two_stage_coherent_oracle \
  --arm naive_rag=naive_rag \
  --arm memory-only=two_stage_memory_only \
  --runs 3
```

Only aggregate counts are printed while running.  Scores appear after all 15
arm/repeat combinations finish.

## 6. test100, once

First record which product arm survived dev100.  Final arms are registered in
`prereg-v2-final.md`: `v2`, `full_context`, `naive_rag`, and—only when it is not
identical to v2—`flat_memory_fallback`.

The commands below show the coherent-v2 case.  If dev selects flat fallback, change
the first variant to `two_stage_fallback` and omit the final duplicate variant/arm.

```bash
.venv/bin/python scripts/freeze_v2.py --capture \
  --name v2-final-preingest \
  --config configs/v2.yaml \
  --manifest results/manifests/test100.json \
  --pre-ingest-store-name test100 \
  --variant two_stage_coherent \
  --variant full_context \
  --variant naive_rag \
  --variant two_stage_fallback
```

```bash
.venv/bin/python scripts/run_frozen_ingest.py --run \
  --freeze-name v2-final-preingest \
  --config configs/v2.yaml \
  --manifest results/manifests/test100.json \
  --store-name test100 \
  --variant two_stage_coherent \
  --variant full_context \
  --variant naive_rag \
  --variant two_stage_fallback
```

After the test store is complete:

```bash
.venv/bin/python scripts/freeze_v2.py --capture \
  --name v2-final \
  --config configs/v2.yaml \
  --manifest results/manifests/test100.json \
  --store-name test100 \
  --variant two_stage_coherent \
  --variant full_context \
  --variant naive_rag \
  --variant two_stage_fallback
```

For `test100`, both freezes also hash the dev100 aggregate and automatic decision.
Changing the selected result or the pre-registered baseline rules therefore stops
the final run instead of silently creating a new experiment.

Run the final preflight without `--run`.  When it passes, add `--run` exactly once:

```bash
.venv/bin/python scripts/run_final_test.py --run \
  --freeze-name v2-final \
  --config configs/v2.yaml \
  --manifest results/manifests/test100.json \
  --store-name test100 \
  --arm v2=two_stage_coherent \
  --arm full_context=full_context \
  --arm naive_rag=naive_rag \
  --arm flat_memory_fallback=two_stage_fallback
```

Quota pauses return 2 and keep ledger status `started`; use the identical command to
resume.  Once ledger status is `complete`, the runner refuses another execution.

## 7. Product hardening

Do not start this from a dev result. After the one-shot final report fixes the
experimental conclusion, use
[`docs/PRODUCTIZATION_V2_PLAN.md`](../docs/PRODUCTIZATION_V2_PLAN.md). It defines
the order and acceptance gates for trusted identity, tenant isolation, hard
deletion/export, backup/restore, cost controls, concurrency, privacy, and
operations. None of those changes may be folded back into the already measured v2
artifact.
