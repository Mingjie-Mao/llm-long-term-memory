# Research tooling index

This index reduces the search surface without moving or deleting scripts that are part of
historical evidence. Frozen, sealed and archived paths remain unchanged.

## Stable operational tools

- `tools/backup_restore.py` — backup, restore, integrity checks and optional index rebuild
- `tools/operate.py` — operational checks
- `tools/temporal_transition_replay.py` — zero-call lifecycle regression replay
- `scripts/repo_stats.py` — repository inventory
- `scripts/check_ingest_state.py` — database/index/checkpoint consistency
- `tools/register_hidden_set.py` — contamination checks for genuinely new sets
- `tools/resolution.py` — QA instrument resolution before a model-calling experiment

## Active analysis tools

- extraction/store fidelity: `tools/store_fidelity.py`, `tools/specificity_repair_pilot.py`
- failure attribution: `tools/failure_taxonomy.py`, `tools/retrieval_replay.py`
- synthesis/count probes: `tools/synthesis_probes.py`, `tools/run_synthesis_probes.py`,
  `tools/count_stage_decomposition.py`
- relation probes: `tools/relation_routing_probe.py`, `tools/routed_scan_gain.py`

## Historical version-bound tools

Files named for v2b/v2c/v2d/v3/v4/v5, `run_v3_*`, frozen verifiers and one-off gate
analyzers reproduce recorded decisions. They are not general product commands. Do not
delete, rename or refactor them merely because static search finds no importer; their
paths and hashes may be evidence.

New reusable functionality belongs under `src/llm_long_term_memory/`. New experiment
orchestration belongs under `research/` or `tools/` with a preregistration and a unique
result namespace.
