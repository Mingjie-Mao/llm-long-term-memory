# v2b batch-8 execution registration

**Registered 2026-09-16 before creating the `v2b-train150` store.**

**Status: superseded before any successful extraction.** The attempt reached the
provider with an already exhausted daily quota, then stopped with zero completed
sessions and zero memories. `prereg-v2b-gate16.md` is the active protocol; this
file and the failed-call usage ledger are retained only as an audit trail.

## Purpose

Run the already-selected batch-8 ingestion candidate at benchmark scale. This is
an engineering-development run, not a new final test: all 500 LongMemEval-S
questions have already been exposed somewhere in the project.

The only intended system change from frozen v2 is:

```text
ingest.sessions_per_request: 15 -> 8
```

`tests/test_batch_size_curve.py::test_v2b_changes_exactly_one_thing_about_v2`
enforces that claim. Retrieval, context construction, fallback, answerer, judge,
embedding model, prompts, schema, and temporal policy remain unchanged.

## Execution boundary

| field | registered value |
|---|---|
| config | `configs/v2b-batch8.yaml` |
| question manifest | `results/manifests/train150.json` |
| store | `v2b-train150` |
| dataset | `longmemeval_s_cleaned.json` |
| role | readable development and scale validation |
| resume policy | checkpointed resume under the same ingest fingerprint |

The existing `train150` batch-15 store is not modified. The new store must have
its own database, vector index, checkpoint, extraction cache, and usage ledger.

## Ordered checks

### Gate 0 — before ingestion

1. Focused tests for the batch-size config, ingest fingerprint, and pipeline pass.
2. The store name has no existing artifacts.
3. A content-addressed pre-ingest freeze is captured with this registration in its
   protocol paths.
4. No other ingestion process holds the project ingest lock.

### Gate 1 — ingestion completeness

After quota-resumable ingestion finishes:

- every manifest session is terminal in the checkpoint;
- store, checkpoint, vector ids, and vector matrix agree;
- the stored ingest fingerprint resolves to batch size 8;
- refused sessions and failed requests are reported rather than silently excluded;
- actual requests and tokens are retained in the usage ledger.

An incomplete store is never evaluated.

### Gate 2 — write-path quality

Before end-to-end QA, report the already-defined zero-call diagnostics:

- memories and user facts per session;
- zero-yield rate;
- specific-information fidelity by facet;
- duplicate and supersession counts;
- source-session and structured-literal coverage where measurable.

The 60-session fidelity result (64.9% at batch 8) is the prior measurement, not a
guaranteed scale result. A material reversal opens a write-path investigation and
stops QA spending.

### Gate 3 — paired development QA

If Gates 1 and 2 pass, compare the existing batch-15 `train150` store with
`v2b-train150` under the same frozen v2 answer path. Read endpoints in this order:

1. memory-only accuracy;
2. fallback trigger rate and fallback success rate;
3. final accuracy;
4. median answer context and token usage;
5. paired wins/losses and exact McNemar statistic.

Runs are repeated three times per arm before interpreting a small difference. The
result remains a development result because `train150` is readable and has been
used for prior development.

## Decision rule

Promote v2b as the new ingestion default only if all of the following hold:

1. ingestion is complete and homogeneous;
2. write-path coverage/fidelity improves in the registered direction;
3. final accuracy has at least as many paired wins as losses against batch 15;
4. memory-only accuracy or fallback demand improves in the expected direction;
5. the measured ingestion-cost increase is accepted explicitly in the result.

No LongMemEval number from this run will be described as unseen, final, or SOTA.
