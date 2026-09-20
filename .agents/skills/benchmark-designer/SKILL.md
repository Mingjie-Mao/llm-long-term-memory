---
name: benchmark-designer
description: Design or extend a frozen long-term-memory benchmark without split contamination and with auditable gold provenance. Use for new eval sets, temporal-memory benchmarks, hidden/frozen splits, or custom test suites; do not use for running an already-defined benchmark unchanged.
---

# Benchmark designer

Freeze train/development/test membership before tuning. Keep test hidden from model
selection and record provenance for every item. Define the gold-answer policy, flag
ambiguous samples, validate gold with humans or deterministic rules before scoring, and
separate dataset defects from model defects.

Cover only categories needed by the stated objective. Available long-term-memory
categories include exact lookup, temporal update, `as_of`, relative dates, current state,
count, duration, comparison, preference transfer, third-party entities, assistant
recommendations, removal, contradiction, paraphrase, and multi-session aggregation.

Use a manifest with stable IDs and a seed where sampling is involved. Record exposure
status and intended use for each split. Reuse the repository's manifest, hidden-set,
probe-safety, and gate-resolvability tooling. Do not inspect or regenerate a frozen final
split merely to improve a candidate.
