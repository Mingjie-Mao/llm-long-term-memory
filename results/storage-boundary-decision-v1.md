# Storage boundary — 2026-09-24 decision

This is an operational audit, not a performance benchmark or a production-readiness
claim. No storage migration is authorized by the evidence currently on disk.

The application still uses SQLite as the durable memory source and a NumPy vector
index as a separately persisted derivative. A process-local lock serializes service
operations, but it does not coordinate multiple processes or make SQLite plus three
index files one transaction. High-QPS, multi-instance or multi-process write
requirements have not been specified or measured. A new vector database would not
solve extraction loss or the frozen 86%/72% answer gap.

Read-only validation on the existing `v2b-gate16` and
`v2b-gate16-repair` stores found exact SQLite/index ID agreement: 3,055 and 3,299
memories respectively. This only checks identity parity at the audit instant; it
does not prove atomicity, concurrent-writer safety or a 30-day soak. Startup validates
index IDs against SQLite and fails on a mismatch; `lltm lifecycle rebuild-index`
can regenerate the derivative index from durable rows after taking the service
offline. Backup/restore has a separate index check and rebuild path.

Decision: retain SQLite + NumPy for the research and single-process integration
profile. Document and enforce single-writer deployment, backup/restore rehearsal and
rebuild-on-mismatch. Reopen storage architecture only after registering a concrete
load and recovery target (writers/processes, memory count, QPS, latency, crash point,
acceptable recovery time), then test the existing design against it. The current
two-file consistency limitation remains open rather than being labelled solved.

## Update 2026-09-28 — single writer enforced, not only documented

The decision above said "document and enforce single-writer deployment". Enforcement
was missing for the service: only the ingest CLI took the store lock, so a service and
an ingest — or two service processes — could each hold the vector index in memory and
save it over the other's. Now:

- `MemoryService` holds the store's writer lock (`<store>.db.lock`, the one the ingest
  CLI already used) for its lifetime; a second writer process fails at startup with
  `AlreadyRunning`. Re-entrant within one process, released on `close()`.
- `lltm lifecycle rebuild-index` takes the same lock, so it cannot run under a live
  service.
- `MemoryService(read_only=True)` opens SQLite read-only and takes no lock, for
  offline analyses that read while an ingest writes.

Tests: `tests/test_store_writer_lock.py`, including a real second process. This
forbids concurrent writers; it does not support them, and it does not make SQLite plus
the index files one transaction. The crash story is unchanged: SQLite is the source of
truth, the index is verified at startup and rebuilt from it on mismatch.
