# Aborted before provider evaluation

This first preparation attempt was rejected during rehearsal preflight because the
manifest accidentally included SQLite's transient shared-memory sidecar. No rehearsal
answers and no provider requests were made under this freeze. Its manifest and source
archive are retained as evidence of the failed preflight; do not execute it.

The corrected frozen package is `../v4.2-development-20260912/`. It closes the backup
connections and sets the copied database to DELETE journal mode before hashing.
