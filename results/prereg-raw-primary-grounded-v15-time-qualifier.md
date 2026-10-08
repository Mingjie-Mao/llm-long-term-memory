# V15 offline precision correction — before any v15 real calls

Initial zero-call v14-response replay computed the correct8weeks but marked it About:
an unrelated phrase "thinking about ..." triggered the calculator's legacy global
approximation matcher. Use the existing scoped_time_qualifiers=True mode, as v10+
already does: only approximation words modifying the actual time expression matter.
Add tests for unrelated about and a genuine approximate relative event date.
Preserve initial replay/context/preview and initial source snapshot; generate r2
replay and context/preview paths. Execute v15 using --offline-replay r2 train path,
and freeze its identity. No new question exclusion, model, budget or gate change.
