# V8 preflight revision 2 — before any v8 provider calls

2026-10-02. The first zero-call v8 replay and request preview are retained under their
original paths. The initial full suite reported 1 failed, 2058 passed: calendar-date
recovery incorrectly accepted invented event prose whose relative date coincided
with a memory's date. This violates the original v8 registration's calendar-only
recovery requirement. No v8 external reader or judge call was made with that source.

Restrict recovery to a bare independently parsed calendar date: parsing without a
session anchor must yield day precision and the parsed expression must equal the
entire supplied quote. Wrong event prose and relative expressions must refuse;
same-session/turn unique raw provenance rebinding remains separately checked.
Add negative regression coverage rather than weakening the provenance test.

Repeat zero-call train150 and dev100 context-only replays in fresh v8-r2 paths, and
create grounded-reader-v8.request-preview-r2.json. The reader diagnostic accepts an
explicit --offline-replay path, verifies the source SHA and original coverage gates,
and records that replay's SHA in its execution inventory. Original v8 output namespace,
same 19 questions, unchanged models, bounded two-pass selection and acceptance rules
remain as registered. Capture this addendum and matching source before provider calls.
