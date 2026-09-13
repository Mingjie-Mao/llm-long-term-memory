# Superseded before provider evaluation

No provider request was ever made under this freeze: `launch-status.json` records
`provider_attempts: 0`, and the rehearsal it passed used canned replies.

It was retired because a second reading was registered into the protocol —
enumeration completeness, in `prereg-v4.2-execution-addendum.md` and
`tools/v42_protocol.py` — after the binary gate was found to promote a genuine
six-probe improvement only about 31% of the time. Both of those files are bound by this
manifest's hashes, so `verify` correctly refuses it. That refusal is the mechanism
working, not a fault.

The rehearsal evidence here is kept: 404/404 rows, 404 fake attempts, Gate 0 PASS,
90/90 candidate count answers through the citation path, and a clean resume after an
interruption at row 17. It attests the schedule and the harness, which did not change.

The freeze to execute is `../v4.2-development-20260912b/`.
