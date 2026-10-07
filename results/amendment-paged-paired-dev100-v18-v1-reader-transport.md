# Amendment 2 to `paged-paired-dev100-v18-v1` — resuming an unsaved answer after a transport failure

2026-10-07 Australia/Sydney, before the restart it allows. Amendment 1
(`results/amendment-paged-paired-dev100-v18-v1-judge-recovery.md`) and the registration
are unchanged.

## Why

On 2026-10-06 18:56 the reader stage of candidate rep3 stopped at 78 of 100 saved
answers: an answer call still timed out (`httpx.ReadTimeout`) after the client's
registered retries (`max_retries 2`, `max_transport_retries 2`, 180 s timeout), and the
exception ended the process. The question's answer was not saved, and no part of it is
used. The registration resumes "whole unsaved questions" after a quota stop and bounds
reader attempts in total (3,960); it does not say what follows a transport failure. At
the stop, all six runs together had used 1,444 answerer attempts.

## Rule

- An answer that was not saved because its call failed with a transport error after the
  registered client retries (`ReadTimeout`, `ConnectTimeout`, `ConnectError`,
  `RemoteProtocolError`, `ReadError`, `WriteTimeout`) is resumed whole, from the same
  frozen inputs, exactly as after a quota stop. Saved answers are never rerun.
- At most three such restarts for the rest of the run, the stop above counted as the
  first. Any other reader failure, or a fourth, stops the run as incomplete.
- The registered 3,960 total reader-attempt ceiling still applies. Every attempt, failed
  ones included, stays in the run's usage files. Each restart is listed under
  `transport_restarts` in `results/analysis/v18-amended-continuation-v1.progress.json`.

This rule does not depend on any answer's content or grade. Both arms are handled
alike, though only the candidate's rep3 answers remain.
