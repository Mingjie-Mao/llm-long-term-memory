# v3.3 phase-5 tune42 archive

This directory preserves the completed compact session-fair evidence comparison.

- The frozen v3.2 reference and v3.3 both reached 73.8% on the same 42 questions.
- Temporal accuracy stayed at 60%, multi-session accuracy stayed at 70%, and
  ordinary-question accuracy stayed at 88.2%.
- Median selected context fell from 1,476.5 to 1,124 tokens, moving from 2.52
  times to 1.92 times the v2 baseline and passing the registered 2 times limit.
- Every pre-registered tune gate passed. This permits, but does not itself run,
  the separately authorised one-shot dev60 validation.
- `source.tar.gz` contains the exact 111 answer-affecting files named by the freeze.
- `conclusion.json` binds the freeze, config, protocol, aggregate, report, source
  snapshot, reused v2/v3.2 references and the new v3.3 row/usage files.
- The aggregate exposes no question ids, question text, answers or judge reasons.

Verify locally without calling a provider:

```bash
.venv/bin/python tools/verify_v3_phase5_tune3.py
```
