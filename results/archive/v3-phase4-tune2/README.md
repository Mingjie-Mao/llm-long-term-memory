# v3.2 phase-4 tune42 archive

This directory preserves the completed adaptive source-evidence comparison.

- The frozen v2 baseline reached 66.7%; v3.2 reached 73.8% on the same 42 questions.
- Temporal accuracy improved from 40% to 60%, multi-session accuracy stayed at 70%,
  and ordinary-question accuracy improved from 82.4% to 88.2%.
- The pre-registered gate still stopped promotion because median selected context was
  2.52 times the v2 baseline, above the fixed 2 times limit.
- `source.tar.gz` contains the exact 109 answer-affecting files named by the freeze.
- `conclusion.json` binds the freeze, config, protocol, aggregate, report, source
  snapshot, reused v2 baseline and new v3.2 row/usage files.
- The aggregate exposes no question ids, question text, answers or judge reasons.

Verify locally without calling a provider:

```bash
.venv/bin/python tools/verify_v3_phase4_tune2.py
```
