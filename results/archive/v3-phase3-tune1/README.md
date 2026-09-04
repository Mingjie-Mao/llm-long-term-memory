# v3.1 phase-3 tune42 archive

This directory preserves the completed tune42 comparison as a negative result.

- v2 and v3.1 each produced 42 sealed rows.
- Both reached 66.7% overall; v3.1 was lower on the temporal and multi-session slices.
- `source.tar.gz` contains the exact 108 answer-affecting files named by the freeze.
- `conclusion.json` binds the freeze, config, protocol, aggregate and four row/usage files.
- The aggregate exposes no question ids, question text, answers or judge reasons.

Verify locally without calling a provider:

```bash
.venv/bin/python tools/verify_v3_phase3_tune1.py
```
