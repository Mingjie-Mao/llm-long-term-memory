# v3 answer pilot conclusion archive

This directory preserves the exact answer-affecting source used for the frozen,
train-only v3 answer pilot and a checksum inventory of its aggregate evidence.

- `source.tar.gz` contains the 107 files named by the pilot freeze.
- `conclusion.json` is generated once and cannot be silently replaced.
- The conclusion references the freeze, configuration, pre-registration, fixed
  48-question manifest, aggregate report and all 12 sealed row/usage files.
- The public aggregate contains no question ids, question text, answers or judge reasons.

Verify locally without calling a model:

```bash
.venv/bin/python tools/verify_v3_answer_pilot.py
```
