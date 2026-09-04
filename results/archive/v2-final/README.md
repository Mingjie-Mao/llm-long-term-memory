# v2 final conclusion archive

This directory preserves the answer-affecting v2 source exactly as it existed at the
`v2-final` freeze and records a checksummed, aggregate-only conclusion manifest.

- `source.tar.gz` contains the 105 source/configuration-runtime files named by
  `results/frozen/v2-final/freeze.json`.
- `conclusion.json` is generated once by `tools/verify_v2_conclusion.py --capture`.
- The manifest references the final aggregate, ledger, configuration, protocol evidence,
  usage records and sealed row files by SHA-256.
- It does not copy the test dataset, question manifest, memory store or per-question rows
  into this archive directory. Their frozen hashes or existing sealed paths remain part of
  the evidence chain.

Verify the preserved conclusion without calling a model or reading test rows:

```bash
.venv/bin/python tools/verify_v2_conclusion.py
```

The combined input fingerprint in `conclusion.json` is SHA-256 over the frozen test
manifest hash, a NUL byte, the frozen dataset hash and a final NUL byte. This binds both
the selected 100 questions and their source dataset without exposing either.
