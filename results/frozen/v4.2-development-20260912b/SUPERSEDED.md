# Superseded after an aborted live attempt

Three provider attempts and 3,188 tokens were spent under this freeze before it stopped
itself on row 2. The evidence and the diagnosis are sealed at
`results/archive/v4.2-aborted-20260912/`.

Two things changed afterwards, both bound by this manifest, so `verify` correctly refuses
it:

- `tools/v42_protocol.py` — the per-row check gated a diagnostic that is documented as
  true even when the answer reads correctly. It now gates what reaches the reader.
- `src/.../synthesis.py` — the enumerate prompt's count block let the model conclude that
  citing labels discharged the prose reply. It now says otherwise.

The freeze to execute is `../v4.2-development-20260912c/`.
