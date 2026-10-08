# V11 exact execution preflight

2026-10-02. Same exposed public train150 manifests: 9 known reasoning errors and
10 originally correct cases (19 total), development/regression evidence only.
Baseline is saved v10 reader traces plus original v1 outcomes; candidate v11.
No significance/overall-accuracy claim from these enriched questions. Existing
registered paired dev100 3-rep gate is not executed until these known source-supported
failures and regressions pass. Original gemini-3.5-flash-lite reader,
gemma-4-31b-it judge and fallback.yaml extraction/embedding configuration unchanged.
Temperature0, unsupported provider seed, typed V7 provider schema, prompt memory-grounded-v11.

Targeted162 passed. Full suite2121 passed27.60sec in unsandboxed test execution;
initial sandbox run was blocked only by binding localhost in timeout fixture.
Ruff check and format checks passed354 files, git diff check passed.
Zero-call saved-v10 replay (r2) computes book total8weeks and literal next chess
entry28, and discloses scoped3/five reports instead of one invented total.
Context/19-first-request preview must match candidate source SHA before provider calls.

Local cached embeddings only: HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1. Initial
preflight commands without those variables were interrupted while optional
HuggingFace HEAD checks retried in the restricted network; no provider calls/results.
Full context replays use unchanged cached MiniLM; final files have new v11 paths.
19 reader requests minimum,38 maximum before SDK retries; input/output token cost
estimated up to400k for readers from prior same-pool runs. Judge19 calls before
service retries (prior judge failures may consume requests/tokens). Save every
answer before any grading. No other user data is included; exact payload preview
results/analysis/grounded-reader-v11.request-preview.json. External real diagnostic
already authorized by user. All usage/errors and exact file inventory/SHA/environment
retained. No commits, no default promotion, no framework or model switch.
