# V12 real-reader execution preflight

Same exposed train15019 cases/manifests, configs/fallback.yaml; reader
 gemini-3.5-flash-lite, judge gemma-4-31b-it, prompt memory-grounded-v12,
original typed V7 schema, temperature0, provider seed unsupported.
Local embedding/cache only with HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1.
Source-supported acceptance rules fixed in v12-acceptance before any new outputs.

Zero-call resolution projection:19 cases,2.3 expected unstable questions; single-run
net below4 is not distinguishable from noise. This is the already failure-enriched
mechanism/regression diagnostic, not a new aggregate improvement claim. All17 source-
supported cases and2 predeclared ambiguous cases have hard acceptance conditions;
a pass only permits the larger registered3-replicate paired validation. No statistical
significance/overall accuracy claim from this one diagnostic. A mechanism may show
up on fewer than4 cases without establishing stable aggregate net improvement.

Zero-call v12 context replays: train150 annotated reach146/146, no body mismatch,
median5815.5/max5999; dev100 context-only median5842.5/max5998. Source SHA matches
exact19 first-request preview. Public-source proof PASS:1119 raw sources exactly
match the upstream public dataset SHA d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442,
380 derived memories have public origin and verified tenant. No private data.

Targeted172passed; full2140passed33.24s; Ruff check/format355files/diff check passed.
At most38 reader requests before SDK retries, estimated up to400k tokens;19 judge
requests before service retries only after known source-supported reader failures
pass. Reader outputs saved before grading; failures and exact usage preserved.
Execution inventory freezes all runtime/config/store/index/source/prereg identities
and git/environment. Archive matching source text before any future modifications.
User authorized completion, external benchmark verification and no commits. Do not
start recursive-summary calls before selection acceptance and broader regression gate.
