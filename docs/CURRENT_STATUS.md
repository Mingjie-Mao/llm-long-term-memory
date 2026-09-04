# Current status — 2026-09-05

This is the one current-state page. The long reports and execution log are evidence
and history; older entries in them are intentionally not rewritten.

## Experiment

| item | state |
|---|---|
| v1 heldout100 | complete: registered single shot 70/100; repeats 71 and 73 |
| v2 train150 | complete: 7,180/7,180 sessions; 18,519 memories |
| v2 selected context | `mean`, radius 1, cap 30; 95.3% Top-3 and assembled source recall |
| v2 dev100 ingest | complete: 4,791/4,791 terminal; 12,436 memories; state checker reports `COMPLETE`, `issues: []` |
| v2 post-ingest freeze | `v2-candidate` was used unchanged for the completed dev100 run |
| v2 dev validation | complete: 5 arms x 3 repeats = 1,500/1,500 sealed rows; aggregate and hash-bound decision written |
| v2 registered decision | `flat20` / `two_stage_fallback`; 67% majority accuracy and 64.3% mean accuracy |
| v2 final pre-ingest freeze | captured and verified as `v2-final-preingest`; source `ad00da20…`, selected arm plus two baselines |
| v2 test100 ingest | complete: 4,714/4,714 terminal; 12,471 memories; SQLite, vector index, checkpoint and extractor fingerprint agree |
| v2 final freeze | captured and verified as `v2-final`; source `ad00da20…`, completed test store and registered three-arm protocol are hash-bound |
| v2 test100 | **complete, one shot spent**: v2 72%, full context 86%, naive RAG 65% |
| v2 conclusion archive | complete: 32 checksum-bound evidence files plus a 105-file frozen-source snapshot; verifier reports `PASS` |
| v3 answer pilot | **complete**: frozen 48-question train-only comparison, 2 arms x 3 repeats = 288/288 sealed rows |
| v3 pilot result | pre-registered gate **passed**: v3 72.9% versus v2 68.8% majority accuracy; +4.2 points, but not statistically conclusive (`p=0.6875`) |
| v3 pilot archive | complete: 107-file exact source snapshot plus 22 checksum-bound evidence files verify independently |
| v3 phase 3 | **complete negative result**: tune42 v2 and v3.1 both 66.7%; v3.1 temporal and multi-session slices each regressed 10 points, so dev60 was not opened |
| v3 phase 4 | **complete stop-result**: v3.2 reached 73.8% versus v2's 66.7%, but its 2.52x median context exceeded the pre-registered 2x limit, so dev60 remains sealed |
| v3 phase 4 archive | complete: 109-file exact source snapshot plus 14 checksum-bound evidence files verify independently |
| v3 phase 5 | **complete; tune gate passed**: v3.3 retained 73.8% accuracy and every registered slice while reducing median context from 1,476.5 to 1,124 tokens (1.92x v2) |
| v3 phase 5 archive | complete: 111-file exact source snapshot plus 17 checksum-bound evidence files verify independently |
| v3 dev60 candidate | **re-frozen after an aborted first execution**: the 2026-09-04T19:06:09Z start hung on its first answerer request and produced zero rows; the provider client now has a request timeout, the freeze was re-captured, and no ledger or result row exists |

The exact sealed order remains in [`results/v2-runbook.md`](../results/v2-runbook.md).
All 15 dev100 repeats and all three final arms contain exactly 100 rows. Each aggregate
binds its row and usage files without exposing question ids, answers, model answers or
judge reasons. The final one-shot ledger is `complete`, records 100 rows per arm and
forbids a fresh second run.

### Final result, in plain language

| method | accuracy | median context tokens | answer + judge tokens | practical reading |
|---|---:|---:|---:|---|
| v2 memory + raw fallback | **72%** | **574** | 159,458 | Chosen product: compact and better than ordinary RAG, but not the accuracy ceiling |
| full conversation | **86%** | 109,059 | 10,932,294 | Best accuracy, but about 190x the context and 69x the answer/judge tokens of v2 |
| naive RAG | **65%** | 12,763 | 1,352,847 | Simpler retrieval is 7 points below v2 while using about 22x its context |

The v2 test result is consistent with the earlier held-out result rather than collapsing
on unseen data: 72% versus the v1 heldout100 runs of 70%, 71% and 73%. Against naive RAG,
v2 is 7 points higher, but one 100-question test is not enough to prove that gap is
stable (`p=0.337`). Full context is 14 points higher than v2 and that paired difference
is strong on this test (`p=0.0043`), at much higher token cost.

The main v2 problem is now clear. It selected a source-containing context for 94% of
questions but answered only 72% correctly. Of the 28 wrong answers, only 3 were caused
by failing to retrieve the source conversation; 14 were still wrong after raw fallback
and 11 were wrong despite already having the source context. The next accuracy work is
therefore answer synthesis and multi-session/temporal reasoning, not simply retrieving
more text.

Across the shared test ingestion and all final answer/judge calls, the recorded total is
2,033 API requests, 48 failed attempts and 26,202,312 tokens. This is usage evidence,
not a currency bill; a dollar figure requires the provider's actual billed model/rates.

### Freeze timeline clarification

The historical `v2-candidate` freeze no longer verifies against today's working source
because `validation.py` was strengthened after the dev decision to bind all five arms.
That does **not** invalidate the dev result: the decision was produced at 14:21 on
2026-08-31 using the still-frozen old source, the strengthening happened at 16:39, and
`v2-final-preingest` was captured at 16:41 with the strengthened source. The actual
test100 run used the later `v2-final` freeze, which verifies successfully.

The registered coherent candidate saved context (343 versus 576 median tokens) but failed
the accuracy gate (63% versus 67% majority accuracy), so the automatic rule retained the
flat memory-plus-raw-fallback product. The report-only `naive_rag` baseline reached 68%
majority accuracy but used 13,416 median context tokens and was not allowed to rewrite the
pre-registered product decision.

### Completed phase-boundary checks

- `load_dev_decision` now validates all 30 sealed row/usage artifacts while keeping the
  product decision limited to the three pre-registered decision arms. A five-arm
  regression test and the real dev100 decision both pass.
- The complete offline suite has 597 passing tests, Ruff reports no violations, and core
  package line coverage remains 83%. Including one-off research scripts lowers the
  combined number to 62%; this is not treated as a reason to write low-value tests for
  retired analyses.
- `configs/v2.yaml` still enables the selected flat raw-fallback behaviour. The final
  freeze binds that config, the corrected source, the dev100 aggregate and decision,
  the registered three-arm final protocol and the completed test100 store.

## Product code

The post-freeze repair branch at `06b567e` fixes the default REST/MCP write composition, uses the
real batch extractor through a single-turn adapter, serializes access to the shared
SQLite/index resources, avoids mutating a shared answer runner, externalizes session
ids consistently, and makes `/healthz` a readiness check. The public playground now
persists expiries, sweeps after restart and in the background, validates inputs, and
uses a hashed dependency lock without the provider SDK.

These source changes are deliberately separate from the completed v2 experiment. The
archived source snapshot and conclusion verifier preserve the exact answer-affecting v2
source even though current development has moved on to v3.

The public showcase was rebuilt on `main` at `3026b78`: the browser-local guided tour
works before the backend wakes, the real engine is isolated as a second layer, mobile
uses a three-step pager, and published figures come from a release manifest. The planned
physical separation of product, research, negative results, and audit evidence is in
[`ARCHITECTURE_SEPARATION_PLAN.md`](ARCHITECTURE_SEPARATION_PLAN.md); it intentionally
does not move frozen source files yet.

A static-site-only truthfulness and accessibility hotfix was deployed on 2026-08-29 as
Cloudflare Pages deployment `18c9bea7`. The production page now describes the 60-minute
lifetime as access expiry rather than a restart-durable deletion guarantee, exposes the
release id/date/source commit/manifest, uses the step pager at tablet widths, creates a
fresh namespace for every preset run, and translates accessible names with the visible
copy. This dirty-worktree deployment deliberately changed no frozen experiment source;
its exact seven-file checksums are recorded under `public-demo/deployments/`.

## Still not a production service

The current API has no trusted tenant identity: `user_id` is supplied by the caller.
Main-service deletion is a provenance-preserving soft delete, not a data-subject hard
delete. PostgreSQL/pgvector, OIDC/JWT, export, backup/restore drills, quotas, alerts and
SLOs require deployment and policy choices and remain the P0–P5 productization work in
[`PRODUCTIZATION_V2_PLAN.md`](PRODUCTIZATION_V2_PLAN.md). They must not be represented
as complete merely because the local prototype passes tests.

## Current action and remaining plan

| order | work | current state | completion standard |
|---:|---|---|---|
| 1 | Preserve the v2 conclusion | **done** | Aggregate result, configuration, protocol, hashes and exact frozen source verify independently |
| 2 | Improve v3 answer reasoning | **done; pilot gate passed** | v3 reached 72.9% versus 68.8% for v2 on the same 48 train-only questions, with all pre-registered safety/cost checks passing |
| 3 | Add multi-session and time-reasoning tests | **tune1 done; v3.1 rejected** | The 84-row comparison tied overall and regressed both target slices; exact negative result and source are archived |
| 4 | Add adaptive context | **tune2 done; v3.2 stopped on cost** | Accuracy and all safety checks improved or held, but median context was 2.52x v2 against a fixed 2x ceiling |
| 5 | Compress v3 evidence | **done** | Exact source sentences are deduplicated and allocated round-robin across sessions; observed median is 1,124 tokens (1.92x v2) |
| 6 | Run final tune42 iteration | **done; PASS** | Exactly 42 v3.3 rows retained v3.2's 73.8% accuracy and passed all 11 registered checks |
| 7 | Use sealed dev60 once | **candidate frozen; awaiting separate API consent** | Run the untouched 60-question set exactly once as two arms x three repeats (360 rows), then report only aggregate results |
| 8 | Create a new hidden final set | pending | Freeze genuinely unseen data and run the final v3 only once after all choices are fixed |
| 9 | Add product safety basics | pending | Trusted login, tenant isolation, hard deletion, export and permission checks |
| 10 | Improve data reliability | pending | Automated backup, successful restore drill, safe concurrency and migrations |
| 11 | Add cost and operations controls | pending | Per-user quota, cost alerts, rate limiting, logs and health checks |
| 12 | Run a small real-user pilot | pending | No cross-user data, deletion works, cost is bounded and key actions are traceable |

The completed pilot used 698 recorded requests, including 19 failed attempts, and 492,615
input-plus-output tokens. The v2 arm used 231,311 tokens; the v3 arm used 261,304 (13.0%
more total tokens and 19.5% more output tokens). Median selected context did not increase:
605 tokens for v3 versus 614 for v2. The six row files and six usage files are sealed by
hash in `results/validation/v3-answer-pilot-aggregate.json`; the public aggregate contains
no question ids, question text, answers or judge reasons.

This is a positive development signal, not final proof. On the 48 paired majority outcomes,
v3 won four questions and lost two, so the exact paired result is not statistically conclusive
(`p=0.6875`). The correct next step is a newly built reasoning-focused development set, not a
rerun or inspection of the already-spent test100 set.

For phase 3, the 102 train150 questions not used by the pilot were deterministically divided
into an inspectable `tune42` and sealed `dev60`. The completed tune42 comparison produced
42 rows per arm: v2 and v3.1 both reached 66.7%, while v3.1 fell from 40% to 30% on temporal
questions and from 70% to 60% on multi-session questions. It therefore cannot proceed to
dev60. The negative result is preserved with a 108-file source snapshot and 14-file evidence
inventory.

Tune1 diagnosis found that every question reached at least one labelled source conversation,
but the old recall metric measured only "any source hit". Among v3.1's 14 wrong answers, 12
actually reached all labelled source conversations; compact memories still lost an exact
number, relative date, event state or list member. v3.2 therefore hydrates up to 800 tokens of
verbatim source only for temporal, aggregation and current-state operations, records full
session coverage, and leaves direct/preference questions short.

The completed v3.2 tune2 run reused the sealed v2 baseline and added exactly 42 candidate
rows. Overall accuracy rose from 66.7% to 73.8%; temporal accuracy rose from 40% to 60%,
multi-session held at 70%, and ordinary questions rose from 82.4% to 88.2%. Four paired
questions changed from wrong to right and one changed from right to wrong, giving +7.1 points
with an inconclusive exact paired result (`p=0.375`). Answer output rose only 3.9%, and
hydration was confined to the registered operation types. However, median context rose from
586 to 1,476 tokens (2.52x), failing the pre-registered 2x ceiling; therefore the formal result
is `STOP`, not a promotion to dev60. The 42-row result, reused baseline, freeze, protocol,
configuration and exact 109-file source snapshot are preserved under
`results/archive/v3-phase4-tune2/` and pass the local verifier.

The immediate engineering target was narrow: hydrated questions consumed 760–800 extra
tokens almost every time, so v3.3 needed to preserve the useful exact source sentence while
dropping neighbouring prose and lowering the hydration budget. The choice was made and
tested on tune data only. No dev60 or test100 content has been opened or sent.

That v3.3 local step is now complete. The compact hydrator keeps retrieval rank within each
session but allocates evidence round-robin across sessions, removes duplicate renderings of
the same source sentence, attaches no neighbouring sentence, and enforces a 425-token cap.
The old ranked allocation remains the default, so existing configurations do not silently
change. A zero-provider-call replay over tune42 reconstructed v3.2's recorded evidence and
projects a median context of 1,124 tokens (1.92x v2, versus v3.2's observed 1,476/2.52x).
Mean labelled-source hydration is projected at 94.9% versus v3.2's 94.3%, with the same
85.7% full-coverage rate. This proves the cost path, not answer accuracy.

The final tune42 iteration is pre-registered and frozen as `v3-phase5-tune3` with source
hash `5afc4af2…` and config hash `c181cbbb…`. It created exactly 42 new
`v3.3-compact` rows and reused the hash-bound v2/v3.2 references. v3.3 tied v3.2 at
73.8% overall, 60% temporal, 70% multi-session and 88.2% ordinary accuracy. Median
context fell from 1,476.5 to 1,124 tokens, or from 2.52x to 1.92x v2; answer output
rose only 2.4%, within the registered 1.1x ceiling. All 11 registered checks passed.

The completed v3.3 arm recorded 243 API attempts, including 142 transient provider or
network failures, and 125,109 input-plus-output tokens. Those failed attempts were
retried and did not become partial evaluation rows: the sealed file has exactly 42
complete answer-and-judge rows. The result, six referenced row/usage artifacts,
configuration, protocol and exact 111-file source snapshot are preserved under
`results/archive/v3-phase5-tune3/`; the local verifier reports `PASS` with a 17-file
evidence inventory. The complete offline suite had 597 passing tests at freeze time.
dev60 and test100 remain sealed and unseen. Passing tune3 permits, but does not
authorise, the one-time dev60 run.

The dev60 execution is now prepared without opening its question text. A prospective
protocol amendment records why the early phase-3 1x-context line is replaced by the
2x-v2 ceiling registered during the later evidence-hydration work; this was fixed
before any dev60 provider call or result. All original accuracy, slice, confidence,
cost and one-shot rules remain. The candidate freeze is `v3-candidate-dev60`, with
source hash `0dfdb56e…` and the unchanged v3.3 config hash `c181cbbb…`. Its runner
writes a durable ledger before the first call, resumes the same row files after
interruptions, forbids a completed rerun, and exposes only aggregate output. The
offline preflight passes and the expanded full suite has 606 passing tests. Starting
the 360-row API run still requires explicit permission to send the 60 dev questions
and their retrieved context to Google Gemini.

### The first dev60 execution was aborted with zero rows

That run started at 2026-09-04T19:06:09Z, issued exactly one answerer request, and then
blocked on it for 31 minutes: the socket stayed `ESTABLISHED` with both queues empty,
the process accumulated 13.7 seconds of CPU time, and the quota counter never advanced
again. `GeminiClient` built the provider client without `http_options.timeout`, which in
google-genai 2.17.0 means no timeout at all, so a half-open connection blocked forever.
Neither retry budget could fire, because both are driven by caught exceptions and a
blocking read raises nothing. The same defect is the better explanation for the
multi-minute stalls recorded during tune3 than the 503s they were filed under.

Nothing was spent. Zero rows were written, no usage checkpoint was produced, and no
dev60 question, answer, judge verdict or aggregate was generated or read. The empty
ledger and its empty row file were deleted, and the aborted ledger is preserved verbatim
in `results/audit/dev60-transport-hang-abort-20260904T190609Z.json`. The one-shot rule
guards against repeating a run whose result was seen; there was no result.

The fix is a 180-second provider request timeout plus four tests, including one that
drives a real socket which accepts and then never answers. 180 seconds is not tuned to
make a run pass: the slowest *successful* answerer call in tune3 took 101 seconds, so a
tighter bound would turn a slow provider into missing rows. The candidate, config,
manifest, store, arms, repeats and every gate are unchanged, and the timeout affects both
arms identically. The freeze was therefore re-captured; the superseded freeze the aborted
run was bound to is kept beside it as `freeze.superseded-20260904T190609Z.json`, so the
`freeze_sha256` in the preserved ledger stays checkable.

Two things are worth stating plainly before the rerun. First, the answerer quota is the
binding constraint, not the algorithm: `gemini-3.5-flash-lite` allows 500 requests a day
and 264 were already spent on 2026-09-04 Pacific, while 360 rows need at least 360
answerer calls and more where the fallback fires, so this run spans at least two quota
days and will genuinely exercise the resume path.

Second, the repository is now clean against both CI gates, and getting there had to
respect the archive. `ruff format --check` failed on 13 files carried in from the v3
work, and they were not all the same kind of file. Nine were formatted: the four free
test files, plus five inside the dev60 freeze, which is legitimate precisely because
that freeze has not been spent — whitespace changed, no gate did, and the freeze was
re-captured afterwards. The other four are `tools/verify_*.py`, and those must never be
formatted: each archived `conclusion.json` records the byte length and sha256 of the
verifier that produced its verdict, and the verifier checks that inventory including its
own entry, so a formatter pass would invalidate the archive it exists to prove. They are
excluded in `pyproject.toml` under a rule rather than a list, since every archived
verifier is frozen evidence by construction. The public release manifest was also stale
at 560 tests and now records the real 606.
