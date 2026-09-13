# Current status — 2026-09-13

This is the current-state page; the phase narratives below are historical evidence.

## Latest verification — 2026-09-13

Current checkout: `main` at `879e645`, plus the offline reread repairs described here.

- The requested v4.2 statistics/budget addendum, 404-row schedule, offline rehearsal,
  freeze and live development comparison are complete. The executed freeze is
  `results/frozen/v4.2-development-20260912e/`; earlier freezes are aborted/superseded.
- Live run: 404 rows, 426 attempts, 646,599 observed tokens, two failed attempts.
  Gate 0 passed; the registered outcome is `not_promoted`. The frozen conclusion's
  23 analysis fields reproduce exactly and its source/data/artifact hashes verify.
- Across the three abandoned journals plus the final journal: 447 attempts and
  675,012 tokens. The final preflight's ten attempts are already in the final journal.
- Original count majority scores: control 4/30, candidate 5/30. A provisional model-made
  gold overlay changes these to 12/30 and 7/30; the descriptive clustered p is .125.
  This does not prove harm or replace the registered conclusion.
- Equal unnamed-member totals (47 versus 47) do not prove identical member selection:
  only 50/90 paired count cells match the same members under the existing matcher.
  The old claims of certain falsification and zero effect have been qualified.
- Gold correction remains provisional: 38 model decisions, zero human decisions.
  Development totals are 149→118 memory members over 30 probes, not the old 271→240
  figure that included held-out probes. The reread now rejects incomplete repeats,
  changed evidence and corrections outside development, and binds its inputs by hash.
- A 38-entry human review packet and blank decision template are ready. Next, review
  membership rules and build a separately versioned entity-counting generator; preserve
  the original probe file and held-out split. No new provider call was made in this audit.
- REST credential-derived identity, export and erasure are implemented. Production
  operations, account budgets and independent final-test data remain unfinished.
- v2 remains 72% / 86% / 65%; v3 dev60 remains 55.0% versus 46.7%, with eight
  meaningful gates out of nine originally reported. No historical benchmark was reopened.

See [v4.2 result](../results/v4.2-result.md),
[gold correction](../results/count-gold-correction.md), and
[human review packet](../results/review/count-gold-review-20260913.md).

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
| v3 dev60 | **complete, one shot spent**: 360/360 rows, ledger `complete`, nine registered gates recorded as passing of which eight carry evidence; v3.3 55.0% against v2's 46.7% majority accuracy at 1.95x context |
| v3 dev60 archive | complete: 112-file exact source snapshot plus a 23-file evidence inventory; `tools/verify_v3_dev60.py` reports `PASS` and eight tests pin that it fails on tampering |
| v4.0 answerer | **implemented and measured on development probes**: timeline rendering for supersession chains, an operation-first verdict, and deterministic count/duration in Python. Runs as variant `two_stage_synthesis`, which keeps v3.3's retrieval and hydration so the registered comparison differs only in the answerer. Three defects were found and fixed after it first looked finished: the policy no variant could select, the derivation nothing recorded, and a count answer format the probe grader misread |
| v3 verdict-schema defect | **found and fixed**: `_complete` sent the base schema while parsing with the subclass, so every v3 extended field held its default. `answer_confidence` read `medium` on 100% of v3 rows in every phase, which made the registered dev60 `no_new_confident_errors` gate pass without testing anything — 8 of 9 gates were meaningful. Accuracy is unaffected; recorded in `results/audit/v3-verdict-schema-never-sent-20260906.json` and the archive is not rewritten |
| v4 probe holdout | **split and locked before any v4 measurement**: 188 of 229 probes had never been answered, so 87 are held out (count 30, duration 30, comparison 27) and 142 are development. `current_state` had only 13 untouched and stays whole rather than yielding a 6-probe stratum. The tool refuses to re-draw |
| relation routing probe | **complete, zero provider calls**: templated upper bound 80.0%, and **every error is `owns` vs `acquired`** — a vocabulary defect, not a routing one. On 72 real questions, 31 route at a margin below 0.05, and the confident routes are confidently wrong about what is being counted. Exhaustive scan may only be wired behind a router that can abstain |
| relation router | **built and measured, zero provider calls**: `owns` merged into `acquired` at the routing layer so the probe set and its held-out split stay valid; templated upper bound 80% → **100%** with no confusions left. Abstains below a 0.15 margin. On the routed slice scan completeness is **100% against top-k's 69%**; overall completeness 50.0% → 66.7%. The residue is concentrated in the abstained slice, so **coverage, not routing, is now the binding constraint** |
| v4 run safety | **two quota-wasting defects found by rehearsal, before paying**: the probe runner had no way to answer only the development half, so the first v4 measurement would have spent all 87 held-out probes; and its rows dropped the v4 derivation, so a paid run would have produced answers indistinguishable from the model's own. Both fixed, both pinned by tests; checklist in [`QUOTA_DISCIPLINE.md`](QUOTA_DISCIPLINE.md) |
| v4.0 attempt 1 | **void by contamination, sealed as a negative result**: Gate 0 passed, but 55 of 142 candidate rows reached the grader as raw JSON because the v4 prompt never required `answer`. Headline table unreadable. On the 87 clean rows v4.0 scored 62.1% vs 51.7%, `duration` **+34.6** — a diagnosis, not a result. Cost 329 requests; no held-out probe touched. Archived at `results/archive/v4.0-attempt1/` |
| v4.0 design fault | **named**: single-variable was enforced against retrieval and proven by Gate 0, but three changes were bundled *inside* the answerer — deterministic arithmetic, operation-first prompt, timeline rendering. The v3.2 failure one layer down. The next attempt must separate them |
| v4.0 pre-registration | **registered before any provider call**: two arms differing only in `answer_policy`, development half only (142 probes, ~355 requests), four written predictions, and a Gate 0 that voids the run unless retrieval is byte-identical across arms — `tools/check_arm_invariant.py`, already verified PASS on rehearsal rows |
| v4.0 flat | **candidate selected, sealed**: separating timeline rendering attributed attempt 1's regression to it — `current_state` 87.8 → 55.1 with it, 85.7 without, exactly 89.4% both ways on unleaked rows. Deterministic arithmetic held: `duration` **+30.0**. Overall 54.9% → 61.3%, paired 22W-13L, `p=0.1755`. Stop condition not triggered |
| v4.0 leak, third pass | **root cause found and closed in code, not prompt**: 9 of 18 real leaks came from the fallback's second call inheriting a system prompt that demands structure it has no schema to parse. The leak flag itself was wrong in both directions (12 reported, 18 real, 4 false positives, 10 misses) and now measures the final answer. Raw structure can no longer reach the reader at all |
| count enumeration ceiling | **measured before v4.1 runs, and lowers what it may claim**: on the 17 count probes whose evidence was already complete, the model under-enumerated on 7 and over-enumerated on 3 — 10 of 17 wrong with every fact in front of it. Scanning cannot reach a reading failure, and the v4.1 predictions were revised downward in advance |
| v4.1 pre-registration | **baseline thresholds filled on 2026-09-06; later run needs audit closure**. Gated on the routed slice — declared in advance, because the router abstains precisely where top-k is worst and the favourable denominator must not be chosen after seeing it. Gate 0 inverts for v4.1: retrieval must be identical on *abstained* questions |
| missing_field guard | **found to be a third write-only field and wired**: defined in the schema, named in the prompt, read by nothing. It now blocks computation when a verdict names an absent operand and supplies operands anyway. The v4.0 pre-registration claimed this guard existed before it did, and says so |
| v4 data protocol | **registered**: LongMemEval-S is exhausted — 0 unused questions — so `dev100` becomes `v4-dev` under a three-decision cap, synthesis probes with SQL ground truth become the development instrument, and a fresh final set is registered as a dependency that must be acquired |
| failure taxonomy | complete: 222 failures over 63 questions from three readable pools, zero provider calls; abstention-with-source is the largest bucket in every pool at 44-50%, retrieval misses 0-6% |
| predicate vocabulary | offline proposal complete, zero provider calls: 38 relation types grouped by arity, 62.3% of memories mapped, singleton keys down 63% on the covered subset |

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

### Historical phase-boundary checks

- `load_dev_decision` now validates all 30 sealed row/usage artifacts while keeping the
  product decision limited to the three pre-registered decision arms. A five-arm
  regression test and the real dev100 decision both pass.
- At that phase boundary the offline suite had 597 passing tests, Ruff reports no violations, and core
  package line coverage remains 83%. Including one-off research scripts lowers the
  combined number to 62%; this is not treated as a reason to write low-value tests for
  retired analyses.
- `configs/v2.yaml` still enables the selected flat raw-fallback behaviour. The final
  freeze binds that config, the corrected source, the dev100 aggregate and decision,
  the registered three-arm final protocol and the completed test100 store.

## Product code

The repairs from branch commit `06b567e`, integrated into this workspace on 2026-09-12, fix the default REST/MCP write composition, uses the
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

The REST API now derives the namespace from a bearer token rather than from the request
(`api/identity.py`), refuses a namespace that does not match the credential with 403, and
offers `GET /v1/export` and `DELETE /v1/data` — a real erasure, distinct from `forget`,
which stays a soft delete because provenance is the product. Cross-tenant read, write,
raw-conversation search and erasure are each pinned by a test written as the attack.

What that does **not** amount to:

- **Bearer tokens are the weaker scheme, chosen so it could land and be tested now.**
  They do not expire, carry no claims, and cannot be revoked without an environment
  change. `principal_from_token` is the seam an OIDC/JWT verifier replaces.
- **Open mode still exists**, because the research CLI, the inspector and the offline
  suite all drive the service without credentials. With no tokens configured any caller
  may name any namespace. It is now reported by `/healthz` as
  `authenticated_access: false` rather than being an unexamined default.
- **The MCP HTTP transport has no equivalent** and must not be exposed beyond localhost.
- **Erasure is not transactional across resources.** SQLite rows go before vectors, so a
  crash between them leaves an orphan vector rather than a searchable memory whose row is
  gone — the safer direction, but not an atomic one.
- **Erasure covers the online data plane only.** Backups, if any exist, are untouched.

PostgreSQL/pgvector, OIDC/JWT, backup/restore drills, quotas, alerts and SLOs require
deployment and policy choices and remain the productization work in
[`PRODUCTIZATION_V2_PLAN.md`](PRODUCTIZATION_V2_PLAN.md). They must not be represented
as complete merely because the local prototype passes tests.

## Historical v3 execution sequence

| order | work | current state | completion standard |
|---:|---|---|---|
| 1 | Preserve the v2 conclusion | **done** | Aggregate result, configuration, protocol, hashes and exact frozen source verify independently |
| 2 | Improve v3 answer reasoning | **done; pilot gate passed** | v3 reached 72.9% versus 68.8% for v2 on the same 48 train-only questions, with all pre-registered safety/cost checks passing |
| 3 | Add multi-session and time-reasoning tests | **tune1 done; v3.1 rejected** | The 84-row comparison tied overall and regressed both target slices; exact negative result and source are archived |
| 4 | Add adaptive context | **tune2 done; v3.2 stopped on cost** | Accuracy and all safety checks improved or held, but median context was 2.52x v2 against a fixed 2x ceiling |
| 5 | Compress v3 evidence | **done** | Exact source sentences are deduplicated and allocated round-robin across sessions; observed median is 1,124 tokens (1.92x v2) |
| 6 | Run final tune42 iteration | **done; PASS** | Exactly 42 v3.3 rows retained v3.2's 73.8% accuracy and passed all 11 registered checks |
| 7 | Use sealed dev60 once | **done; one shot spent** | 360/360 rows, ledger `complete`, nine gates recorded as passing but only eight carry evidence; +8.3 points at `p=0.1797` |
| 7a | Widen the failure sample | **done, zero provider calls** | 222 failures over 63 questions; abstention-with-source 44-50% in every pool, retrieval misses 0-6% |
| 7b | Propose a controlled relation vocabulary | **done, zero provider calls** | 38 relation types by arity; 62.3% of memories mapped; singleton keys down 63% where it applies |
| 7c | Test the fallback-depth hypothesis | **done: hypothesis falsified** | 17 of 18 abstentions already had a gold turn among the three shown; raising `max_turns` would reach one question |
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

### dev60 result

The registered one-shot run completed on 2026-09-05: 360 rows, three repeats per arm,
ledger `complete`, and a hash-bound aggregate. Nine gates were recorded as passing.
Eight of them tested something: `no_new_confident_errors` compared a field that was
structurally constant, so it passed without measuring anything
([audit record](../results/audit/v3-verdict-schema-never-sent-20260906.json)). The
accuracy figures below do not read that field and are unaffected. `tools/verify_v3_dev60.py`
still prints "nine gates passed" because the verifier hashes itself into the conclusion
it protects, so its text cannot be corrected without invalidating the archive.

| | `v2-control` | `v3.3-compact` |
|---|---:|---:|
| majority accuracy | 46.7% | **55.0%** |
| mean accuracy over three runs | 49.4% | 55.6% |
| standard deviation across runs | 1.9pp | 2.5pp |
| unanimous agreement across runs | 85% | **95%** |
| median context tokens | 573 | 1,115 — **1.95x** |
| fallback trigger rate | 30.6% | 22.8% |
| selected source recall | 98.3% | 98.3% |

Every pre-declared slice improved and none regressed: temporal 44.4% to 55.6%,
multi-session 33.3% to 38.9%, knowledge-update 70.0% to 80.0%, ordinary 50.0% to
57.1%, and high-confidence-wrong stayed at zero for both arms.

Three things belong next to that table rather than under it. The paired result is
**+8.3 points at `p=0.1797`** — seven wins against two losses on sixty questions,
which is not statistically conclusive, and the gate was written to be passed by
consistency across slices rather than by that p-value. Both arms score far below
their `tune42` numbers because `dev60` is deliberately the hard slice: eighteen
temporal and eighteen multi-session questions out of sixty. And v3.3 reached the
better score while triggering the raw fallback *less* often, 22.8% against 30.6%,
so the gain is not bought with extra second passes.

**The aggregate independently corroborates the failure taxonomy without reading a
single row.** Selected source recall is 98.3% for both arms while accuracy is 55.0%
— a 43-point gap between finding the evidence and answering from it. That is the same
conclusion the 222-failure taxonomy reached on other data, now visible in a validation
aggregate the protocol permits reporting.

Per that protocol, individual `dev60` rows are not read and no parameter may be tuned
after its first provider call. The candidate is fixed. What this permits is a new
hidden final set; it does not authorise one.

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
