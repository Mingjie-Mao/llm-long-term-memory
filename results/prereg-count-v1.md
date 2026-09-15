# Query-time count over a reviewed source pool — pre-registration

> **Paused 2026-09-14, before any provider call and before any human verdict.** The project
> owner stopped this line rather than finish it. Every reason below was measured at zero
> cost, and no run of this plan could have changed any of them.
>
> - **It cannot reach a decision.** 18 of the packet's 40 entity questions carry a flag, and
>   `count_review.py` refuses to accept a flagged question, so a frozen set could hold at
>   most 22 questions in 20 clusters — 11 of them with every proposed member found whole in
>   the user's own words. An exact paired test over 20 clusters needs a split as lopsided as
>   8–0 or 15–5, and this plan already says no promotion decision follows.
> - **It does not measure the dominant loss.** A question's pool is the turn before and after
>   each extracted memory's anchor, so a fact extraction dropped is out of scope, and so is
>   retrieval. On `test100` v2 trails full context by 14 questions; reconstructing per-type
>   counts from the aggregate's per-type accuracy (they reproduce all three arms' totals),
>   multi-session — the category counting belongs to — accounts for 2 of them.
> - **The comparison favours control by construction.** Control counts the generator's
>   proposed list, not the production answerer's output, and the only questions that can be
>   accepted are those whose generator evidence passed the relation table.
> - **The paid path does not run.** `run_count_candidate.py --execute` fails at import:
>   `load_config` does not exist. Past that, `GeminiClient(config)` has the wrong arguments;
>   `Completion` has no `usage` field, so tokens would be recorded as `None`; the default
>   model `gemini-2.5-flash` is the one D12 records as withdrawn, not the pinned answerer;
>   rows are written only at the end, with no checkpoint and no usage ledger; there are no
>   repeats; and neither the refusal stop nor the clustered test is implemented. It fails
>   before any request, so nothing was spent.
>
> What stays: the policy, the packet builder, the review page and
> `runtime/evidence_count.py`. The mechanism — quote each member from a user turn, let code
> count, refuse when the pool is uncertain — is carried forward as a candidate to measure on
> an external benchmark whose gold this project did not derive
> ([roadmap](../docs/PROJECT_REPORT.zh-CN.md#十一后续改进)). Resuming this plan needs the paid path fixed, the sections
> below amended before any run, and a fresh approval.

**Registered 2026-09-14, before any provider call on this candidate.** The mechanism,
the review instrument and the dry run are built; zero provider calls have been made
([`review/count-v1/`](review/count-v1/)). Nothing below may be edited after the run starts.

This measures counting over a **bounded source pool** attached to each question, not
end-to-end retrieval over a whole history. No number produced here may be quoted beside a
LongMemEval accuracy figure, and none of it is an independent test result.

## Why the ruler is being fixed before the method

The old generator's members fail structurally before any model reads them. Over the 40
reviewable entity questions and their 182 proposed members
([`review/count-v1/stage-decomposition.json`](review/count-v1/stage-decomposition.json),
zero provider calls):

| where a proposed member stops | members | questions blocked first here |
|---|---:|---:|
| carried only by a predicate the relation table does not approve | 43 | 14 |
| label only partly present in the user's own turn | 18 | 13 |
| named only in assistant-authored text | 10 | 2 |
| named by nobody in the supplied pool | 5 | 0 |
| quoted whole, under an approved predicate | 106 | — |

Two limits bound that table: proposed members are produced *from* extracted memories, so
a fact extraction dropped entirely cannot appear in it at all; and a fully supported
member can still be the wrong answer. Neither the table nor this run measures extraction
loss.

## Gold

The gold is the frozen human review, and only that:
`count_review.py build` → local review page → `count_review.py freeze`. Freezing refuses
unless a named human approved the rule set, every reviewable item carries a verdict and a
reason, each accepted member cites a verbatim span of a supplied source, and the evidence
store hashes to what the packet was built from. Model output never selects gold.

**Not in the pool.** 33 items are quarantined before the packet is written: 27 whose
namespace is in the sealed v3 development manifest, 6 more that share an external session
with one. Their text is never rendered.

## The two arms

| arm | what answers | differs by |
|---|---|---|
| control | the existing generator's member list, counted | — |
| candidate | `runtime.evidence_count`: review every supplied source once, quote each member from a user turn, code counts, refuse when the pool is uncertain | one change |

Same questions, same source pool, same model, same temperature (0.0), same budget. The
candidate may refuse; a refusal is never scored as correct.

## Registered before the run

- **Primary**: exact-count agreement with the frozen human answer, candidate vs control,
  paired over the same questions.
- **Secondary**: member precision and recall against the reviewed member lists; refusal
  rate; total input and output tokens; wall-clock latency.
- **Clustering**: questions sharing a source session are one unit. 40 reviewable entity
  questions currently fall into 41 clusters across the whole packet; the frozen set's own
  cluster count is recorded at freeze time and is the n used for any paired test.
- **Analysis**: paired exact test over clusters, two-sided, α = 0.05. With an n this
  small the result is a development signal. **No promotion decision follows from it**, and
  no held-out or sealed set is opened by this run.
- **Stop**: if the candidate refuses more than half the questions, report the refusal
  analysis and stop rather than tuning the prompt against this set.

## Cost control

`run_count_candidate.py` dry-runs by default: it builds every prompt, reports the input
size, and makes zero provider calls. `--execute` refuses to start unless `COUNT_PREREG`
names this plan. The dry-run figure is recorded in the run report before any spend.
