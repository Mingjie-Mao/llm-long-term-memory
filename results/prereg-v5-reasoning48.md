# v5.0 parallel source-local evidence — pre-registration

Date: 2026-09-22 (Australia/Sydney)

## Classification

This is a **DEVELOPMENT mechanism experiment**. It is not an unseen final evaluation.

LongMemEval-S is exhausted: all 500 questions are partitioned across five disjoint sets
with none remaining (`results/hidden-set-status.md`). The 48 questions used here are the
v2c reasoning-48 set, whose per-question rows were read in full while attributing v2c's
eleven failures (`results/analysis/v5-offline-gate.md`). They are development evidence
and nothing measured here may be reported as a benchmark estimate or compared with the
frozen 72% on `test100`, which used a different protocol.

Every question in the set is included. No subset is selected on the failures, because
selecting the questions a mechanism was designed from and then scoring it on them
produces a number with no inferential content — the v2b gate already recorded that its
post-selected discordant repeat had no valid p-value.

## Question

Does attaching verbatim turns from the sessions retrieval already selected, **before**
the first answer call rather than after the answerer refuses, produce a net improvement
over v2c at an acceptable context cost?

## Why this and not something else

`results/analysis/v5-offline-gate.md` attributes all eleven v2c failures by hand:

| cause | n |
|---|---:|
| `reader` — the evidence was in the selected context and the answer still failed | 6 |
| `local_raw` — a sentence the extractor dropped, in raw turns of a selected session | 3 |
| `repack` — the fact was in the store and not selected | 1 |
| `retrieval` — the gold session was never found | 1 |

Raw text reached the reader in 10 of 48 questions; the other 38 saw structured memory
only. The three `local_raw` failures share one shape, and it is the shape this mechanism
addresses:

- `gpt4_7a0daae1` — "I just received my new tennis racket today" is a user turn in a
  session that *was* selected; only the assistant's recommendations were extracted.
- `a4996e51` — "up to 50 hours per week" is an assistant turn; the surviving memory says
  40–45 hours and was not selected.
- `58ef2f1c` — the raw turn says the dinner was "back on Valentine's Day"; the structured
  fact kept only "February 2023".

**v5.1 (whole-session repacking) is not registered here.** The gate reaches one failure
for it, and that one is compounded by a cheaper defect: for `gpt4_74aed68e` the dated
"February 14" memory was in the store and unselected, while an undated restatement
carried its session's date as `event_time`. Splitting event time from observation time
may fix it without touching the context shape, so repacking is not bought on this
evidence.

## Arms

All three read the same completed `stores/v2c-reasoning48.db`. No ingestion runs, no
store is written, and nothing under `results/frozen/` or `results/sealed/` is touched.

| arm | variant | raw windows | label |
|---|---|---|---|
| A — baseline | `two_stage_v2c` | conditional only | reuses the committed `two_stage_v2c.reasoning48-v2c8.jsonl`; **not re-run** |
| B — fixed | `two_stage_v5_fixed` | 3 for every question | `v5-reasoning48-fixed` |
| C — planned | `two_stage_v5_planned` | 2 direct / 2 preference / 3 temporal / 3 current-state / 6 aggregation | `v5-reasoning48-planned` |

Arm A is reused rather than repeated. Re-running it would spend 96 calls to re-measure
an answerer whose run-to-run variance is already on record, and the comparison is paired
per question either way. The consequence is stated in **Limitations**.

B spends exactly what the conditional path spends (`fallback.max_turns = 3`), so the
only difference from A is *when* the turns arrive. C's budget is measured, not chosen:
replaying the three failures against the committed store showed the tennis-racket turn
is reachable at two windows, the Valentine's Day turn at three, and the 50-hours turn
not until six.

Everything else is held at v2c: same store, same retrieval weights, same `top_k = 20`,
same answer policy (`v2c`), same prompts, same models, same judge. The conditional
fallback stays enabled in all three arms.

## Offline pre-verification

Run before any paid call, against the committed store, with no model calls:

| question | arm B (3 windows) | arm C | rendered characters |
|---|---|---|---:|
| `gpt4_7a0daae1` | recovered | recovered | 832 at 2 windows |
| `58ef2f1c` | recovered | recovered | 1,495 at 3 windows |
| `a4996e51` | **not** recovered | recovered | 2,523 at 6 windows |

So B is predicted to reach two of the three and C all three. If the paid run does not
show the recovered sentences in context, the run is void rather than negative — the
mechanism would not have been exercised.

## Configuration

- Config: `configs/v5.yaml` (copy of `configs/v2c.yaml` with a new name and description;
  retrieval, hydration, fallback and quota settings identical).
- Store: `stores/v2c-reasoning48.db`, read-only.
- Models: extractor `gemini-3.1-flash-lite` (not used here), answerer
  `gemini-3.5-flash-lite`, judge `gemma-4-31b-it`, embedder `all-MiniLM-L6-v2`.
- Answer prompt version: `memory-aware-v2c.8`. Judge prompt version: `lme-type-aware-v2`.
- Dataset: LongMemEval-S, the 48 question ids in
  `results/raw/two_stage_v2c.reasoning48-v2c8.jsonl`, in that order.
- One run per arm, seedless; the answerer is called at temperature 0.

## Estimated cost

96 answerer calls (48 × 2 arms) plus second passes where the answerer still requests
source, and 96 judge calls. Against the observed free-tier limits — 500 rpd on
`gemini-3.5-flash-lite`, 1,500 rpd on `gemma-4-31b-it` — this fits inside one quota day.

## Primary endpoint and gates

Primary: **paired win/loss against arm A over all 48 questions**, per arm.

Promote an arm to a wider measurement only if all hold:

1. **net positive** — wins exceed losses against A;
2. **no control regression** — no question correct in A and wrong in the candidate whose
   attribution in the offline gate is `reader`; the mechanism adds evidence and must not
   be seen to break answers it was not addressing;
3. **the mechanism fired** — every question records `parallel_raw_turns > 0` unless its
   selected memories name no source session;
4. **the sentences arrived** — the three `local_raw` questions contain their recovered
   turn in context, as the offline pre-verification predicts for that arm;
5. **context cost** — median context tokens ≤ 2,000, against v2c's 636. A mechanism that
   reaches naive-RAG context sizes has given back what the project is measured on;
6. **frozen and sealed artifacts unmodified.**

Between B and C: prefer B unless C wins strictly more questions. A larger budget that
buys nothing is a cost, not a tie.

## Stopping rule

If neither arm is net positive, v5.0 is recorded as a negative result and raw evidence
stays conditional. No re-tuning of the budget on these 48 questions is permitted
afterwards — a repaired candidate needs a new label and a new question set.

## Limitations, stated before the numbers

- **Development set, read in full.** The failures were attributed by hand before this was
  written. Nothing here is an unseen estimate.
- **Single run per arm, and a reused baseline.** The answerer is stochastic at
  temperature 0 in practice; a one-run paired comparison cannot separate a small
  mechanism effect from answerer variance, and reusing A's committed rows means the
  baseline was measured on a different day from the candidates. No significance is
  claimed, and none will be reported.
- **Six of eleven failures are out of reach by construction.** The ceiling for this
  mechanism on this set is small, and the dominant failure bucket is the reader.
- `58ef2f1c` needs the reader to know that Valentine's Day is February 14th even once the
  turn is in context, so recovering the turn may not convert to a correct answer.
