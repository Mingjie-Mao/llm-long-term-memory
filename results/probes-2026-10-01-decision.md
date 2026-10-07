# Two probes, 2026-10-01 — decisions

Both registered before any call (`results/prereg-strong-answerer-probe-v1.md`,
`results/prereg-preference-extraction-probe-v1.md`); the registrations and evaluation
scripts were hash-checked unchanged before running. Diagnostics, not accuracy figures.

## Stronger answerer on the 9 reasoning errors — **real lever** (model effect +5)

| arm | answerer (verified in the usage log) | fixed of 9 |
|---|---|---:|
| lite rerun (noise control) | `gemini-3.5-flash-lite` | **0** |
| strong | `gemini-3.6-flash` | **5** |

Read by hand (train150 permits it): the strong model found "Target" where the lite one
said it had no record although the line was in context; counted two charity events with
their dates where lite found "no specific" record; computed one week between a Friday
purchase and the next Friday's delivery where lite said zero. One fix is arguable —
`f9e8c073`, where both answers mention three and then five sessions and only the strong
one was graded right; without it the effect is +4, still at the registered threshold.

Registered next step: a full comparison of the answering model. On the free tier
`gemini-3.6-flash` ran 19 answer calls today including 6 provider failures; a 100-question
run needs roughly 110–135 answer calls per arm, so about a week per arm at the probed
20-a-day limit, or a paid tier. That is a budget decision. Not visible here: whether the
stronger model gets wrong some of what flash-lite answers right.

Cost: lite 13 answer calls (1 failed), strong 19 (6 failed), judge 49 (31 failed and
retried on provider `500 INTERNAL`).

## Preference shown in passing — **record, do not pursue**

| extractor | preference memory on a gold turn | any memory on a gold turn | memories / session |
|---|---:|---:|---:|
| plain | 3 / 24 | 14 / 24 | 6.3 |
| observer instruction | **6 / 24** (+3; needed +6) | 21 / 24 (+7) | 6.2 |

Rule 1 fails (+3 against +6); rule 2 passes (no volume growth). Not pursued. Recorded,
not acted on: the instruction raised *any* memory on a gold turn from 14 to 21 with no
more memories — an unregistered secondary observation that would need its own
registration. The observer's six preference lines on gold turns were read; none invents
a preference the conversation lacks ("prefers winding down by 9:30 pm" for "I'm usually
in bed by 9:30").

Cost: 12 extractor calls.

## Regression probe, 10 of flash-lite's correct answers — **low risk by the registered rule**

Registered in `results/prereg-strong-answerer-regression-probe-v1.md` (revised before any
call from 20 questions to 10; the 20-question draw was never run). Registration, script
and manifest hash-checked unchanged after the run.

| type | sampled | strong got wrong |
|---|---:|---:|
| temporal-reasoning | 2 | 0 |
| multi-session | 2 | 0 |
| knowledge-update | 2 | 0 |
| single-session-preference | 2 | **1** |
| single-session-user (numeric) | 1 | 0 |
| single-session-assistant (numeric) | 1 | 0 |

**R = 1 / 10** — the registered reading is *regression risk is low; proceed to the full
strong vs flash-lite comparison*. Flash-lite's own run-to-run flips would put 0.59 here.

The one regression is real, not grading noise. `a89d7624` asks what to do in Denver; the
gold wants answers built on the user's own Denver history (meeting Brandon Flowers, live
music). Flash-lite opened with that history; `gemini-3.6-flash` listed nearly the same
venues as a generic guide and never mentioned it, and was graded wrong for exactly that.
The pattern to watch in a full comparison is the stronger model answering preference
questions generically.

Projected net on train150, as registered: conservative **−7.7** (one regression in ten
extrapolated to 127 correct answers, against the +5 fixed among the 9 probed errors, the 14
retrieval-limited errors credited nothing) and noise-adjusted **−0.2**. **Neither shows a
net benefit**: ten questions rule out a large regression and cannot price a small one —
one in ten is consistent with a true rate from well under 1% to over 40%. Whether the
stronger answerer is a net gain is the full comparison's question.

Cost: 21 `gemini-3.6-flash` answer calls (9 failed and retried on provider errors), 23
judge calls (13 failed and retried). Nothing committed.
