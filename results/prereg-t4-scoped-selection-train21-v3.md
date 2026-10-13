# Scoped t4 selection acceptance v3: full event-reference review

Registered 2026-10-10 Australia/Sydney before fresh v3 calls. No commits or model changes.

## Evidence and correction

Development-only fixed 21 previously exposed train150 cases, exact `PILOT` IDs and
manifest `results/manifests/train150.json`. No resampling, no provider seed supported,
temperature 0. The frozen manifest seed is 0. Not population accuracy or unseen final.

The v2 acceptance completed: original gold 19/21, supported 18/19, faithful gaps 2/2,
median context6374, **FAIL**. Five source-bound repairs behaved as intended; wedding
count remained wrong (five versus original gold three). Preserve v1/v2 rows/rules.

The v2 correction of the wedding auxiliary was premature: reviewing only explicit
attendance verbs missed indirect/undated USER records in non-reference sessions:

- `1d6e01e8`, 2023-10-15 07:39, turn0: wore a silver locket to an unnamed cousin's
  wedding "last month", i.e. September. It is not established whether this cousin
  is the named cousin in the August vineyard report, another cousin, or a date error.
- `81b971b8_2`, same day 10:57, turn4: sister's wedding was amazing; planned together
  and "special to be a part of it". No event date/year or explicit attendance.

The three named wedding records establish examples, not an exhaustive total of
exactly three. Forcing five also resolves identity/attendance/date uncertainties
without evidence. Restore this case as S0 with secondary S5 unsupported aggregation.
This is an explicit new protocol, not an alteration of either prior failed gate.
All 21 original-gold grades still remain required and separately reported.

## Candidate and validation

Original t4 core and accepted heldout100 aggregate remain unchanged. Experimental
`EventCountRepairRunner.from_t4`, prompt `memory-t4-scoped-selection-v2`, config
`configs/fallback.yaml`, existing train150 store/MiniLM384 index. Same answerer
`gemini-3.5-flash-lite`, judge `gemma-4-31b-it`, extractor `gemini-3.1-flash-lite`
(zero new extraction). No question IDs, reference flags or gold enter answering.

The previous scoped collection/place/ordinal/period/payment repairs are unchanged.
Additional mechanism: for a year-scoped attended-event count, inspect owned USER
event references, including indirect attendance. Repeated unnamed owners across
sessions together with undated participation trigger a source-bound clarification.
Do not reject all anaphora or merely missing an exact day; named dated controls,
future plans, assistant illustrations and other users are tested.
For this route replace the old raw excerpt block with complete category-matching
USER source records, preserve memory state, and account for the actual changed
context. Never add the entire history or hide discarded-source metadata.
The model receives the evidence and missing fields; its unmodified response and
the source-bound final output are separately saved. No citation-denial cascade.

Zero-call replay `t4-scoped-selection-train21-v3.preflight-r2.json` precedes calls.
The earlier preflight is retained: duplicate evidence caused an excessive 7997
context on the wedding item, corrected by replacing rather than appending raw.
Offline replay is composition/provenance evidence, never accuracy.

## Gates, cost and artifacts

Save all 21 fresh reader answers before grading; no answer/judge caches. Save 21
unmodified original-gold grades and three separate faithful-gap grades. Pass requires:

1. All 18 cases with answerable source evidence pass original grading.
2. All three faithful-gap grades pass: report period/overlap for `f9e8c073`; literal
   payments720 plus unresolved requested-period/year for `gpt4_731e37d7`; unresolved
   wedding identity, date/year, attendance and total for `gpt4_2f8be40d`.
3. Median actual supplied context <=6500. Whole raw or annotated reference coverage
   is diagnostic only, not the acceptance criterion.

The fixed enriched set detects these concrete cases; no statistical improvement
claim will be inferred. Failure blocks full regression and summary calls. Success
allows a **new** full-regression registration with this candidate; the unexecuted
v1 full-regression protocol based on the failed v2 acceptance is withdrawn.

21 readers, at most42 nominal generations /126 shared client attempts; 21+3 grades,
at most288 judge attempts with three durable recoveries and 180-second cooldown.
Client API/transport retries2 each (three shared attempts per invocation). About
150k reader-input tokens, plus grading. Failed calls consume quota and are recorded.
No quota reset bypass or changing rules after outcomes. Exhaustion is INCOMPLETE.

The new driver freezes source/config/data, prior failed gates, zero-call replay,
this registration, git SHA/Python/models and literal source snapshots before calls.
Resume rejects changed identities. All outputs use `t4-scoped-selection-train21-v3`.
Original and auxiliary grading remain separate, with exact reader SHA bindings.
