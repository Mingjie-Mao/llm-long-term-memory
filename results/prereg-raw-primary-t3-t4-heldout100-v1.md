# t3 and t4 against v1 — heldout100, three runs per arm — pre-registration

Date: 2026-10-07 (Australia/Sydney). Written before any answer or judge call of this
experiment. Replaces the unrun registrations `results/prereg-raw-primary-time-notes-v1.md`
(t1) and `results/prereg-raw-primary-count-latest-v1.md` (t2), withdrawn below.

## Classification

**A development comparison on an exposed set, recorded as regression evidence.**
`heldout100` decided v1 in aggregate (`results/raw-primary-heldout100-decision.md`).
Before this registration and without any answer, it was used for: an aggregate count of
relative-date phrases in its temporal questions' gold turns (20 of 27); the stub-client
context precheck; the all-gold reach of the candidate packing (86.0% → 93.5%); and —
disclosed because it touched a design choice — the reach of 3 heldout100 questions was
among those compared when the time-window packing was picked from four variants
(`research/EXPERIMENT_INDEX.md`, 2026-10-07). No answer or failure of it was read. Not
an unseen result, not an accuracy figure; LongMemEval-S has no unseen question left.

## Arms

| arm | variant | changes from v1 |
|---|---|---|
| v1 | `two_stage_raw_primary` | — |
| t3 | `two_stage_raw_primary_t3` | answering: headers dated to the question; relative dates in the user's turns resolved (`time_notes`); latest value wins; counts and questions about dates, durations or order answered from notes, 1,024 output tokens on those (`memory-aware-v2-clt`) |
| t4 | `two_stage_raw_primary_t4` | t3, plus retrieval: the user's turns packed first (75% of the 4,000-token budget, `prefer_user`); the period a question names searched by time, the user's turns in it packed first up to half the budget (`time_window`) |

## Evidence before this run

- Probes on train150 out-of-design failures (`results/t-probe-v1-decision.md`,
  `results/analysis/t-probe-v2.md`), one run each, against a v1 rerun: temporal failures
  fixed 0 → 5 (t3), count/knowledge-update failures 2 → 5, controls lost 1 of 12. t1's
  date notes alone moved its target by 1; the notes path is the active part.
- Retrieval reach at 4,000 tokens (`tools/candidate_coverage_offline.py`; zero calls):
  v1 → t4 packing, train150 78.1% → 90.4%, dev100 83.5% → 92.8%, heldout100 86.0% →
  93.5%. Code and the offline tool pick identical turns on all 350 questions.
- Stub-client context (heldout100): v1 median 5,587; t3 5,735 (p90 5,898); t4 6,142
  (p90 6,290, max 6,444). t4 exceeds the 6,000 used before: more user turns carry date
  notes. The limit here is **6,500**, set before any call with about 350 tokens of margin
  over the measured median.
- A crash found by the t4 precheck ("6,000 years ago" has no calendar date) was fixed in
  `time_notes.resolve` before registration, with a test; it changes nothing that did
  not crash, so the t3 probe's rows stand.
- Full test suite at registration: 2,420 passed, 0 failed, 0 skipped.

## Decision rules

Arms are compared on each question's mean correctness over three runs.

**t4 passes** if all hold (`tools/raw_primary_gate_t3t4.py`):

1. t4 − v1 mean net **≥ +4.0**;
2. no question type's t4 − v1 mean net below **−2.0**;
3. median context of t4 over its three runs **≤ 6,500** tokens.

Registered readings, not deciding: **answering contributes** if t3 − v1 ≥ +2.0;
**retrieval contributes** if t4 − t3 ≥ +2.0. Reported: per-run scores, per-type nets
for both comparisons, median output tokens, fallback second calls, t3's median context,
an exact sign test over questions whose mean moved (no significance claimed).

A pass makes t4 the candidate for a final measurement on data not yet used (BEAM, or
another set), which needs its own registration. A failure is recorded; nothing is tuned
on heldout100 against these runs.

## Cost and schedule

Nine runs, 900 answers: about 1,050 answerer calls (notes questions are one call; some
take the fallback second call) and 900 judge calls — about three answerer quota days
(500/day; 83 left today). Order: v1, t3, t4 for rep1, then rep2, then rep3
(`tools/run_t3t4_heldout.py`, which resumes saved rows and waits for quota resets).
If all nine are not complete by **2026-10-14 00:00 Sydney**, the experiment is recorded
as incomplete.

## Registration hashes (sha256, first 16)

| file | sha256 |
|---|---|
| `src/llm_long_term_memory/retrieve/time_notes.py` | `24e3f1ae5c649bbc` |
| `src/llm_long_term_memory/retrieve/time_window.py` | `a7bacba8a40e5b92` |
| `src/llm_long_term_memory/retrieve/excerpts.py` | `28d4420aa9c9ba71` |
| `src/llm_long_term_memory/evaluation/runners/memory.py` | `ca9b68ddb69698ca` |
| `src/llm_long_term_memory/cli.py` | `662e3c27853ca349` |
| `src/llm_long_term_memory/answering.py` | `3d932a89cd017ecb` |
| `configs/fallback.yaml` | `397c0c59908079c9` |
| `tools/raw_primary_gate_t3t4.py` | `1058ebb933e40db6` |
| `tools/run_t3t4_heldout.py` | `ba8d95c5c48a1661` |
| `stores/heldout100.db` | `c8db4c2c0cec5f0a` |
| `results/manifests/heldout100.json` | `19f833d8d9457706` |

## Reproduction

```bash
python tools/run_t3t4_heldout.py --execute
python tools/raw_primary_gate_t3t4.py
```
