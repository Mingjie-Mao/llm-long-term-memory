# v2e reasoning-48 decision

> **DEVELOPMENT RESULT.** These 48 questions have been read in full, twice. Nothing
> here is a benchmark estimate, and it does not belong on a version curve with the
> frozen 72% on `test100`.

## Decision

**STOP: do not promote. Keep the code, default off, and record this as a negative
result.**

The mechanism's correctness claim stands on its own — an undated fact should not render
the day it was mentioned as the day it became true — and the code stays for that reason.
What this run establishes is that the claim does not convert into measurable QA gain at
this scale, and that this question set cannot fairly judge it.

## Evidence

| Endpoint | Result |
|---|---|
| Baseline `two_stage_v2c` (committed rows, not re-run) | 37/48 |
| Candidate `two_stage_v2e` | 37/48 |
| Paired | 5 wins / 5 losses / 38 ties, net **+0** |
| Exact McNemar | p = 1.0000 — shape only; one run per arm |
| Median context | 634 → 692 tokens (**+9.0%**) |
| Marker reached the context | 48/48 |
| `temporal-reasoning` | 9 → 8 |

| Registered gate | Result |
|---|---|
| `net_positive` | **FAIL** |
| `no_temporal_regression` | **FAIL** — `6e984301`, `982b5123` |
| `marker_reached_every_temporal_question` | PASS — 12/12 |
| `context_within_10_percent` | PASS — +9.0%, against a 10% limit |

## The result is a null, not a harm

Splitting the ten moved questions by whether the mechanism could reach them at all:

| | wins | losses |
|---|---:|---:|
| `temporal-reasoning` — the mechanism applies | 1 | 2 |
| every other type — it does not | 4 | 3 |

**Seven of the ten moves are in question types a date label cannot affect.** The
clearest case is `e48988bc`, which v2e answers correctly and v2c does not: the v2d
retrieval gate established that its gold session was never retrieved at all, and
changing how a date is rendered cannot conjure a session that retrieval never found.
The two abstention questions that flipped were attributed to the reader in
`results/analysis/v5-offline-gate.md` — answering where they should decline — which is
also outside this mechanism.

So the movement is consistent with answerer run-to-run variance, and the mechanism
produced no effect this run can detect. The pre-registration anticipated a possible
net loss, on the grounds that removing a cue the reader had been guessing from could
turn lucky guesses into abstentions. That is not what happened: it neither lost
systematically nor gained.

## Why this set cannot judge it

30 of the 48 questions carry undated evidence in their selected context, but only 12
are `temporal-reasoning`. The mechanism's reach and the questions able to register it
differ by more than a factor of two, so most of where it applies is invisible to the
score. Judging it needs a set enriched for temporal questions, not a general one.

## Provenance

- The first attempt stopped at question 42 on a provider `503 UNAVAILABLE`, and the
  harness resumed to 48 from its own rows. No question was answered twice.
- 158 provider requests, 48 of them failed and retried: the answerer failed 2 of 64,
  the judge 46 of 94, with a p50 of 25s and a max of 62s. Every question ended with
  exactly one successful judgement, but the run was made while the provider was
  unstable and that is part of what the variance above may be.
- Store: `stores/v2e-reasoning48.db`, a copy of the v2c store migrated and backfilled
  by `tools/backfill_event_time.py`. The committed `stores/v2c-reasoning48.db` still
  has no `observed_at` column, which is the proof it was never opened for writing.

## A defect found in the instrument

`tools/v2e_gate.py` returned 1 from `main()` on STOP and then discarded it, so the
process exited 0 while the report said the gate had closed. A gate that exits 0 on
STOP is one that anything reading the status rather than the prose walks straight
through. Fixed to `sys.exit(main())`. This is the second time in this line of work that
an exit code has misreported a failure — the first was a `| tail` pipeline reporting 0
for a crashed run.

## What must not happen next

No re-tuning of the marker, the wording, or the budget on these 48 questions. They are
exhausted twice over. A repaired candidate needs a new label and a different question
set, and the honest next measurement for this mechanism is a temporally enriched set
that does not yet exist.

## What is kept

- `Memory.event_time` / `observed_at` / `occurred_at` / `event_time_is_stated` — the
  representation, which is independently correct and which every consumer now reads
  through `occurred_at` with no behaviour change.
- `ingest/event_time.py`, including the anchor guard that stopped
  "about a month before May 21, 2023" being read as an event on 21 May.
- `tools/backfill_event_time.py`, which is what makes the signal exist on any store
  written before the split.
- `two_stage_v2e`, off by default.
