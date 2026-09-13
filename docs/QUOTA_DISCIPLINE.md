# Before a run that costs quota

Quota is the binding constraint on this project — not compute, not ideas. `dev60` cost
826 requests and `test100` 647; the daily answerer budget is 500. So the question before
any paid run is not "is this a good experiment" but "what would this run tell me that
something free would not, and what would waste it".

This page is the checklist, and every line on it exists because something already went
wrong.

## What has actually been wasted

| what happened | cost | how it would be caught now |
|---|---|---|
| The dev60 run hung on its first request for 31 minutes and produced **zero rows** — the provider client had no timeout | a quota day and an aborted freeze | `test_client_timeout.py`; a socket that accepts and never answers now raises |
| The v3 answerer was sent the base verdict schema while being parsed with the subclass, so `confidence` was its default on **100% of rows in every v3 phase** | a registered `dev60` gate that could not fail | `test_verdict_schema_is_sent.py`; `assert_gate_input_observed` refuses a gate whose input is pinned at its default |
| The probe grader read a count reply's *first* integer, and looked for a bare `A`/`B` in prose that never contained one — **36 of 60** comparison replies scored unparseable | a whole reading of a paid run, rebuilt offline | verdicts are a derived column; `--regrade` re-scores saved replies for free |
| A `max_turns` fix was scoped from a plausible story | would have been a config change plus an eval run | `fallback_depth_probe.py` falsified it offline: it reaches **1 of 18** questions |

The pattern is not carelessness. Every one of these looked finished from the inside, and
none of them raised.

## The checklist

**1. Run the free diagnosis first.** `tools/failure_taxonomy.py` reads rows already on
disk. It decides which layer the next change belongs in, and it has twice changed the
answer. The `heldout100` rows sat unread for fifteen days while ~2,094 requests went into
improving a layer that was not the bottleneck.

**2. Try to kill the hypothesis offline.** Retrieval is deterministic given the store, so
most retrieval claims can be replayed for nothing —
`fallback_depth_probe.py`, `relation_routing_probe.py` and `routed_scan_gain.py` are all
that shape. A hypothesis that survives a free attempt to falsify it is worth paying for;
one that does not was going to fail anyway.

**3. Rehearse the whole path with a fake provider.** `--dry-run` exists for this, and the
canned reply must be shaped for *the policy under test* — a base-shaped reply rehearses
nothing that the new policy changed, passes, and lets the paid run fail on its first
probe. The v4 rehearsal found two defects this way: the runner could not restrict itself
to the development half, and its rows dropped the derivation.

**4. Check what the run will write, not just that it runs.** A row missing the field the
conclusion depends on is a full-price run with no conclusion in it.

**5. Name the half.** `--half development` is the default because the held-out probes are
spent by whatever run answers them, and a default of `all` would have spent them on the
first development measurement.

**6. Make grading derivable.** Model output is the expensive artifact. Verdicts, strata
and recall flags are recomputed from saved replies, so a grading defect costs a rerun of
a script rather than a rerun of the quota.

**7. Checkpoint per item and resume by id.** Every paid run so far has been interrupted —
by quota, by DNS, by a 503 — and none has lost more than the item in flight.

**8. Know the arithmetic before starting.** 500 answerer requests a day, and the fallback
pushes real consumption to about 1.34 requests per row. A 360-row run is two quota days
and must be planned as one.

## What is worth paying for

Only a question that a free method cannot answer. So far that has been exactly two kinds:
what an actual model does with an actual context, and whether a frozen candidate beats
its control on a sealed set. Everything else on this project — every failure taxonomy,
every retrieval ceiling, every routing measurement, the entire predicate vocabulary
analysis — has been answerable for nothing.
