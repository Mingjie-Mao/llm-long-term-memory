# P10-final — frozen for the held-out run

Nothing in this directory is regenerated. `freeze.json` records the system as it
stood when development on `dev50` stopped, and `scripts/freeze.py` re-derives every
hash in it:

```bash
python scripts/freeze.py            # exits non-zero if anything moved
```

The check earns its place. This project has already shipped a store whose `meta`
claimed one extractor while 68% of its rows came from another, so a freeze that is
only a sentence in a README is not a freeze. Changing `fallback.max_turns` from 3
to 5 makes the check fail with that exact line.

The freeze is defined by `freeze.json`, not by the commit that introduced it — so
rewriting history cannot quietly detach it from what it describes.

## What is frozen

| | |
|---|---|
| store | `two-stage-p10` — 6,233 memories, 6,202 active, 50 namespaces, 2,348 sessions, 24,590 turns, **0 rows with `scope NULL`** |
| extractor | `two-stage-p10-v2`, `gemini-3.1-flash-lite`, 15 sessions/request, dedup 0.9200 |
| answerer | `memory-aware-v2` |
| judge | `lme-type-aware-v2` |
| retrieval | `top_k` 20, `candidate_limit` 50, rerank off, temporal on, weights `semantic 1.0` and **the other four at 0.0** |
| fallback | on, `max_turns` 3, `max_chars` 2400 |
| held-out | `heldout100.json`, 100 questions, **never run** |

Prompt *texts* are hashed as well as their version strings, because a version
string is edited by hand and the text is what the model actually sees.
`fallback.py` is hashed whole, since its level-selection logic was rewritten
during development and a silent revert would not show up in the config.

Not covered: results, scripts, documentation. Those can be corrected afterwards
without changing an answer.

## The development result this freezes

`dev50`, the same fifty questions used for every decision below:

| | | |
|---|---:|---:|
| final accuracy | 36/50 | **72.0%** |
| answered from structured memory alone | 27/50 | **54.0%** |
| rescued by the raw archive | 9/50 | **18.0%** |

Memory alone ties `naive_rag` at 54.0% and sits below `full_context` at 56.0%. The
archive is what puts the product ahead of both. The 72% should not be quoted
without that split.

## Why the freeze is here rather than after more work

`dev50` is exhausted. It was used to iterate prompts and tune gates, then to
choose the fallback design, then to locate every remaining failure — and finally
to decide which modules *not* to build. That last use is the deepest form of
adaptation: six planned modules were cancelled on evidence from these fifty
questions.

| module | ceiling on dev50 | decision |
|---|---|---|
| hybrid retrieval | **0 questions** | not built |
| typed value schema | 0.14% of memories affected | not built |
| memory-utility filtering | no measured accuracy effect | not built |
| adaptive extraction | no signal to route retries on | not buildable |
| reasoning router | 1 question of 6 | not built |
| smaller extraction batches | 4 of 9, at 12x the requests | not adopted |

Every one of those is a judgement made by looking at the answers. Continuing to
develop against `dev50` would keep producing decisions with no way to tell
whether they generalise, so the honest move is to stop and open the held-out set
once.

## The rule

After this point, `dev50` is not evidence for a change. If something here has to
move, the freeze record moves with it and the reason is written down — and the
held-out result that follows describes the new system, not this one.

`heldout100.json` has never been run and its questions have never been read. It is
the only measurement left that these decisions have not already been fitted to.
