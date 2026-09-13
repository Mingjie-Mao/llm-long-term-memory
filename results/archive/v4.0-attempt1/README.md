# v4.0 attempt 1 — void by contamination, and what it still established

**Run 2026-09-06. 329 answerer requests. 142 development probes per arm.**
Rows: [`probes.v3.3-dev.jsonl`](probes.v3.3-dev.jsonl),
[`probes.v4.0-dev.jsonl`](probes.v4.0-dev.jsonl).
Registered as [`prereg-v4-synthesis.md`](../../prereg-v4-synthesis.md).

## Gate 0 passed

`tools/check_arm_invariant.py` reported retrieval identical across both arms on all 142
probes — `retrieved_ids`, `evidence_found`, `evidence_needed`, `context_complete`. Any
difference below is attributable to the answerer alone. That much of the design worked.

## The headline table, which should not be read

| operation | n | v3.3 | v4.0 | Δ |
|---|---:|---:|---:|---:|
| `duration` | 30 | 40.0% | 66.7% | **+26.7** |
| `count` · context complete | 17 | 17.6% | 35.3% | +17.6 |
| `count` · context incomplete | 13 | 15.4% | 7.7% | −7.7 |
| `comparison` | 33 | 54.5% | 24.2% | **−30.3** |
| `current_state` | 49 | 87.8% | 55.1% | **−32.7** |

Abstention rose from 0.7% to 5.6%. Three of four registered predictions failed.

## Why it should not be read

**55 of 142 candidate rows reached the grader as raw JSON.** The control leaked none.

    gold  San Francisco
    v3.3  The user currently lives in San Francisco.
    v4.0  ```json
          { "operation": "current_state", "missing_field": "user's home city" }
          ```

The v4 prompt described the operand fields in detail and never said `answer` was
mandatory. For operations that need no computation the model filled `operation` and left
`answer` empty; the runner then fell back to the raw completion, which under a structured
schema is the structure itself.

| operation | leaked rows |
|---|---:|
| `comparison` | **31 / 33** |
| `current_state` | 16 / 49 |
| `duration` | 4 / 30 |
| `count` | 4 / 30 |
| total | **55 / 142** |

`comparison` was hit hardest for a reason that is a design fault, not chance: `compute`
deliberately returns no sentence for it — knowing which date is earlier does not say
which described event it belongs to — so `answer` was its only possible source of a
reply. Code and model each relied on the other.

## What survives, on the 87 uncontaminated rows

| | n | v3.3 | v4.0 | Δ |
|---|---:|---:|---:|---:|
| **all clean rows** | 87 | 51.7% | **62.1%** | **+10.4** |
| `duration` | 26 | 42.3% | **76.9%** | **+34.6** |
| `count` | 26 | 19.2% | 26.9% | +7.7 |
| `current_state` | 33 | 84.8% | 78.8% | −6.1 |
| `comparison` | 2 | — | — | sample destroyed |

**The arithmetic hypothesis held where it could be measured.** `duration` +34.6 points is
what "the model names the dates, Python subtracts them" is supposed to produce.

This is a subgroup defined by an outcome of the run, so it is a diagnosis and not a
result. It says the mechanism is worth another attempt; it does not say by how much.

## The mistake, named

v4.0 was registered as single-variable against retrieval, and Gate 0 proves it was. But
**inside** the answerer it changed three things at once — deterministic arithmetic,
an operation-first prompt, and timeline rendering for supersession chains — and when the
result moved in two directions there was no way to attribute either.

That is the v3.2 failure one layer down. v3.2 moved accuracy and context together and had
to be reported `STOP` because nothing said which change did what. Gate 0 was written to
stop that happening across layers and did. Nothing was written to stop it happening
within one.

## What the 329 requests bought

1. **Deterministic arithmetic works** — `duration` +34.6 points on clean rows.
2. **Gate 0 works** — retrieval provably did not move.
3. **`missing_field` is not dead code** — the real model populated it on 8 of 142 rows,
   with values like `'start date and end date'`. It had never been exercised before.
4. **A prompt defect, found for 329 requests instead of during a `dev100` gate at 800.**

## Fixed before any rerun

- The prompt now requires `answer` in prose for every operation, and says why.
- `answer_was_raw_structure` is recorded per row, so this contamination is countable
  rather than discovered by reading answers.
- Three tests pin the requirement and the flag.

## What a rerun costs

The control arm is reusable — it is unaffected by the prompt defect and Gate 0 already
verified the pairing. Only the candidate arm needs re-running: 142 probes ≈ 175 requests.

**The next attempt must also separate the three answerer changes**, or it will produce
another table nobody can attribute.
