# v4.2 preflight history — three attempts before the run that counted

Total cost of the three: **31 provider attempts, ~42,000 tokens**, all on development
probes. No held-out probe touched. Each attempt stopped on eight rows or fewer rather than
producing 404 rows of confounded evidence.

Every iteration below was decided on a **format property** — whether the model populates
`answer` — and no verdict, accuracy figure or effect size was read while deciding it.

| attempt | freeze | change under test | cost | result |
|---|---|---|---|---|
| 1 | `…0912b` | first enumerate prompt | 3 attempts | **aborted on row 2.** The per-row check gated `answer_was_raw_structure`, a diagnostic the runner documents as true even when the answer reads correctly. Gate corrected to test what reaches the reader. Underneath it, a real defect: the candidate omitted `answer` where the control supplied it. |
| 2 | `…0912c` | prompt told to write the reply too | 9 attempts | **3 of 4 candidate probes still omitted `answer`.** On `current_state_0041` the control answered substantively and the candidate returned "I do not know." purely from the missing field. |
| 3 | `…0912d` | prompt appended rather than rewritten | 9 attempts | **2 of 4 still omitted it**, both non-count. Improvement, not a fix. Prompt iteration stopped here: a third revision fitted against four probes would be tuning on the instrument. |
| 4 | `…0912e` | **`answer` required by the v4 schema** | 10 attempts | **8 of 8 populated, both arms, no raw structure.** The confound is closed. |

## Why the prompt could not fix it

The runner already says it: *"a prompt is not a contract"*. The prompt has asked for
`answer` on every operation since v4.0, and two measured revisions did not make the model
comply. The first rewrote the shared count block and cost the *other* operations their
prose — reshaping one operation's instructions degraded the rest, which is a second
difference between the arms and exactly the confound the single-variable rule exists to
prevent.

`answer` is now required on `SynthesisVerdict`, the v4 base, so **both** v4 arms carry the
requirement and the comparison still differs only in count-member citation. The v2 and v3
verdicts are untouched, and the archived v4 runs keep their own frozen source, so nothing
already measured changes meaning.

## What this cost, and what it bought

31 attempts is about 5% of the 600-request budget. It bought a run whose two arms differ
in the mechanism under test and not in whether they answer at all — which is the
difference between a readable result and v4.0 attempt 1, where 55 of 142 rows reached the
grader as raw JSON and the headline table was unusable.
