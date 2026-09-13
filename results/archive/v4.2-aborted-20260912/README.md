# v4.2 live attempt 1 — aborted on row 2, sealed as a diagnosis

**Cost: 3 provider attempts, 3,188 tokens, 1 recorded failure. No held-out probe touched.**
The run stopped itself on its second row rather than producing 404 rows of confounded
evidence, which is what the per-row validation is for.

## What stopped it

`validate_rows` rejected the candidate row on `answer_was_raw_structure`. That check was
wrong: the field is a *diagnostic*, and the runner documents it as true even when `compute`
or a populated `answer` produced a perfectly good reply afterwards — it asks "did the model
emit a structure at any point". The gate now tests what actually reaches the reader, and
records the diagnostic instead of dying on it.

## What it found, which was worth the three requests

The two completions are the evidence, and they differ in a way that is not the mechanism:

| arm | reply |
|---|---|
| control | `status`, **`answer`: "The user attended 3 distinct events."**, `operation`, `items` |
| candidate | `status`, `operation`, `items`, `member_labels` — **no `answer`** |

`compute` rescued this particular row: the citations were valid, so the reader saw
"3 distinct: Easter Sunday service…, dentist appointment, St. Joseph's Day celebration…".

But nothing rescues `current_state` or `lookup`, where no arithmetic runs. The candidate
would have abstained there while the control answered — a second difference between the
arms, on the three strata that exist to check for exactly that. The registered safety
direction would have failed on prompt formatting rather than on the mechanism, and the
result would have been unreadable in the same way v4.0 attempt 1 was.

Cause: the enumerate prompt's count block ended "The total is not your job", which the
model appears to have extended to the prose as well. It now says the reply still is its
job, and that filling `member_labels` does not discharge it. Two tests pin that both
prompts demand a prose answer and that they differ only inside the count block.

The corrected freeze is `results/frozen/v4.2-development-20260912c/`.
