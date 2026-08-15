# Root Cause: `single-session-assistant` scores 0/4

**Verified 2026-08-14 against `stores/two-stage-hydrated.db`. No API quota spent.**

## The gap

| Category | full_context | naive_rag | v1 | two_stage | two_stage_hydrated |
|---|---:|---:|---:|---:|---:|
| `single-session-assistant` (4) | **4/4** | **4/4** | 0/4 | 0/4 | 0/4 |

Both baselines answer every one; every memory variant answers none. A gap that
uniform is a policy defect, not a ranking problem.

## Trace

Where the answer actually lives, and what survived into the store:

| Question | Gold | Answer turn | Turns stored | Gold in raw text | Memories from that session |
|---|---|---|---:|:---:|---:|
| `1b9b7252` | Mindful.org | **assistant** | 8 | yes | 1 |
| `41275add` | Mayo Clinic video URL | **assistant** | 2 | yes | **0** |
| `8aef76bc` | Mod Podge sealant | **assistant** | 8 | yes | 2 |
| `4388e9dd` | Andy's stained white shirt | user (script brief) | 6 | yes | **0** |

**The raw evidence is present in every case.** `turns` holds the answer-bearing
text, and `source_session_id` resolves to a `turns` row for 4,843 / 4,843 memories,
so provenance is intact. Nothing was lost at the storage layer.

What the extractor produced from those sessions instead:

```
answer_ultrachat_115151  ->  [semantic] user/physical_limitations:
                               "The user is unable to participate in strenuous activities."
                             (assistant turn 3 held Mindful.org — not extracted)

answer_ultrachat_563222  ->  [semantic] user/collections:
                               "The user has a lot of wine corks and bottle caps lying around."
                             [semantic] user/diy_projects:
                               "The user intends to try making a wine cork bulletin board."
                             (assistant turn 1 held Mod Podge — not extracted)

answer_sharegpt_81riySf_0 -> nothing. The only user turn is "any youtube video i
                             can share with them?", which asserts no user fact. The
                             assistant's reply, containing the URL, was discarded.

answer_sharegpt_qTi81nS_0 -> nothing. The user turn is a request to write a script;
                             the assistant's reply is the script. Neither is a fact
                             *about the user*.
```

## Root cause

The extraction prompt already carries an instruction to record assistant facts:

> *Also record concrete things the assistant told this user that they might ask
> about later — a named product it recommended, a figure it gave. Write those as
> "The assistant recommended …". Not its generic advice.*

So this is **not a missing rule. It is a rule the model does not reliably follow.**

Measured compliance:

| Store | Memories | `subject='assistant'` | Share |
|---|---:|---:|---:|
| v1 | 2,007 | 12 | 0.55% |
| two-stage | 4,843 | 326 | **6.73%** |

The two-stage rewrite already improved compliance 12x, which is part of why it wins
the ablation. But 6.73% is still far below the rate at which assistant turns carry
answer-relevant specifics, and the failures share a shape: the assistant's reply is
a **long enumerated list** ("1. Wine Cork Bulletin Board… 2. …"), and the model
appears to classify the whole list as the "generic advice" the prompt tells it to
skip — discarding the one named product or URL buried inside it.

Two structural contributors:

1. **The prompt opens with "the facts the user stated about themselves."** The
   assistant clause arrives late, after that framing and immediately before a Skip
   rule. Everything around it pulls toward user-profile extraction.
2. **The worked example contains no assistant fact.** This project has already
   measured that worked examples matter — adding them moved source fidelity
   33.6% → 36.6%. The one instruction with no example is the one being ignored.

## A third failure this exposes

`4388e9dd` is not an assistant problem at all. The answer (Andy's shirt) sits in a
**user** turn, but the subject is a fictional character, not the user. The store's
subject distribution is 4,517 `user` against 326 `assistant` and essentially nothing
else, so a fact about any third entity has nowhere to live.

The defect is therefore broader than speaker role: **memory is being equated with
user-profile extraction.** The correct framing is that memory should preserve
task-relevant information regardless of who said it or who it is about.

## Fix (designed, not yet applied — it requires re-ingestion)

Not "store every assistant turn": that floods the store with model chatter and is
why the current conservative rule exists. Classify instead, and make the classes
explicit in the schema as `memory_scope`:

| Scope | Keep | Example |
|---|:---:|---|
| `USER_FACT` / `USER_PREFERENCE` / `USER_PLAN` | ✅ | "The user stopped drinking coffee." |
| `ASSISTANT_FACT` | ✅ | "The assistant said Mindful.org has free guided imagery exercises." |
| `ASSISTANT_COMMITMENT` | ✅ | "The assistant agreed to draft the itinerary." |
| `SHARED_CONTEXT` | ✅ | "Andy wears an untidy, stained white shirt in the script." |
| `ASSISTANT_TRANSIENT` | ❌ | Generic advice, pleasantries, restatements of the question |

Concrete changes:

1. **Add a worked example containing an assistant fact and a third-entity fact.**
   Cheapest change with the best measured precedent in this repo.
2. **Name the artifact types that must never be dropped**: URLs, product and brand
   names, figures, and titles — even when they appear inside a list the model would
   otherwise treat as generic.
3. **Add `memory_scope` to the schema** so retrieval and packing can weight or
   filter by it, and so the store stops implying every memory is about the user.
4. **Re-gate**: `lltm ingest coverage` on the evidence-only split, and add
   assistant-answer questions to the gate set, which currently cannot see this bug.

**Cost:** prompt-only changes are free to write but require re-ingestion to take
effect (~600 extractor requests for the full dev-50 store, i.e. more than one
free-tier day). Sequenced accordingly in [docs/ROADMAP.md](../docs/ROADMAP.md).

## Note on `single-session-preference` (0/3)

Not the same problem and must not be pooled with it. `full_context` sees the entire
history and still scores 0/3, so the failure is upstream of memory entirely —
answerer prompt, judge, or reference answers. Audit those three by hand before
touching any memory code.
