# Controlled relation vocabulary — offline proposal

**Zero provider calls.** The encoder is the local MiniLM the project already loads.
Nothing was written to any store; the output is
[`predicate-map.csv`](predicate-map.csv), a reviewable 5,054-row proposal.
Tool: [`tools/predicate_vocabulary.py`](../../tools/predicate_vocabulary.py).

## The problem being solved

`predicate` is nominally the predicate of a triple and is in practice a label the
extractor invents per fact. On `train150`: 5,054 distinct values over 18,519
memories, 3,578 used exactly once, and 3,459 memories whose predicate is the literal
string `none`. The consequence is measurable — 81% of `(user_id, subject, predicate)`
keys hold one memory and so can never supersede, and supersession fires on **0.41%**
of the store.

String normalisation was measured first and is not the fix: collapsing case,
separators, plurals and word order merges 5% of the vocabulary. These are not
spelling variants, they are separate inventions.

## The vocabulary

38 relation types, grouped by **arity** — the property that decides what a second
value on the same key means, and the thing the free-text predicate cannot express.

| arity | meaning | types |
|---|---|---|
| `single` | one value true at a time; a new value closes the old interval | `lives_in` `works_as` `works_at` `studies` `relationship_status` `health_status` `current_goal` `uses_tool` `prefers` `avoids` `owns_pet` `drives` `team_size` `routine` `dietary_rule` |
| `set` | values accumulate; "how many X" is an exhaustive scan, never a top-k | `acquired` `owns` `visited` `attended` `read` `watched` `listened_to` `ate_at` `used_service` `completed` `practises_hobby` `cooked` `grows` `knows_person` `financial_activity` |
| `event` | a point in time; nothing supersedes, order is the question | `event_occurred` `planned_event` `milestone` `changed_state` |
| `speech` | what the assistant said; filtered by `source_role`, not by subject | `recommended` `advised_against` `explained` `committed_to` |

Mapping is a rule pass then an embedding pass. Pure similarity was tried first and
was not good enough: `book_recommendation` scored 0.439 against `recommended`,
because MiniLM is comparing a two-word label against a sentence and the shared head
noun carries almost no weight. The head of the compound *is* the relation, so 24
head-word rules run first and the encoder only sees what they miss. That moved the
unmapped share from 58.8% to 19.0%.

## Coverage

| bucket | memories | share |
|---|---:|---:|
| mapped to a relation type | 11,537 | **62.3%** |
| `other` — below the 0.35 similarity floor | 3,523 | 19.0% |
| `unkeyed` — predicate is literally `none` | 3,459 | 18.7% |

`unkeyed` is kept separate on purpose. It is not a weak match; it is the extractor
declining to key the fact at all, and mapping it by similarity would invent a
relation nobody asserted.

## What it buys, measured like for like

Comparing raw predicate against relation type **on the same 11,537 memories** — the
earlier store-wide comparison was not honest, because the mapped side silently
dropped the 38% it does not cover:

| key | keys | singleton keys | memories on multi-memory keys |
|---|---:|---:|---:|
| `(user, subject, raw predicate)` | 5,248 | 4,084 | 7,453 — 64.6% |
| `(user, subject, relation_type)` | **2,871** | **1,505** | **10,032 — 87.0%** |

Singleton keys fall by 63%. On the `set`-arity subset, which is what an enumeration
question would scan, sets with three or more members go from **167 to 357**.

## Two corrections to the earlier projection

The estimate quoted before this ran was "57% → 91% of memories on multi-memory keys".
That was an **upper bound** computed by dropping the predicate entirely and keying on
`(user, subject)` alone. It assumed perfect canonicalisation over the whole store.
What a real mapping achieves is 64.6% → 87.0% **on the 62% it covers**, which is a
smaller claim.

And the median `set` still has **one** member even after mapping. Exhaustive
enumeration helps the 357 sets with three or more members; it is not a universal win,
and any claim that it fixes counting has to be measured on questions, not on keys.

## What this says about where to spend next

The 3,459 `unkeyed` memories are 18.7% of the store and no mapping can reach them —
they need the extractor to emit a relation type in the first place, which is a
`response_schema` change on Stage B rather than an offline pass. That is plausibly a
larger lever than raising the similarity floor on the remaining 19%.
