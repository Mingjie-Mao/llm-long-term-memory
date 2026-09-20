# v2d refusal taxonomy

> Rows: `results/raw/two_stage_v2d.v2d-gate16.jsonl`. No model calls; derived from the committed rows alone.

- questions: **16**
- code produced the number: **2**
- no calculation was requested (`lookup`): **3**
- no computation record at all: **2**
- refused, model prose kept: **9** (of which wrong: 6)
- refusals classifiable from the row: **0**; unclassifiable: **9**

## By cause

| cause | refusals |
|---|---:|
| `unclassifiable_legacy_row` | 9 |

## By operation

| operation | refusals |
|---|---:|
| `average` | 1 |
| `count` | 4 |
| `difference` | 1 |
| `duration` | 1 |
| `sum` | 2 |

## Why some rows cannot be classified

These rows were written before `v2d.compute` recorded a machine-readable
`cause`. Each carries one sentence that stands for more than one failure, and
the operands the model supplied were not kept, so the breakdown cannot be
recovered without running the questions again.

| question | reason recorded | could be |
|---|---|---|
| `1192316e` | numeric operands are invalid | `operand_arrays_misaligned`, `numeric_unparseable`, `label_not_in_context`, `too_few_operands` |
| `2e6d26dc` | an item has no valid source label | `member_text_empty`, `label_not_in_context` |
| `9d25d4e0` | an item has no valid source label | `member_text_empty`, `label_not_in_context` |
| `c18a7dc8` | numeric operands are invalid | `operand_arrays_misaligned`, `numeric_unparseable`, `label_not_in_context`, `too_few_operands` |
| `d23cf73b` | an item has no valid source label | `member_text_empty`, `label_not_in_context` |
| `d682f1a2` | an item has no valid source label | `member_text_empty`, `label_not_in_context` |
| `gpt4_59149c77` | a date operand or citation is invalid | `date_unparseable`, `label_not_in_context` |
| `gpt4_a1b77f9c` | numeric operands are invalid | `operand_arrays_misaligned`, `numeric_unparseable`, `label_not_in_context`, `too_few_operands` |
| `gpt4_d12ceb0e` | numeric operands are invalid | `operand_arrays_misaligned`, `numeric_unparseable`, `label_not_in_context`, `too_few_operands` |
