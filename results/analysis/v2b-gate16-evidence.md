# v2b gate16 — zero-call evidence gate

**No model/API calls were made.** Both stores are compared on the same frozen 16 questions.

## Extraction fidelity on all 780 gate sessions

Batch15: **39.8%**, 2,024 source-linked active memories.  
Batch8: **47.5%**, 3,049 source-linked active memories.  
Change: **+7.7 points**.

| facet | batch15 | batch8 | retained change |
|---|---:|---:|---:|
| quantity | 41.0% | 48.8% | +67 |
| duration | 39.3% | 46.1% | +12 |
| money | 50.5% | 61.9% | +11 |
| date | 11.8% | 17.6% | +1 |
| relative_time | 22.4% | 24.0% | +3 |
| proper_noun | 41.4% | 50.0% | +86 |

## Evidence endpoints

| endpoint | batch15 | batch8 |
|---|---:|---:|
| whole-store literal (measurable) | 9 | 8 |
| top-20 literal (measurable) | 9 | 7 |
| top-20 any gold session | 15 | 16 |
| top-20 all gold sessions | 12 | 16 |
| raw top-2 any gold session | 14 | 14 |
| raw top-2 all gold sessions | 7 | 7 |
| median top-20 memory tokens | 362 | 392 |

## Support floor

Batch15 median/p10 lexical support: 88.9% / 71.4%; below 25%: 2.
Batch8 median/p10 lexical support: 87.5% / 66.7%; below 25%: 4.

This remains a low-support screen, not a hallucination rate.

## Changed questions

- `57f827a0` — top20_any_source: False→True, top20_all_sources: False→True
- `58ef2f1c` — all_store_literal: True→False, top20_literal: True→False
- `6e984301` — top20_all_sources: False→True
- `778164c6` — all_store_literal: True→False, top20_literal: True→False
- `a96c20ee` — all_store_literal: False→True, top20_literal: False→True, top20_all_sources: False→True
- `d905b33f` — top20_all_sources: False→True
- `gpt4_45189cb4` — all_store_literal: False→True, top20_literal: False→True
- `gpt4_7a0daae1` — top20_literal: True→False
