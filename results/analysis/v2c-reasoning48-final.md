# v2c.8 reasoning-48 confirmation

Decision: **PASS**

- configured judge: **37/48 (77.1%)**
- valid-item accuracy, excluding audited invalid `778164c6`: **76.6%**
- median context: **636 tokens**
- source-session recall: **95.8%**

## Registered checks

- PASS — `overall`
- PASS — `knowledge-update`
- PASS — `multi-session`
- PASS — `single-session-assistant`
- PASS — `single-session-preference`
- PASS — `single-session-user`
- PASS — `temporal-reasoning`
- PASS — `median_context_tokens`
- PASS — `source_session_recall`

## Accuracy by type

- `knowledge-update`: 87.5%
- `multi-session`: 58.3%
- `single-session-assistant`: 83.3%
- `single-session-preference`: 100.0%
- `single-session-user`: 83.3%
- `temporal-reasoning`: 75.0%

## Historical directional comparison

- archived v2-control majority: 68.8%
- archived v3-reasoned majority: 72.9%
- v2c.8 single run: 77.1%

These are directional, not a paired significance claim: v2c.8 has one run, while each archived comparator is a three-run majority.

## Eleven judged failures

- false-premise handling: 2 — `0ddfec37_abs`, `gpt4_70e84552_abs`
- composition/arithmetic: 6 — `a4996e51`, `a9f6b44c`, `d905b33f`, `gpt4_2f8be40d`, `gpt4_74aed68e`, `gpt4_7a0daae1`
- source-detail selection: 2 — `58ef2f1c`, `dd2973ad`
- retrieval miss: 1 — `e48988bc`

No candidate changes or reruns are made from these 48 rows. The next honest measurement is a separately frozen final set.
