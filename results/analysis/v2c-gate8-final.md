# v2c.8 gate-8 final

Date: 2026-09-17 (Australia/Sydney)

Decision: **PASS — advance one candidate to 48-question confirmation.**

## Valid items

The seven source-consistent questions were correct in both frozen v2c.3 hybrid
repetitions: **7/7 and 7/7**. They cover update replacement, exact-date recovery,
multi-session calculation, temporal composition, and three regression controls.

## Invalid item

`778164c6` is excluded from the valid-item accuracy denominator because its question
and gold cannot both be reconciled with the source conversation. v2c.8 answered it
twice with the required production behavior: explicitly identifying Escovitch Fish
as the recommended Jamaican snapper dish and Grilled Snapper with Mango Salsa as the
fruit-bearing dish. Both configured judge calls also returned correct.

## What is established—and what is not

- Established: each diagnosed v2b failure mechanism now has a stable repair on the
  small gate, without regressing its controls.
- Established: the hybrid arm improves the seven valid rows from the v2c.2
  memory-only pattern of 5/7 to 7/7 on this selected gate.
- Not established: population accuracy. Eight mechanism-selected rows cannot support
  an expected overall LongMemEval score.
- Next: one run on the pre-existing frozen 48-question set, paired against the frozen
  prior arm. No repeated baseline and no three-run replication at this stage.
