# v3.3 dev60 validation — sealed

The registered one-shot `dev60` run, complete and spent. Verify with:

```bash
python tools/verify_v3_dev60.py
```

## Result

| | `v2-control` | `v3.3-compact` |
|---|---:|---:|
| majority accuracy | 46.7% | **55.0%** |
| mean accuracy over three runs | 49.4% | 55.6% |
| standard deviation across runs | 1.9pp | 2.5pp |
| unanimous agreement across runs | 85% | 95% |
| median context tokens | 573 | 1,115 — 1.95x |
| raw-fallback trigger rate | 30.6% | 22.8% |
| selected source recall | 98.3% | 98.3% |

Pre-registered slices, none regressed: temporal 44.4% → 55.6%, multi-session
33.3% → 38.9%, knowledge-update 70.0% → 80.0%, ordinary 50.0% → 57.1%,
high-confidence-wrong 0 → 0.

All nine registered gates passed. The paired result is **+8.3 points at `p=0.1797`**
— seven wins against two losses on sixty questions, which is **not statistically
conclusive**. The gate was written to turn on consistency across slices rather than
on that p-value, and the honest reading is a consistent improvement of uncertain size.

Both arms score well below their `tune42` figures because `dev60` is deliberately the
hard slice: eighteen temporal and eighteen multi-session questions out of sixty. The
two sets are not comparable.

## What the aggregate says about where the remaining error is

Selected source recall is **98.3%** for both arms while accuracy is 55.0%. That
43-point gap between finding the evidence and answering from it is visible entirely at
the aggregate level, and it agrees with the separate 222-failure taxonomy built on
readable pools ([`results/failure-taxonomy.md`](../../failure-taxonomy.md)). No
individual `dev60` row was read to establish it.

## Reading rule

Per [the pre-registration](../../prereg-v3-dev60.md) and
[the data protocol](../../data-protocol.md): **aggregate metrics and pre-registered
slices only.** Individual rows are not read, and `dev60` must not be used to tune any
later candidate. The one-shot ledger forbids a second run.

## What is in this package, and what deliberately is not

| | |
|---|---|
| `conclusion.json` | the outcome, a summary checked against the aggregate, and a hash inventory of every bound file |
| `source.tar.gz` | the exact answer-affecting source, byte-for-byte verified against the freeze |
| `README.md` | this page |

The 360 per-question rows are **bound by hash in place** under
`results/sealed/v3-dev60/`, not copied here. Copying them into a committed archive
directory would publish exactly what the seal exists to withhold, so the package
records what they hash to instead. That is the whole difference between the result
being auditable and the questions being spent.

The pre-registration, the aggregate, the one-shot ledger, the configuration, the
manifest and this verifier are bound the same way: by path and hash, at their own
locations, so a single edit anywhere in the chain fails verification.
