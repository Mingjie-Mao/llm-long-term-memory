"""Refuse a gate the instrument cannot read.

Every v4 comparison so far was decided by a margin smaller than the disagreement between
two runs of one configuration. `v4.0-flat` and `v4.0-flat2` are the same variant on the
same store with byte-identical retrieval on all 142 probes, and they differ by net 7
probes with 38 verdicts flipped. A gate written in single probes — "at most one probe may
regress" — rejects and promotes that same configuration depending on the draw.

So a gate has to be checked against the noise *before* it is registered, and the check
has to be mechanical, because the moment to talk yourself into a threshold is exactly the
moment you already know the mechanism's offline numbers.

## What is measured, and what is modelled

**Observed**: binary correctness disagreement, wins + losses, across the historical
pair. Categorical switches between wrong, abstained and unparseable do not move accuracy.
The original runs lack source-bound provenance, so pure sampling is an assumption.

**Modelled**: how far a net difference can stray under the null. For paired binary
outcomes only the discordant probes move, and under "no true difference" each is a fair
coin, so the net is a sum of D fair +/-1 steps: `sd(net) = sqrt(D)`, `D = d * n`. That is
the standard paired-sign/McNemar null and it assumes only that the two runs are exchangeable
— which is what identical configuration and identical retrieval buy.

**Modelled more loosely**: the effect of repeats. Majority-of-`r` voting pushes a probe's
per-run probability away from 0.5, and the reduction is computed from a homogeneous-`p`
approximation — every probe assumed to share the `p` solving `2p(1-p) = d`. Real probes
are heterogeneous, so treat all figures as conditional estimates, not power guarantees.
For example, a mixture of deterministic probes and p=0.5 probes retains the same
disagreement after odd-majority voting; the homogeneous model then overstates its benefit.

A gate passes when the effect it claims to detect exceeds `--sigmas` standard deviations
of the null. The default of 2 is not a p-value; it is the minimum at which a result is not
routinely produced by resampling the same arm.
"""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DIAGNOSIS = REPO / "results/analysis/v4-probe-diagnosis.json"


def disagreement_rate(noise: dict) -> float:
    """Binary correctness discordance; wrong-to-abstained does not move accuracy."""
    n = noise["n"]
    changed = noise["wins"] + noise["losses"]
    if n <= 0 or noise["wins"] < 0 or noise["losses"] < 0 or changed > n:
        raise ValueError("invalid paired correctness counts")
    return changed / n


def _flip_rate_after_repeats(rate: float, repeats: int) -> float:
    """Per-probe disagreement after majority-of-`repeats` voting, homogeneous-p model."""
    if repeats <= 1:
        return rate
    if repeats % 2 == 0:
        raise ValueError("majority voting needs an odd number of repeats")
    # 2p(1-p) = rate has two roots, mirror images about 0.5; either gives the same
    # flip rate, so take the one above 0.5.
    discriminant = max(0.25 - rate / 2, 0.0)
    p = 0.5 + sqrt(discriminant)
    # Probability the majority of `repeats` draws is correct.
    majority = 0.0
    need = repeats // 2 + 1
    for wins in range(need, repeats + 1):
        majority += _binomial(repeats, wins) * p**wins * (1 - p) ** (repeats - wins)
    return 2 * majority * (1 - majority)


def _binomial(n: int, k: int) -> int:
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def minimum_detectable_effect(rate: float, stratum: int, repeats: int, sigmas: float) -> float:
    """Smallest net probe difference that clears the null on a stratum of this size."""
    discordant = _flip_rate_after_repeats(rate, repeats) * stratum
    return sigmas * sqrt(discordant)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnosis", type=Path, default=DIAGNOSIS, help="file holding the measured noise floor"
    )
    parser.add_argument(
        "--stratum",
        type=int,
        action="append",
        default=None,
        help="stratum size the gate is read on; repeatable. Default: every v4 stratum.",
    )
    parser.add_argument(
        "--effect",
        type=float,
        default=None,
        help="net probe difference the gate claims to detect. Omit to just print the table.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="runs per arm, majority voted")
    parser.add_argument("--sigmas", type=float, default=2.0)
    args = parser.parse_args()

    if not args.diagnosis.is_file():
        print(f"STOP: no measured noise floor at {args.diagnosis}")
        print("Run tools/diagnose_v4_probes.py first. A gate registered against an")
        print("unmeasured instrument is the thing this script exists to prevent.")
        return 2

    payload = json.loads(args.diagnosis.read_text(encoding="utf-8"))
    noise = payload["noise_floor"]
    rate = disagreement_rate(noise)

    print(f"observed disagreement — {' vs '.join(noise['pair'])}")
    print(f"  probes compared          {noise['n']}")
    print(f"  retrieval identical on   {noise['retrieval_identical']}")
    print(f"  verdict categories changed {noise['verdict_changed']}")
    print(f"  correctness changed     {noise['wins'] + noise['losses']}  ({rate:.1%} per probe)")
    print(f"  observed net difference  {noise['net']:+d}")
    print("  Conditional model only: historical source identity is not established;")
    print("  repeat estimates assume homogeneous, independent per-probe draws.")
    if noise["retrieval_identical"] != noise["n"]:
        print("  WARNING: retrieval was not identical, so this pair bounds more than sampling")

    strata = args.stratum or sorted(
        {entry["n"] for entry in payload["scoreboard"]["v4.0-flat"]["by_kind"].values()}
        | {noise["n"]}
    )
    print(f"\nminimum detectable effect at {args.sigmas:g} sigma, in net probes")
    print(f"  {'stratum':>8}  {'1 run':>8}  {'3 runs':>8}  {'5 runs':>8}")
    for size in strata:
        cells = "  ".join(
            f"{minimum_detectable_effect(rate, size, r, args.sigmas):8.1f}" for r in (1, 3, 5)
        )
        print(f"  {size:8d}  {cells}")

    if args.effect is None:
        print("\nNo --effect given, so nothing was judged. Registering a gate means naming")
        print("the effect it must detect and re-running this with it.")
        return 0

    print(f"\njudging a gate that claims to detect {args.effect:+g} net probes")
    print(f"  on {args.repeats} run(s) per arm")
    failed = []
    for size in strata:
        needed = minimum_detectable_effect(rate, size, args.repeats, args.sigmas)
        ok = abs(args.effect) >= needed
        print(f"  stratum {size:4d}: needs {needed:5.1f}  {'OK' if ok else 'NOT RESOLVABLE'}")
        if not ok:
            failed.append(size)

    if failed:
        print(
            f"\nSTOP: on strata {failed} this gate would decide by a margin the instrument "
            "reproduces by chance."
        )
        print("Raise the effect, pool strata, add repeats, or state the gate as a direction")
        print("rather than a threshold. Do not register it as written.")
        return 1
    print("\nOK under the stated model; this is not a statistical-power guarantee.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
