"""Paired comparison when questions come in clusters.

A benchmark can ask many questions of one conversation, all answered from one
store, so their outcomes are not independent: a store that lost a session loses every
question resting on it. A test that counts those questions as independent overstates its
own n — the error the count review's shared-evidence clustering was built to prevent.

So the unit here is the conversation. Under the null that two arms are exchangeable, each
conversation's net difference was as likely to come out with the opposite sign, and the
sign-flip test enumerates or samples those flips. The minimum detectable effect uses the
same null, with the correlation inside a conversation entering as a design effect,
`1 + (m - 1) * ICC`. The ICC is an assumption until two runs of one arm measure it.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from math import sqrt
from random import Random

EXACT_UP_TO = 16
"""Clusters up to which every sign assignment is enumerated (2^16 = 65,536)."""


def design_effect(cluster_size: float, icc: float) -> float:
    if cluster_size < 1:
        raise ValueError("a cluster holds at least one question")
    if not 0 <= icc <= 1:
        raise ValueError("the intra-cluster correlation lies in [0, 1]")
    return 1 + (cluster_size - 1) * icc


def minimum_detectable_effect(
    n_questions: int, discordance: float, cluster_size: float, icc: float, sigmas: float = 2.0
) -> float:
    """The smallest accuracy difference, as a fraction, that clears `sigmas` of the null.

    Under the null each question disagrees between the arms with probability
    `discordance`, and each disagreement is a fair +/-1, so the net over n questions has
    variance `n * discordance`, inflated by the design effect when questions correlate.
    """
    if n_questions <= 0:
        raise ValueError("a stratum holds at least one question")
    if not 0 <= discordance <= 1:
        raise ValueError("discordance is a probability")
    variance = n_questions * discordance * design_effect(cluster_size, icc)
    return sigmas * sqrt(variance) / n_questions


def sign_flip_p_value(
    cluster_nets: Sequence[float], permutations: int = 100_000, seed: int = 0
) -> float:
    """Two-sided p-value of the summed per-cluster difference under random sign flips.

    Exact over all 2^k assignments up to `EXACT_UP_TO` clusters. Beyond that, `permutations`
    seeded draws, counting the observed assignment among them so the p-value is never zero.
    """
    nets = [float(net) for net in cluster_nets]
    observed = abs(sum(nets))
    if not nets or observed == 0:
        return 1.0
    tolerance = 1e-9
    if len(nets) <= EXACT_UP_TO:
        extreme = sum(
            1
            for signs in product((1, -1), repeat=len(nets))
            if abs(sum(sign * net for sign, net in zip(signs, nets, strict=True)))
            >= observed - tolerance
        )
        return extreme / 2 ** len(nets)
    rng = Random(seed)
    extreme = 1
    for _ in range(permutations):
        flipped = sum(net if rng.random() < 0.5 else -net for net in nets)
        if abs(flipped) >= observed - tolerance:
            extreme += 1
    return extreme / (permutations + 1)
