"""Frozen v4.2 schedule, evidence validation and paired development inference."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

ARMS = {"control": "two_stage_synthesis_flat", "candidate": "two_stage_synthesis_enumerate"}
SEED = 420912
MAX_REQUESTS = 600
MAX_OBSERVED_TOKENS = 2_000_000
FIELDS = ("retrieved_ids", "evidence_found", "evidence_needed", "context_complete")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def append_json(path: Path, value) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_rows(path: Path) -> list[dict]:
    return (
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if path.exists()
        else []
    )


def schedule(probes: list[dict]) -> list[dict]:
    counts = Counter(p["kind"] for p in probes)
    if counts != {"count": 30, "duration": 30, "comparison": 33, "current_state": 49}:
        raise ValueError("v4.2 requires the exact 142-probe development strata")
    if len({p["probe_id"] for p in probes}) != len(probes):
        raise ValueError("duplicate development probe")
    rng = random.Random(SEED)
    cells = []
    for repeat in (1, 2, 3):
        selected = [p for p in probes if repeat == 1 or p["kind"] == "count"]
        rng.shuffle(selected)
        for i, p in enumerate(selected):
            order = ["control", "candidate"] if (i + repeat) % 2 else ["candidate", "control"]
            for arm in order:
                cells.append(
                    {
                        "cell_id": f"{p['probe_id']}.r{repeat}.{arm}",
                        "probe_id": p["probe_id"],
                        "kind": p["kind"],
                        "repeat": repeat,
                        "arm": arm,
                        "variant": ARMS[arm],
                    }
                )
    return cells


def validate_rows(
    rows: list[dict], cells: list[dict], probes: list[dict], freeze_id: str, mode: str
) -> dict:
    expected = {c["cell_id"]: c for c in cells}
    specs = {p["probe_id"]: p for p in probes}
    found = {}
    evidence = {}
    for row in rows:
        key = row.get("cell_id")
        if key not in expected or key in found:
            raise ValueError("unknown or duplicate scheduled cell")
        c = expected[key]
        p = specs[c["probe_id"]]
        if any(row.get(k) != v for k, v in c.items()):
            raise ValueError("row does not match its scheduled arm/repeat/type")
        if row.get("freeze_id") != freeze_id or row.get("mode") != mode:
            raise ValueError("row belongs to another freeze or execution mode")
        if (
            row.get("question") != p["question"]
            or row.get("gold") != p["answer"]
            or row.get("namespace") != p["namespace"]
        ):
            raise ValueError("row question identity changed")
        if row.get("verdict") not in {"correct", "wrong", "abstained", "unparseable"}:
            raise ValueError("unobserved verdict")
        ids = row.get("retrieved_ids")
        if (
            not isinstance(ids, list)
            or not all(isinstance(x, str) and x for x in ids)
            or len(ids) != len(set(ids))
            or type(row.get("context_complete")) is not bool
            or type(row.get("evidence_found")) is not int
            or type(row.get("evidence_needed")) is not int
            or not 0 <= row["evidence_found"] <= row["evidence_needed"]
        ):
            raise ValueError("invalid retrieval evidence")
        # `answer_was_raw_structure` is a *diagnostic*, and the runner documents it as
        # true even when `compute` or a populated `answer` produced a perfectly good reply
        # afterwards — it asks "did the model emit a structure at any point". Gating on it
        # aborted this run on its second row: the candidate omitted `answer`, `compute`
        # supplied "3 distinct: ..." from the cited labels, and the answer read correctly.
        #
        # What must actually be gated is what reaches the reader, so that is what is
        # checked. The diagnostic is required to be present and is recorded, because the
        # rate at which each arm leans on the repair is itself a difference between arms.
        if not isinstance(row.get("answer_was_raw_structure"), bool):
            raise ValueError("raw-structure diagnostic not observed")
        if str(row.get("text") or "").lstrip().startswith(("{", "```")):
            raise ValueError("raw structure reached the reader")
        values = [row[k] for k in FIELDS]
        previous = evidence.setdefault(c["probe_id"], values)
        if values != previous:
            raise ValueError("Gate 0 failed across arms or repeats")
        found[key] = row
    # Appending only a schedule prefix makes restart deterministic.
    if list(found) != [c["cell_id"] for c in cells[: len(found)]]:
        raise ValueError("rows are not a prefix of the frozen schedule")
    return found


# Registered 2026-09-12, before any provider call, alongside the existing binary gate
# rather than in place of it. No registered threshold is moved.
#
# **Why a second reading was needed.** The binary gate asks whether a count answer flipped
# from wrong to right. Simulated against the measured pair churn, it promotes a genuine
# six-probe improvement only about 31% of the time and an eight-probe one about 56% — so
# the most likely outcome of a paid run, even if the candidate works, is "inconclusive".
# Thirty binary outcomes is simply a coarse ruler.
#
# **What this measures instead.** The quantity the candidate is built to change: gold facts
# that were in the retrieved context and were never named. Control 130, v4.0-flat 45,
# v4.0-flat2 45, v4.1-scan 44. That the two runs of one configuration agree exactly at the
# total is partly luck — 11 of 30 probes move and the movements cancel — but the paired
# per-probe spread is sd 0.83, so the standard error of the 30-probe total is 4.6 facts and
# the 2-sigma band is +/-9. The control-to-v4 step of 85 facts is about 19 sigma on it.
#
# So this ruler resolves a 22% reduction in the remaining 45 where the binary gate cannot
# reliably resolve its own registered effect.
#
# **What it inherits.** Membership is decided by token overlap against the probe's
# SQL-derived gold memories, so the absolute level carries the matcher's error. The matcher
# is byte-identical for both arms and the reading is paired, so a *difference* survives a
# biased absolute level — which is why only the difference is registered.
UNCITED_TWO_SIGMA_FACTS = 10


def uncited_gold_facts(
    rows: list[dict], probes: list[dict], memories: dict[str, dict[str, str]]
) -> dict:
    """Per arm and probe, the mean number of in-context gold facts left unnamed.

    Averaged over the three repeats rather than majority-voted: the quantity is a count,
    and averaging keeps the resolution that collapsing it to a binary throws away.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from diagnose_v4_probes import _match

    specs = {p["probe_id"]: p for p in probes if p["kind"] == "count"}
    per_arm: dict[str, dict[str, list[int]]] = {arm: defaultdict(list) for arm in ARMS}
    for row in rows:
        spec = specs.get(row.get("probe_id"))
        if spec is None or row.get("arm") not in per_arm:
            continue
        gold = spec["evidence_memory_ids"]
        items = (row.get("synthesis_computation") or {}).get("items") or []
        named = {g for g in (_match(item, gold, memories) for item in items) if g}
        retrieved = set(row.get("retrieved_ids") or [])
        per_arm[row["arm"]][row["probe_id"]].append(
            sum(1 for g in gold if g not in named and g in retrieved)
        )

    means = {
        arm: {pid: sum(v) / len(v) for pid, v in probe_values.items() if v}
        for arm, probe_values in per_arm.items()
    }
    shared = sorted(set(means["control"]) & set(means["candidate"]))
    differences = [means["candidate"][pid] - means["control"][pid] for pid in shared]
    n = len(differences)
    total = sum(differences)
    if n > 1:
        mean = total / n
        variance = sum((d - mean) ** 2 for d in differences) / (n - 1)
        standard_error_total = (variance / n) ** 0.5 * n
    else:
        standard_error_total = 0.0
    return {
        "probes": n,
        "control_unnamed": round(sum(means["control"][pid] for pid in shared), 2),
        "candidate_unnamed": round(sum(means["candidate"][pid] for pid in shared), 2),
        # Negative means the candidate left fewer gold facts unnamed, which is the
        # direction the mechanism predicts.
        "difference": round(total, 2),
        "standard_error_of_total": round(standard_error_total, 2),
        "sigma": round(total / standard_error_total, 2) if standard_error_total else None,
        "registered_threshold_facts": -UNCITED_TWO_SIGMA_FACTS,
        "cleared": total <= -UNCITED_TWO_SIGMA_FACTS,
        "reading_rule": (
            "Paired difference only. The absolute level inherits the token matcher's "
            "error; the matcher is identical across arms, so the difference does not."
        ),
    }


def exact_cluster_p(differences: list[int]) -> float:
    """Two-sided exact sign-flip, swapping all probes in a namespace together."""
    distribution = {0: 1}
    for delta in differences:
        nxt = defaultdict(int)
        for total, count in distribution.items():
            nxt[total + delta] += count
            nxt[total - delta] += count
        distribution = nxt
    observed = abs(sum(differences))
    return sum(n for d, n in distribution.items() if abs(d) >= observed) / 2 ** len(differences)


def analyse(
    rows: list[dict],
    cells: list[dict],
    probes: list[dict],
    freeze_id: str,
    mode: str,
    memories: dict[str, dict[str, str]] | None = None,
) -> dict:
    indexed = validate_rows(rows, cells, probes, freeze_id, mode)
    if len(indexed) != len(cells):
        return {
            "outcome": "incomplete",
            "rows": len(rows),
            "expected_rows": len(cells),
            "freeze_id": freeze_id,
        }
    counts = [p for p in probes if p["kind"] == "count"]
    paired = []
    for p in counts:
        correct = {
            arm: sum(
                indexed[f"{p['probe_id']}.r{r}.{arm}"]["verdict"] == "correct" for r in (1, 2, 3)
            )
            >= 2
            for arm in ARMS
        }
        paired.append(
            (p["namespace"], int(correct["candidate"]) - int(correct["control"]), correct)
        )
    clusters = defaultdict(list)
    for namespace, delta, _ in paired:
        clusters[namespace].append(delta)
    cluster_p = exact_cluster_p([sum(v) for v in clusters.values()])
    net = sum(d for _, d, _ in paired)
    wins, losses = sum(d == 1 for _, d, _ in paired), sum(d == -1 for _, d, _ in paired)
    discordant = wins + losses
    binomial_p = min(
        1.0, 2 * sum(comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2**discordant
    )
    # Namespace bootstrap interval is descriptive, not another promotion test.
    groups = list(clusters.values())
    rng = random.Random(SEED)
    boot = []
    for _ in range(10_000):
        draw = rng.choices(groups, k=len(groups))
        boot.append(sum(map(sum, draw)) / sum(map(len, draw)))
    boot.sort()
    safety = {}
    for kind in ("duration", "comparison", "current_state"):
        per_arm = {
            arm: sum(
                r["verdict"] == "correct" for r in rows if r["arm"] == arm and r["kind"] == kind
            )
            for arm in ARMS
        }
        safety[kind] = {**per_arm, "net": per_arm["candidate"] - per_arm["control"]}
    safety_pass = sum(v["net"] < 0 for v in safety.values()) < 2
    candidate_counts = [r for r in rows if r["arm"] == "candidate" and r["kind"] == "count"]
    cited = sum(
        (r.get("synthesis_computation") or {}).get("counted_from") == "cited_labels"
        for r in candidate_counts
    )
    mechanism = cited >= 72
    outcome = "inconclusive"
    if net <= -6 or not safety_pass:
        outcome = "not_promoted"
    elif net >= 6 and cluster_p <= 0.05 and mechanism:
        outcome = "promote_to_next_development_comparison"
    elif not mechanism:
        outcome = "mechanism_not_demonstrated"
    return {
        "freeze_id": freeze_id,
        "mode": mode,
        "outcome": outcome if mode == "live" else "rehearsal_complete",
        "not_a_benchmark": True,
        "rows": len(rows),
        "gate_0": "PASS",
        "count_probes": len(counts),
        "namespaces": len(clusters),
        "majority_correct": {arm: sum(v[arm] for _, _, v in paired) for arm in ARMS},
        "net_count": net,
        "wins": wins,
        "losses": losses,
        "paired_table": {
            "both_correct": sum(v["control"] and v["candidate"] for _, _, v in paired),
            "candidate_only_correct": wins,
            "control_only_correct": losses,
            "both_incorrect": sum(not v["control"] and not v["candidate"] for _, _, v in paired),
        },
        "paired_cluster_p_two_sided": cluster_p,
        "probe_level_mcnemar_p_descriptive": binomial_p,
        "namespace_bootstrap_95_percentile_delta": [boot[249], boot[9749]],
        "safety": safety,
        "safety_pass": safety_pass,
        "candidate_count_cited": cited,
        "candidate_count_rows": 90,
        "mechanism_observed": mechanism,
        # Registered alongside the binary gate, never in place of it. The binary outcome
        # above is the promotion decision; this is the higher-resolution reading of the
        # mechanism, and a candidate that clears this while failing the gate has shown
        # that it does what it claims and not that it is worth promoting.
        "enumeration_completeness": (
            uncited_gold_facts(rows, probes, memories) if memories else None
        ),
        "limitations": [
            "Development SQL-derived probes, not new independent benchmark questions.",
            "Cluster sign-flip assumes exchangeable arm outcomes between independent namespaces.",
            "Three repeats do not create 90 independent count questions.",
            "The binary gate promotes a true six-probe gain about 31% of the time against "
            "the measured pair churn; 'inconclusive' is the expected outcome, not a "
            "surprise, and enumeration_completeness is the reading that resolves at this "
            "sample size.",
            "enumeration_completeness decides nothing on its own. It is evidence about "
            "the mechanism, not a promotion criterion.",
        ],
    }
