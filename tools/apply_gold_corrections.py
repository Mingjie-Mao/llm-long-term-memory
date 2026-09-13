"""Re-read every count answer already paid for, against the corrected gold.

Steps 2-4 of fixing the instrument. Zero provider calls: every row exists.

**The probe set is never edited.** `synthesis-probes.json` is hashed into
`v4-probe-split.json`, the only unseen check v4 has, so the corrections live in their own
overlay and are applied at read time. A corrected answer is the original minus the members
the adjudication excluded.

Two readings come out of this, and they answer different questions:

- **accuracy against corrected gold** — how many count answers were right all along, and
  were scored wrong because the gold counted intentions as members.
- **corrected enumeration completeness** — of the members that genuinely belong, how many
  were in context and never named. This is the real reading gap, and it is the number that
  says whether any answerer change is still needed.

The adjudication was frozen before any of this was computed, and it is model-made and
rule-tagged rather than human-made. Both facts are carried into the output, because a
corrected score is only as trustworthy as the adjudication behind it.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CORRECTIONS = REPO / "results/analysis/count-probe-gold-corrections.json"
PROBES = REPO / "results/analysis/synthesis-probes.json"

# Every run whose count rows were paid for, and the arm label to report it under.
RUNS = {
    "control-v3.3": "results/archive/v4.0-flat/probes.v3.3-dev.jsonl",
    "v4.0-flat": "results/archive/v4.0-flat/probes.v4.0-flat.jsonl",
    "v4.0-flat2": "results/raw/probes.v4.0-flat2.jsonl",
    "v4.1-scan": "results/raw/probes.v4.1-scan.jsonl",
}
# The v4.2 run interleaves both arms in one file and repeats count probes three times.
V42 = "results/frozen/v4.2-development-20260912e/live/rows.jsonl"


def as_int(value: object) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def corrected_gold(probes: dict, excluded: dict[str, set[str]]) -> dict[str, dict]:
    """Per count probe: the original answer, the corrected one, and the surviving members."""
    out = {}
    for pid, drop in excluded.items():
        if pid not in probes or probes[pid]["kind"] != "count":
            raise ValueError("exclusion targets an unknown or non-count probe")
        if not drop <= set(probes[pid]["evidence_memory_ids"]):
            raise ValueError("exclusion targets a nonmember")
    for pid, probe in probes.items():
        if probe["kind"] != "count":
            continue
        drop = excluded.get(pid, set())
        evidence = probe["evidence_memory_ids"]
        if len(evidence) != len(set(evidence)) or int(probe["answer"]) != len(evidence):
            raise ValueError("original gold is not a unique memory-row count")
        members = [g for g in probe["evidence_memory_ids"] if g not in drop]
        out[pid] = {
            "original": int(probe["answer"]),
            "corrected": len(members),
            "members": members,
            "dropped": len(drop),
        }
    return out


def validate_overlay(adjudication: dict, probes: dict, memories: dict) -> dict[str, set[str]]:
    """A proposed overlay cannot reach the holdout or silently use stale evidence."""
    seen = set()
    excluded: dict[str, set[str]] = {}
    for member in adjudication["members"]:
        pid, mid = member["probe_id"], member["memory_id"]
        probe = probes.get(pid)
        if probe is None or probe["kind"] != "count":
            raise ValueError("overlay must target development count probes only")
        if (pid, mid) in seen or mid not in probe["evidence_memory_ids"]:
            raise ValueError("duplicate overlay entry or unknown gold member")
        seen.add((pid, mid))
        if member["question"] != probe["question"] or member["memory"] != memories[mid]["content"]:
            raise ValueError("overlay question or memory evidence changed")
        verdict, rule = member["verdict"], member["rule"]
        if verdict not in {"keep", "exclude"} or rule not in adjudication["rules"]:
            raise ValueError("invalid adjudication verdict/rule")
        if (rule in {"R4", "R5"}) != (verdict == "keep"):
            raise ValueError("adjudication rule contradicts its verdict")
        if member["confidence"] not in {"high", "medium", "disputed"}:
            raise ValueError("invalid adjudication confidence")
        if verdict == "exclude":
            excluded.setdefault(pid, set()).add(mid)
    counts = Counter(m["verdict"] for m in adjudication["members"])
    if adjudication["counts"]["adjudicated"] != len(seen) or any(
        adjudication["counts"][k] != counts[k] for k in ("keep", "exclude")
    ):
        raise ValueError("adjudication counts disagree with its entries")
    return excluded


def validate_count_rows(rows: list[dict], probes: dict, repeats: int) -> None:
    expected = {pid for pid, p in probes.items() if p["kind"] == "count"}
    observed = defaultdict(list)
    for row in rows:
        if row["kind"] != "count":
            continue
        pid = row["probe_id"]
        if pid not in expected:
            raise ValueError("count rows contain an unknown or non-development probe")
        probe = probes[pid]
        if any(
            row[k] != probe[p]
            for k, p in (("question", "question"), ("namespace", "namespace"), ("gold", "answer"))
        ):
            raise ValueError("count row identity differs from its probe")
        observed[pid].append(row.get("repeat", 1))
    if set(observed) != expected or any(
        sorted(v) != list(range(1, repeats + 1)) for v in observed.values()
    ):
        raise ValueError("incomplete or duplicate count repeats; no corrected comparison")


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, help="must match the executed frozen database")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "results/analysis/count-gold-corrected-reread-20260913.json",
    )
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from diagnose_v4_probes import _gold_texts, _match
    from probe_safety import select_probes
    from v42_protocol import exact_cluster_p, sha, validate_rows

    if not CORRECTIONS.is_file():
        print(f"STOP: no adjudication at {CORRECTIONS}")
        return 2
    adjudication = json.loads(CORRECTIONS.read_text(encoding="utf-8"))
    raw = PROBES.read_bytes()
    split = REPO / "results/manifests/v4-probe-split.json"
    chosen = select_probes(json.loads(raw), raw, split, "development")
    probes = {p["probe_id"]: p for p in chosen}
    freeze_root = (REPO / V42).parents[1]
    freeze_path = freeze_root / "freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    store = args.store or Path(freeze["private_data"]) / "train150.db"
    expected_store = next(f["sha256"] for f in freeze["data"] if f["path"] == "train150.db")
    if sha(store) != expected_store:
        raise ValueError("store differs from the executed frozen database")
    memories = _gold_texts(store)
    excluded = validate_overlay(adjudication, probes, memories)
    gold = corrected_gold(probes, excluded)

    def read(path: Path) -> list[dict]:
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    # Build the arm -> rows map, splitting the v4.2 file by arm.
    arms: dict[str, list[dict]] = {}
    for name, relative in RUNS.items():
        path = REPO / relative
        arms[name] = read(path)
        if len(arms[name]) != len(chosen):
            raise ValueError("incomplete historical development run")
        validate_count_rows(arms[name], probes, 1)
    v42 = REPO / V42
    rows = read(v42)
    validated = validate_rows(rows, freeze["cells"], chosen, sha(freeze_path), "live")
    if len(validated) != len(freeze["cells"]):
        raise ValueError("v4.2 is incomplete; no corrected comparison")
    for arm in ("control", "candidate"):
        arms[f"v4.2-{arm}"] = [r for r in rows if r["arm"] == arm]
        validate_count_rows(arms[f"v4.2-{arm}"], probes, 3)
    conclusion = json.loads((freeze_root / "live/conclusion.json").read_text(encoding="utf-8"))
    for artifact in conclusion["artifacts"]:
        if sha(freeze_root / artifact["path"]) != artifact["sha256"]:
            raise ValueError("executed v4.2 evidence hash mismatch")

    report = {}
    corrected_outcomes = {}
    for name, rows in arms.items():
        counts = [r for r in rows if r["kind"] == "count" and r["probe_id"] in gold]
        by_probe_original: dict[str, list[bool]] = {}
        by_probe_corrected: dict[str, list[bool]] = {}
        unnamed_original: dict[str, list[int]] = {}
        unnamed_corrected: dict[str, list[int]] = {}
        for row in counts:
            pid = row["probe_id"]
            spec = gold[pid]
            parsed = as_int(row.get("parsed"))
            items = (row.get("synthesis_computation") or {}).get("items") or []
            retrieved = set(row.get("retrieved_ids") or [])
            by_probe_original.setdefault(pid, []).append(parsed == spec["original"])
            by_probe_corrected.setdefault(pid, []).append(parsed == spec["corrected"])
            for pool, sink in (
                (probes[pid]["evidence_memory_ids"], unnamed_original),
                (spec["members"], unnamed_corrected),
            ):
                named = {g for g in (_match(i, pool, memories) for i in items) if g}
                sink.setdefault(pid, []).append(
                    sum(1 for g in pool if g not in named and g in retrieved)
                )

        def majority(table: dict[str, list[bool]]) -> int:
            return sum(sum(v) * 2 >= len(v) + (len(v) % 2 == 0) for v in table.values())

        def mean_total(table: dict[str, list[int]]) -> float:
            return round(sum(sum(v) / len(v) for v in table.values()), 2)

        report[name] = {
            "probes": len(by_probe_original),
            "repeats_per_probe": round(len(counts) / max(len(by_probe_original), 1), 2),
            "correct_against_original_gold": majority(by_probe_original),
            "correct_against_corrected_gold": majority(by_probe_corrected),
            "unnamed_members_original_gold": mean_total(unnamed_original),
            "unnamed_members_corrected_gold": mean_total(unnamed_corrected),
        }
        corrected_outcomes[name] = {
            pid: sum(v) * 2 > len(v) for pid, v in by_probe_corrected.items()
        }

    dropped = sum(len(v) for v in excluded.values())
    changed = [p for p, g in gold.items() if g["corrected"] != g["original"]]
    deltas = defaultdict(int)
    pairs = Counter()
    for pid in gold:
        control = corrected_outcomes["v4.2-control"][pid]
        candidate = corrected_outcomes["v4.2-candidate"][pid]
        pairs[(control, candidate)] += 1
        deltas[probes[pid]["namespace"]] += int(candidate) - int(control)
    input_paths = [
        CORRECTIONS,
        PROBES,
        split,
        freeze_path,
        freeze_root / "live/conclusion.json",
        REPO / V42,
        *(REPO / r for r in RUNS.values()),
        Path(__file__),
        REPO / "tools/diagnose_v4_probes.py",
        REPO / "tools/v42_protocol.py",
    ]
    payload = {
        "schema_version": 1,
        "provider_calls": 0,
        "scope": "development_only",
        "review_status": "provisional_model_overlay_pending_human_review",
        "counting_unit": "memory rows, not fully validated distinct entities",
        "inputs": [{"path": str(p.relative_to(REPO)), "sha256": sha(p)} for p in input_paths],
        "frozen_store_sha256": expected_store,
        "v42_corrected_paired_comparison": {
            "net": sum(deltas.values()),
            "candidate_only_correct": pairs[(False, True)],
            "control_only_correct": pairs[(True, False)],
            "both_correct": pairs[(True, True)],
            "both_incorrect": pairs[(False, False)],
            "namespaces": len(deltas),
            "cluster_p_two_sided_descriptive": exact_cluster_p(list(deltas.values())),
            "registered_conclusion_recomputed": False,
        },
        "probe_set_never_edited": (
            "synthesis-probes.json is hashed into v4-probe-split.json, so corrections are "
            "an overlay applied at read time."
        ),
        "adjudication": {
            "file": str(CORRECTIONS.relative_to(REPO)),
            "by": adjudication["adjudicated_by"],
            "counts": adjudication["counts"],
        },
        "gold": {
            "count_probes": len(gold),
            "probes_changed": len(changed),
            "members_excluded": dropped,
            "members_before": sum(g["original"] for g in gold.values()),
            "members_after": sum(g["corrected"] for g in gold.values()),
        },
        "per_run": report,
        "per_probe_gold": {
            p: {"original": g["original"], "corrected": g["corrected"]}
            for p, g in sorted(gold.items())
            if g["corrected"] != g["original"]
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    g = payload["gold"]
    print(f"adjudication: {adjudication['counts']}")
    print(
        f"gold: {g['probes_changed']}/{g['count_probes']} probes changed, "
        f"{g['members_before']} -> {g['members_after']} members"
    )
    print()
    header = (
        f"{'run':16s} {'orig':>6s} {'corr':>6s} {'delta':>6s} | {'unnamed orig':>13s} {'corr':>7s}"
    )
    print(header)
    print("-" * len(header))
    for name, r in report.items():
        o, c = r["correct_against_original_gold"], r["correct_against_corrected_gold"]
        print(
            f"{name:16s} {o:6d} {c:6d} {c - o:+6d} | "
            f"{r['unnamed_members_original_gold']:13.1f} {r['unnamed_members_corrected_gold']:7.1f}"
        )
    try:
        shown = args.out.relative_to(REPO)
    except ValueError:  # --out pointed outside the repository
        shown = args.out
    print(f"\nwrote {shown}")
    return 0


def main() -> int:
    try:
        return _main()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"STOP: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
