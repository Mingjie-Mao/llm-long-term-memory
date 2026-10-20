"""Publish the audited regression beside the unchanged historical release result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STEM = "original-test100-v2-t4-dispatch-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regression_figures(root: Path) -> dict:
    gate_path = root / f"results/analysis/{STEM}.gate.json"
    audit_path = root / f"results/analysis/{STEM}.completion-audit-v1.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        gate.get("pass") is not True
        or audit.get("gate_reproduces") is not True
        or gate.get("experiment_class") != "exposed-original-test100-regression"
        or audit.get("provider_calls") != 0
        or gate.get("inventory_sha256") != audit.get("inventory_sha256")
        or gate.get("questions") != 100
        or audit.get("questions") != 100
        or any(x.get(k) != 600 for x in (gate, audit) for k in ("reader_rows", "grade_rows"))
    ):
        raise ValueError("complete, passing, source-bound regression audit required")
    inputs = audit["inputs"]
    if inputs.get(gate_path.relative_to(root).as_posix()) != sha256(gate_path):
        raise ValueError("gate is not bound to the completion audit")
    for name, digest in inputs.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or sha256(path) != digest:
            raise ValueError(f"audited input changed: {name}")
    counts = gate["accuracy"]["per_run_correct"]
    means = gate["accuracy_percent"]
    for arm in ("control", "candidate"):
        values = counts[arm]
        if (
            len(values) != 3
            or any(type(n) is not int or not 0 <= n <= 100 for n in values)
            or not math.isclose(sum(values) / 3, means[arm], abs_tol=1e-9)
        ):
            raise ValueError("three-run accuracy does not match the registered gate")
    net = means["candidate"] - means["control"]
    if not math.isclose(net, gate["accuracy"]["mean_net"], abs_tol=1e-9):
        raise ValueError("paired mean difference does not match")
    tokens = {
        arm: sum(c["input_tokens"] + c["output_tokens"] for c in models.values())
        for arm, models in gate["resources_by_arm_and_model"].items()
    }
    return {
        "experiment": STEM,
        "evidence_class": "exposed-regression",
        "questions": 100,
        "repeats_per_arm": 3,
        "control_correct": counts["control"],
        "candidate_correct": counts["candidate"],
        "control_accuracy": means["control"],
        "candidate_accuracy": means["candidate"],
        "mean_difference_pp": net,
        "control_recorded_tokens": tokens["control"],
        "candidate_recorded_tokens": tokens["candidate"],
        "reader_rows": 600,
        "grade_rows": 600,
        "gate_passed": True,
        "gate_path": gate_path.relative_to(root).as_posix(),
        "gate_sha256": sha256(gate_path),
        "audit_path": audit_path.relative_to(root).as_posix(),
        "audit_sha256": sha256(audit_path),
    }


def update(root: Path, *, check: bool = False) -> None:
    path = root / "public-demo/site/release.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = regression_figures(root)
    if check:
        if manifest.get("regression") != expected:
            raise ValueError("public regression figures differ from the audited result")
        return
    manifest["regression"] = expected
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    update(REPO, check=args.check)
    print(
        "public regression figures verified" if args.check else "public regression figures updated"
    )
