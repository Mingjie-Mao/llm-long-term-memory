"""Restore content-verified historical v2 sources for an exposed test100 regression.

No models, questions, answers or original artifacts are executed or rewritten here.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from analysis_io import sha256_file  # noqa: E402
from t4_savings_regression import immutable_json  # noqa: E402

STEM = "test100-v2-source-restore-v1"
ROOT = REPO / "stores/experiments" / STEM


def write_verified(path, payload, digest):
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("restored source differs from captured hash")
    if path.exists():
        if sha256_file(path) != digest:
            raise ValueError(f"existing restored source changed: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def main():
    frozen = REPO / "results/frozen/v2-final/freeze.json"
    system = json.loads(frozen.read_text(encoding="utf-8"))["system"]
    reference = system["git_head_for_reference_only"]
    restored = {}
    for rel, receipt in system["source"]["files"].items():
        if Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise ValueError("unsafe historical source path")
        refs = [reference]
        payload = subprocess.check_output(["git", "show", reference + ":" + rel], cwd=REPO)
        if hashlib.sha256(payload).hexdigest() != receipt["sha256"]:
            refs = subprocess.check_output(
                ["git", "log", "--format=%H", "--all", "--", rel], cwd=REPO, text=True
            ).splitlines()
            for ref in refs:
                result = subprocess.run(
                    ["git", "show", ref + ":" + rel], cwd=REPO, capture_output=True
                )
                if (
                    result.returncode == 0
                    and hashlib.sha256(result.stdout).hexdigest() == receipt["sha256"]
                ):
                    payload, refs = result.stdout, [ref]
                    break
        write_verified(ROOT / rel, payload, receipt["sha256"])
        restored[rel] = {"sha256": receipt["sha256"], "git_ref": refs[0]}
    config = yaml.safe_dump(system["config"]["resolved"], sort_keys=True).encode()
    write_verified(ROOT / "captured-v2.yaml", config, hashlib.sha256(config).hexdigest())
    result = {
        "provider_calls": 0,
        "experiment_class": "exposed-test100-regression-preparation",
        "historical_freeze_sha256": sha256_file(frozen),
        "generator_sha256": sha256_file(Path(__file__)),
        "root": str(ROOT.relative_to(REPO)),
        "restored_files": restored,
        "config_sha256": sha256_file(ROOT / "captured-v2.yaml"),
        "models": system["models"],
        "limitations": "Source restoration only. Baseline loading, original data identity, "
        "new registration and fresh same-question paired comparison still required.",
    }
    immutable_json(REPO / f"results/analysis/{STEM}.json", result)
    print(
        json.dumps({"provider_calls": 0, "restored_files": len(restored), "root": result["root"]})
    )


if __name__ == "__main__":
    main()
