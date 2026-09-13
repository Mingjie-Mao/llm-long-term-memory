"""Offline validation before a synthesis run can construct a provider client."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def select_probes(spec: dict, probe_bytes: bytes, split_path: Path, half: str) -> list[dict]:
    if not split_path.is_file():
        raise ValueError("split manifest is missing; refusing to select all probes")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if (
        split.get("probe_set_sha256") != hashlib.sha256(probe_bytes).hexdigest()
        or split.get("store_fingerprint") != spec["store_fingerprint"]
    ):
        raise ValueError("split manifest does not match the probe set/store")
    ids = [p["probe_id"] for p in spec["probes"]]
    dev, held = split["development"], split["held_out"]
    if (
        len(ids) != len(set(ids))
        or len(dev) != len(set(dev))
        or len(held) != len(set(held))
        or set(dev) & set(held)
        or set(dev) | set(held) != set(ids)
        or not dev
        or not held
    ):
        raise ValueError("split must be nonempty, disjoint and cover every probe exactly once")
    wanted = set(ids if half == "all" else split[half])
    return [p for p in spec["probes"] if p["probe_id"] in wanted]


def run_identity(repo: Path, config: Path, variant: str, probe_bytes: bytes, half: str) -> dict:
    source = hashlib.sha256()
    paths = [
        *sorted((repo / "src").rglob("*.py")),
        repo / "tools/run_synthesis_probes.py",
        repo / "tools/probe_safety.py",
        repo / "results/analysis/predicate-map.csv",
    ]
    for path in paths:
        source.update(str(path.relative_to(repo)).encode())
        source.update(b"\0")
        source.update(path.read_bytes())
    return {
        "variant": variant,
        "half": half,
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "probe_set_sha256": hashlib.sha256(probe_bytes).hexdigest(),
        "source_sha256": source.hexdigest(),
    }


def resume_ids(path: Path, probes: list[dict], identity: dict) -> set[str]:
    if not path.exists():
        return set()
    by_id = {p["probe_id"]: p for p in probes}
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        probe = by_id.get(row.get("probe_id"))
        if probe is None or row.get("question") != probe["question"]:
            raise ValueError("existing rows belong to a different probe set or half")
        if row["probe_id"] in done:
            raise ValueError("existing rows contain duplicate probe ids")
        if row.get("run_identity") != identity or row.get("variant") != identity["variant"]:
            raise ValueError(
                "existing rows have different or unrecorded run provenance; archive them"
            )
        done.add(row["probe_id"])
    return done


def claim_holdout(path: Path, output: Path, identity: dict) -> dict:
    """One global ledger, independent of labels/output directories; resume the same run."""
    expected = {"output": str(output.resolve()), "run_identity": identity}
    if path.exists():
        ledger = json.loads(path.read_text(encoding="utf-8"))
        if ledger.get("status") == "complete":
            raise ValueError("held-out run is already complete; the one shot is spent")
        if any(ledger.get(k) != v for k, v in expected.items()) or not output.exists():
            raise ValueError(
                "a different held-out run already exists; only its exact resume is allowed"
            )
        return ledger
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger = {**expected, "status": "in_progress"}
    # Exclusive creation also prevents two differently labelled jobs claiming it.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(ledger, handle, indent=2)
        handle.write("\n")
    return ledger


def complete_holdout(path: Path, ledger: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({**ledger, "status": "complete"}, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
