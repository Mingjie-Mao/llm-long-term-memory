"""Verify the completed frozen t3/t4 comparison without reading item failures."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
from analysis_io import rows_by_question, sha256_file  # noqa: E402
from raw_primary_gate_t3t4 import T3, V1, V2, evaluate  # noqa: E402
from recursive_summary_report import model_workload  # noqa: E402
from run_paged_paired import save_json  # noqa: E402


def audit():
    from llm_long_term_memory.llm.usage import UsageTracker

    prereg = REPO / "results/prereg-raw-primary-t3-t4-heldout100-v1.md"
    registered = dict(
        re.findall(r"\| `([^`]+)` \| `([0-9a-f]{16})` \|", prereg.read_text(encoding="utf-8"))
    )
    if len(registered) != 11:
        raise ValueError("registration inventory incomplete")
    changed = [p for p, h in registered.items() if sha256_file(REPO / p)[:16] != h]
    if changed:
        raise ValueError(f"registered inputs changed: {changed}")
    manifest = REPO / "results/manifests/heldout100.json"
    ids = set(json.loads(manifest.read_text(encoding="utf-8"))["question_ids"])
    if len(ids) != 100:
        raise ValueError("manifest no longer has 100 unique questions")
    inputs, resources, prompts = {}, {}, {}
    for arm, names in (("v1", V1), ("t3", T3), ("t4", V2)):
        records = []
        for name in names:
            rp, up = [REPO / f"results/raw/{name}.{s}" for s in ("jsonl", "usage.json")]
            rows = rows_by_question(rp)
            if set(rows) != ids or any(type(r["correct"]) is not bool for r in rows.values()):
                raise ValueError("incomplete/foreign/invalid rows")
            versions = sorted({r["answer_prompt_version"] for r in rows.values()})
            expected = "memory-aware-v2" if arm == "v1" else "memory-aware-v2-clt"
            if versions != [expected]:
                raise ValueError("answer prompt identity mismatch")
            prompts[name] = versions
            loaded = UsageTracker._load_records(up)
            if any(r.model not in {"gemini-3.5-flash-lite", "gemma-4-31b-it"} for r in loaded):
                raise ValueError("model changed")
            records.extend(loaded)
            inputs.update({str(p.relative_to(REPO)): sha256_file(p) for p in (rp, up)})
        resources[arm] = model_workload(records)
    gate_path = REPO / "results/analysis/raw-primary-t3-t4-heldout100-gate.json"
    recomputed = evaluate()
    if recomputed != json.loads(gate_path.read_text(encoding="utf-8")):
        raise ValueError("stored gate does not reproduce")
    inputs.update({str(p.relative_to(REPO)): sha256_file(p) for p in (prereg, gate_path, manifest)})
    return {
        "provider_calls": 0,
        "experiment_class": "exposed-heldout100-regression-audit",
        "new_answers": 900,
        "grades": 900,
        "registered_inputs_current": True,
        "gate_reproduces": True,
        "gate": recomputed,
        "resources_by_model": resources,
        "prompt_versions": prompts,
        "inputs": inputs,
        "generator_sha256": sha256_file(Path(__file__)),
        "limitations": "Not unseen-final; no item failures inspected; no significance claim.",
    }


if __name__ == "__main__":
    value = audit()
    out = REPO / "results/analysis/t3t4-completion-audit-v1.json"
    if out.exists() and json.loads(out.read_text(encoding="utf-8")) != value:
        raise ValueError("audit exists with different evidence")
    if not out.exists():
        save_json(out, value)
    print(
        json.dumps(
            {
                k: value[k]
                for k in (
                    "provider_calls",
                    "new_answers",
                    "grades",
                    "registered_inputs_current",
                    "gate_reproduces",
                )
            }
        )
    )
