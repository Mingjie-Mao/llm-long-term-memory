"""Zero-call resource ceiling from original USER history, without dev item diagnosis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
from analysis_io import sha256_file  # noqa: E402
from recursive_summary_nav import leaf_records  # noqa: E402
from run_paged_paired import expected_calls, save_json  # noqa: E402
from run_recursive_summary_cost import PILOT, STEM, VARIANT, instances_for  # noqa: E402


def parent_count(leaves):
    count = 0
    while leaves > 1:
        count += leaves // 4 + (leaves % 4 > 1)
        leaves = (leaves + 3) // 4
    return count


def estimate():
    from llm_long_term_memory import cli
    from llm_long_term_memory.runtime.grounded_answering import personal_archive_review

    result = {
        "provider_calls": 0,
        "experiment_class": "exposed-split-offline-resource-preflight",
        "pilot_questions": len(PILOT),
        "inputs": {},
        "by_split": {},
    }
    for split in ("train150", "dev100"):
        _, _, runner, _, _ = cli._build(
            VARIANT, "configs/fallback.yaml", split, client_override=object(), read_only_store=True
        )
        try:
            instances = instances_for(split)
            users = {i.store_namespace for i in instances if personal_archive_review(i.question)}
            leaf_counts = [len(leaf_records(runner.store, user)) for user in users]
            nodes = sum(n + parent_count(n) for n in leaf_counts)
            result["by_split"][split] = {
                "questions": len(instances),
                "personal_histories": len(users),
                "leaf_summary_calls": sum(leaf_counts),
                "parent_summary_calls": nodes - sum(leaf_counts),
                "total_nominal_initial_summary_calls": nodes,
                "max_leaf_count": max(leaf_counts, default=0),
                # Two durable invocations, each with two API-error attempts plus
                # one additional transport/TPM attempt (client counters differ).
                "summary_call_attempt_ceiling": nodes * 6,
                "query_call_upper_before_client_retries": (
                    sum(14 if personal_archive_review(i.question) else 2 for i in instances)
                    * (1 if split == "train150" else 3)
                ),
            }
            repeats = 1 if split == "train150" else 3
            candidate = result["by_split"][split]["query_call_upper_before_client_retries"]
            control = (
                sum(expected_calls(runner.store, i, True) // 2 for i in instances) * repeats
                if split == "dev100"
                else 0
            )
            result["by_split"][split]["query_call_upper_by_arm"] = {
                "control": control,
                "candidate": candidate,
            }
            result["by_split"][split]["query_attempt_ceiling_both_arms"] = 3 * (control + candidate)
            for p in (REPO / f"stores/{split}.db", REPO / f"results/manifests/{split}.json"):
                result["inputs"][str(p.relative_to(REPO))] = sha256_file(p)
        finally:
            runner.store.close()
    result["inputs"]["tools/recursive_summary_preflight.py"] = sha256_file(Path(__file__))
    return result


def main():
    out = REPO / f"results/analysis/{STEM}.preflight-r2.json"
    value = estimate()
    if out.exists() and json.loads(out.read_text(encoding="utf-8")) != value:
        raise ValueError("preflight changed; use a new namespace")
    if not out.exists():
        save_json(out, value)
    print(json.dumps(value))


if __name__ == "__main__":
    main()
