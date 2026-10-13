"""Zero-provider-call sizing for a future t4 recursive-navigation comparison.

This is resource planning only: it neither builds summaries nor reads answer golds,
grades, or failed questions. It does not use the failed v18 summary driver.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
from analysis_io import sha256_file  # noqa: E402
from recursive_summary_nav import SUMMARY_SYSTEM  # noqa: E402
from recursive_summary_preflight import parent_count  # noqa: E402
from run_recursive_summary_cost import instances_for  # noqa: E402
from t4_savings_regression import immutable_json  # noqa: E402

STEM = "t4-recursive-summary-feasibility-v1"


def size_histories(store, users, roles):
    from llm_long_term_memory.evaluation.grounded.evidence_pages import archive_pages

    counts, leaf_input, capped = [], 0, 0
    for user in sorted(users):
        archive = archive_pages(store, user, max_tokens=4000, roles=roles, split_oversized=True)
        counts.append(len(archive.pages))
        capped += not archive.complete
        leaf_input += sum(
            int((len(page.render()) + len(SUMMARY_SYSTEM)) / 4.6) for page in archive.pages
        )
        for page in archive.pages:
            for source in page.sources:
                origin = store.get_session(source.session_id)
                if origin is None or origin.user_id != user:
                    raise ValueError("foreign source in summary sizing")
    leaves, parents = sum(counts), sum(parent_count(n) for n in counts)
    return {
        "role_scope": list(roles),
        "histories": len(users),
        "histories_with_omitted_raw_turns_at_32_page_cap": capped,
        "complete_build_sized": capped == 0,
        "leaf_pages": leaves,
        "parent_nodes": parents,
        "nominal_build_generations_for_included_pages": leaves + parents,
        "median_leaf_pages": statistics.median(counts) if counts else 0,
        "max_leaf_pages": max(counts, default=0),
        "approx_leaf_input_tokens": leaf_input,
        "summary_output_token_ceiling_for_included_pages": 512 * (leaves + parents),
        "histories_over_navigation_leaf_selection_cap_8": sum(n > 8 for n in counts),
        "limits": "Input estimate uses characters/4.6. Parent input, retries, updates, "
        "navigation and answers are excluded; this is not measured total cost. "
        "More than eight available leaves does not imply a query needs more than eight.",
    }


def estimate():
    from llm_long_term_memory import cli

    result = {
        "provider_calls": 0,
        "experiment_class": "exposed-split-offline-resource-planning",
        "baseline": "t4 scoped-selection repair, subject to full regression passing",
        "summary_calls_authorized_by_this_tool": 0,
        "by_split": {},
        "inputs": {},
    }
    for split in ("train150", "dev100"):
        _, _, runner, _, _ = cli._build(
            "two_stage_raw_primary_t4",
            "configs/fallback.yaml",
            split,
            client_override=object(),
            read_only_store=True,
        )
        try:
            instances = instances_for(split)
            users = {i.store_namespace for i in instances}
            result["by_split"][split] = {
                "questions": len(instances),
                "scope": "fixed21 known pilot" if split == "train150" else "all dev100",
                "user_only_recipe": size_histories(runner.store, users, ("user",)),
                "both_roles_sizing_only": size_histories(
                    runner.store, users, ("user", "assistant")
                ),
            }
        finally:
            runner.store.close()
        for rel in (f"stores/{split}.db", f"results/manifests/{split}.json"):
            result["inputs"][rel] = sha256_file(REPO / rel)
    for rel in (
        "tools/t4_recursive_summary_feasibility.py",
        "tools/recursive_summary_nav.py",
        "tools/recursive_summary_preflight.py",
        "src/llm_long_term_memory/runtime/evidence_pages.py",
        "configs/fallback.yaml",
    ):
        result["inputs"][rel] = sha256_file(REPO / rel)
    result["requirements_for_later_protocol"] = [
        "A passing complete t4 fresh dev100 regression is required before summary calls.",
        "Keep assistant originals available for questions about assistant artifacts.",
        "Summaries navigate; only tenant-owned original sources support factual answers.",
        "Measure build/update/navigation/answer failures and tokens separately per model.",
        "Report cold one-query cost and amortization at 1/3/10/50/100 diverse queries.",
        "Fresh paired accuracy and faithful-gap checks are required; sizing proves neither.",
    ]
    return result


def main():
    result = estimate()
    immutable_json(REPO / f"results/analysis/{STEM}.json", result)
    print(json.dumps({k: result[k] for k in ("provider_calls", "by_split")}))


if __name__ == "__main__":
    main()
