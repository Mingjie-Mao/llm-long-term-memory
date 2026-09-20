"""Analyse the v2b gate16 QA rows against the frozen three-run control."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))

from analysis_io import read_json, rows_by_question  # noqa: E402

from llm_long_term_memory.stats import paired_outcomes  # noqa: E402

MANIFEST = REPO / "results/manifests/v2b-gate16.json"
REPEAT_MANIFEST = REPO / "results/manifests/v2b-gate16-discordant.json"
OUT_JSON = REPO / "results/analysis/v2b-gate16-qa.json"
OUT_MD = REPO / "results/analysis/v2b-gate16-qa.md"
BASELINES = [
    REPO / f"results/sealed/v3-answer-pilot/v2-control.rep{repeat}.jsonl" for repeat in (1, 2, 3)
]
CANDIDATES = {
    "memory_only": REPO / "results/raw/two_stage_memory_only.v2b-gate16.jsonl",
    "final": REPO / "results/raw/two_stage_fallback.v2b-gate16.jsonl",
}
REPEATS = {
    "memory_only": REPO / "results/raw/two_stage_memory_only.v2b-gate16-repeat2.jsonl",
    "final": REPO / "results/raw/two_stage_fallback.v2b-gate16-repeat2.jsonl",
}


def _paired(old: dict[str, bool], new: dict[str, bool], ids: list[str]) -> dict:
    return paired_outcomes(old, new, ids)


def analyse() -> dict:
    manifest = read_json(MANIFEST)
    ids = manifest["question_ids"]
    baselines = [rows_by_question(path) for path in BASELINES]
    candidates = {name: rows_by_question(path) for name, path in CANDIDATES.items()}

    baseline_final = {qid: sum(bool(run[qid]["correct"]) for run in baselines) >= 2 for qid in ids}
    # This is the archived project's operational memory-only endpoint: a correct
    # final row counts only when no raw fallback was used. It is a proxy because
    # the old runs did not persist their pre-fallback draft answer.
    baseline_memory_proxy = {
        qid: sum(
            bool(run[qid]["correct"]) and run[qid]["notes"].get("fallback_level", "none") == "none"
            for run in baselines
        )
        >= 2
        for qid in ids
    }
    candidate_correct = {
        name: {qid: bool(rows[qid]["correct"]) for qid in ids} for name, rows in candidates.items()
    }
    comparisons = {
        "memory_only": _paired(baseline_memory_proxy, candidate_correct["memory_only"], ids),
        "final": _paired(baseline_final, candidate_correct["final"], ids),
    }

    repeat_ids = [
        qid
        for qid in ids
        if qid
        in {
            *comparisons["memory_only"]["wins"],
            *comparisons["memory_only"]["losses"],
            *comparisons["final"]["wins"],
            *comparisons["final"]["losses"],
        }
    ]
    repeat_manifest = {
        "name": "v2b-gate16-discordant",
        "variant": manifest["variant"],
        "seed": manifest["seed"],
        "n": len(repeat_ids),
        "note": (
            "Post-gate diagnostic repeat containing only rows discordant with the frozen "
            "three-run batch15 control on either the memory-only proxy or final endpoint. "
            "It must not be interpreted as an accuracy sample."
        ),
        "question_ids": repeat_ids,
    }

    raw_effect = []
    for qid in ids:
        final_row = candidates["final"][qid]
        level = final_row["notes"].get("fallback_level", "none")
        if level != "none":
            raw_effect.append(
                {
                    "question_id": qid,
                    "fallback_level": level,
                    "memory_only_correct": candidate_correct["memory_only"][qid],
                    "final_correct": candidate_correct["final"][qid],
                }
            )

    result = {
        "questions": len(ids),
        "targeted_gate_not_accuracy_estimate": True,
        "baseline": {
            "final_majority_correct": sum(baseline_final.values()),
            "memory_only_proxy_majority_correct": sum(baseline_memory_proxy.values()),
            "memory_only_proxy_caveat": (
                "Correct final answers with fallback_level=none; pre-fallback drafts "
                "were not saved."
            ),
        },
        "comparisons": comparisons,
        "raw_fallback_effect": raw_effect,
        "repeat_ids": repeat_ids,
        "rows": [
            {
                "question_id": qid,
                "question_type": candidates["final"][qid]["question_type"],
                "baseline_memory_proxy": baseline_memory_proxy[qid],
                "baseline_final": baseline_final[qid],
                "candidate_memory_only": candidate_correct["memory_only"][qid],
                "candidate_final": candidate_correct["final"][qid],
                "fallback_level": candidates["final"][qid]["notes"].get("fallback_level", "none"),
            }
            for qid in ids
        ],
    }

    if all(path.exists() for path in REPEATS.values()):
        repeat_rows = {name: rows_by_question(path) for name, path in REPEATS.items()}
        repeat_comparisons = {
            "memory_only": _paired(
                baseline_memory_proxy,
                {qid: bool(repeat_rows["memory_only"][qid]["correct"]) for qid in repeat_ids},
                repeat_ids,
            ),
            "final": _paired(
                baseline_final,
                {qid: bool(repeat_rows["final"][qid]["correct"]) for qid in repeat_ids},
                repeat_ids,
            ),
        }
        result["repeat_comparisons"] = repeat_comparisons
        result["repeat_stability"] = {
            name: {
                "agreements": sum(
                    bool(candidates[name][qid]["correct"])
                    == bool(repeat_rows[name][qid]["correct"])
                    for qid in repeat_ids
                ),
                "n": len(repeat_ids),
            }
            for name in CANDIDATES
        }
        memory_net = repeat_comparisons["memory_only"]["net"]
        final_net = repeat_comparisons["final"]["net"]
        result["promotion"] = {
            "decision": "stop_no_48" if memory_net <= 0 or final_net < 0 else "promote_to_48",
            "reason": (
                "The discordant-row repeat did not reproduce a positive memory-only net; "
                "the preregistered promotion rule requires memory-only improvement."
                if memory_net <= 0
                else "Both repeated endpoints satisfy the preregistered direction gates."
            ),
        }

    REPEAT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    REPEAT_MANIFEST.write_text(json.dumps(repeat_manifest, indent=2) + "\n")
    return result


def render(result: dict) -> str:
    memory = result["comparisons"]["memory_only"]
    final = result["comparisons"]["final"]
    lines = [
        "# v2b gate16 — paired QA analysis",
        "",
        "This is a targeted mechanism gate, **not** a representative accuracy estimate.",
        "",
        "| Endpoint | Frozen batch15 | Batch8 | Wins | Losses | Net | Exact p |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Memory-only proxy | {memory['old_correct']}/16 | {memory['new_correct']}/16 | "
        f"{len(memory['wins'])} | {len(memory['losses'])} | {memory['net']:+d} | "
        f"{memory['exact_mcnemar_p']:.4f} |",
        f"| Final fallback | {final['old_correct']}/16 | {final['new_correct']}/16 | "
        f"{len(final['wins'])} | {len(final['losses'])} | {final['net']:+d} | "
        f"{final['exact_mcnemar_p']:.4f} |",
        "",
        "The archived memory-only value is a proxy: a row counts only when its final answer "
        "was correct and no raw fallback ran; old pre-fallback drafts were not saved.",
        "",
        f"Raw fallback triggered on **{len(result['raw_fallback_effect'])}/16** rows.",
        "",
        "The first run is directionally positive but ambiguous. Per the preregistration, "
        f"only the **{len(result['repeat_ids'])}** discordant rows are eligible for one repeat; "
        "there is no basis yet for an automatic 48-question expansion.",
    ]
    if "repeat_comparisons" in result:
        lines += ["", "## Discordant-row repeat", ""]
        for name, label in (("memory_only", "Memory-only"), ("final", "Final fallback")):
            repeat = result["repeat_comparisons"][name]
            stability = result["repeat_stability"][name]
            lines.append(
                f"- {label}: {len(repeat['wins'])} wins / {len(repeat['losses'])} losses "
                f"(net {repeat['net']:+d}); first/repeat agreement "
                f"{stability['agreements']}/{stability['n']}."
            )
        lines += [
            "",
            f"**Decision: `{result['promotion']['decision']}`.** {result['promotion']['reason']}",
            "The repeat set was selected from first-run disagreements, so no inferential "
            "p-value is attached to it.",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    result = analyse()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    OUT_MD.write_text(render(result))
    print(render(result), end="")
    print(f"Repeat manifest: {REPEAT_MANIFEST}")


if __name__ == "__main__":
    main()
