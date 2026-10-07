"""Quote retention/provenance replay of exposed train150 preferences; no LLM calls."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]

from analysis_io import sha256_file  # noqa: E402

from llm_long_term_memory.evaluation.datasets.longmemeval import load  # noqa: E402
from llm_long_term_memory.ingest.personal_context import (  # noqa: E402
    VERSION,
    PersonalContextRepair,
)
from llm_long_term_memory.store import Memory  # noqa: E402


def measure():
    manifest = REPO / "results/manifests/train150.json"
    allowed = set(json.loads(manifest.read_text(encoding="utf-8"))["question_ids"])
    archive_path = REPO / "results/raw/preference-extraction-probe-v1.json"
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    rows = []
    for instance in load("s", REPO / "data"):
        if (
            instance.question_id not in allowed
            or instance.question_type != "single-session-preference"
        ):
            continue
        # Exposed train150 only, before accessing any gold field.
        sessions = [s for s in instance.sessions if any(t.has_answer for t in s.turns)]
        arms = {}
        for arm, saved in archive["arms"].items():
            added, anchors_fixed, reference_retained, reference_total = [], 0, 0, 0
            for session in sessions:
                baseline = [
                    Memory(**copy.deepcopy(m))
                    for m in saved["memories"]
                    if m.get("source_session_id") == session.session_id
                ]
                original = [m.content for m in baseline]
                before = [m.source_turn_index for m in baseline]
                outcome = PersonalContextRepair().repair(session, baseline, "user")
                assert original == [m.content for m in baseline]
                anchors_fixed += sum(
                    m.source_turn_index != old for m, old in zip(baseline, before, strict=True)
                )
                for m in outcome.memories:
                    raw = session.turns[m.source_turn_index]
                    assert (
                        raw.role == "user"
                        and raw.content[m.source_char_start : m.source_char_end] == m.content
                    )
                added.extend(outcome.memories)
                for index, turn in enumerate(session.turns):
                    if not turn.has_answer or turn.role != "user":
                        continue
                    reference_total += 1
                    # A quotation is a navigation aid, not proof all answer facts were extracted.
                    reference_retained += any(
                        m.source_turn_index == index for m in outcome.memories
                    )
            arms[arm] = {
                "quotes_added": len(added),
                "quote_words": sum(m.token_count for m in added),
                "numeric_anchors_fixed": anchors_fixed,
                "reference_user_turns": reference_total,
                "reference_turns_with_quotes": reference_retained,
                "old_contents_preserved": True,
            }
        rows.append({"question_id": instance.question_id, "arms": arms})
    return {
        "experiment_class": "development-offline-provenance",
        "provider_calls": 0,
        "repair_version": VERSION,
        "questions": len(rows),
        "rows": rows,
        "input_sha256": sha256_file(archive_path),
        "manifest_sha256": sha256_file(manifest),
        "source_sha256": sha256_file(REPO / "src/llm_long_term_memory/ingest/personal_context.py"),
        "limitations": (
            "Quote retention, not semantic completeness or QA accuracy. More indexed text."
        ),
        "generator_sha256": sha256_file(Path(__file__)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    output = REPO / f"results/analysis/personal-context-repair-v1{args.label}.train150.json"
    if output.exists():
        raise SystemExit("Refusing to overwrite historical evidence")
    result = measure()
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"questions": result["questions"], "rows": result["rows"]}, indent=2))
