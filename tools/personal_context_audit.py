"""Inspect archived personal-context extraction on train150, without provider calls.

Coverage here is an anchor/type proxy. It does not establish semantic support or
question correctness. Speaker mismatches are review alerts, never automatic repairs.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

PERSONAL_SCOPES = {"preference", "profile", "plan", "commitment", "event"}
WORDS = re.compile(r"[a-z0-9]+")


def audit(instances, archive, train_ids):
    rows = []
    for instance in instances:
        # Filter before touching reference labels, including on dev100 instances.
        if (
            instance.question_id not in train_ids
            or instance.question_type != "single-session-preference"
        ):
            continue
        sessions = {s.session_id: s for s in instance.sessions}
        gold = {
            (s.session_id, i)
            for s in sessions.values()
            for i, t in enumerate(s.turns)
            if t.has_answer
        }
        if not gold:
            continue
        arms = {}
        for arm, saved in archive["arms"].items():
            memories = []
            for memory in saved["memories"]:
                sid = memory.get("source_session_id")
                if sid not in sessions or not any(s == sid for s, _ in gold):
                    continue
                session = sessions[sid]
                index = memory.get("source_turn_index")
                valid = type(index) is int and 0 <= index < len(session.turns)
                speaker = session.turns[index].role if valid else None
                on_gold = valid and (sid, index) in gold
                content = memory.get("content", "")
                tokens = set(WORDS.findall(content.lower()))
                numbers = {w for w in tokens if w.isdigit()}
                alternatives = []
                if speaker == "assistant" and numbers:
                    for i, turn in enumerate(session.turns):
                        own = set(WORDS.findall(turn.content.lower()))
                        if (
                            turn.role == "user"
                            and numbers <= own
                            and len((tokens & own) - numbers) >= 2
                        ):
                            alternatives.append(i)
                memories.append(
                    {
                        "content": content,
                        "type": memory.get("type"),
                        "scope": memory.get("scope"),
                        "session_id": sid,
                        "turn_index": index,
                        "source_role": speaker,
                        "on_reference_turn": bool(on_gold),
                        "user_turn_candidates_for_numeric_anchor_review": alternatives,
                    }
                )
            arms[arm] = {
                "old_preference_proxy": any(
                    m["on_reference_turn"]
                    and (m["type"] in {"preference", "profile"} or m["scope"] == "preference")
                    for m in memories
                ),
                "user_personal_context_proxy": any(
                    m["on_reference_turn"]
                    and m["source_role"] == "user"
                    and m["scope"] in PERSONAL_SCOPES
                    for m in memories
                ),
                "memories": memories,
            }
        rows.append(
            {
                "question_id": instance.question_id,
                "reference_turns": [
                    {
                        "session_id": sid,
                        "turn_index": i,
                        "role": sessions[sid].turns[i].role,
                        "text": sessions[sid].turns[i].content,
                    }
                    for sid, i in sorted(gold)
                ],
                "arms": arms,
            }
        )
    return {
        "experiment_class": "train150 archived extraction diagnosis; not QA accuracy",
        "provider_calls": 0,
        "questions": len(rows),
        "proxy_counts": {
            arm: {
                key: sum(r["arms"][arm][key] for r in rows)
                for key in ("old_preference_proxy", "user_personal_context_proxy")
            }
            for arm in archive["arms"]
        },
        "limitations": (
            "Proxies require semantic review. Numeric anchor alerts may be false positives."
        ),
        "rows": rows,
    }


def main():
    from llm_long_term_memory.evaluation.datasets.longmemeval import load

    manifest = REPO / "results/manifests/train150.json"
    archive = REPO / "results/raw/preference-extraction-probe-v1.json"
    output = REPO / "results/analysis/personal-context-first-loss-v1.train150.json"
    if output.exists():
        raise SystemExit("STOP: preserve existing diagnosis; use a new namespace")
    result = audit(
        load("s", REPO / "data"),
        json.loads(archive.read_text()),
        set(json.loads(manifest.read_text())["question_ids"]),
    )
    result["inputs"] = {
        str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), manifest, archive, REPO / "data/longmemeval_s_cleaned.json")
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
