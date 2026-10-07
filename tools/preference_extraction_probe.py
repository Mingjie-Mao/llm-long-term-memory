"""Does asking for preferences shown in passing get them written down? Extraction only.

    python tools/preference_extraction_probe.py              # plan and cost, no calls
    python tools/preference_extraction_probe.py --run        # ~12 extractor calls
    python tools/preference_extraction_probe.py --analyse-only

Registered in `results/prereg-preference-extraction-probe-v1.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import write_report  # noqa: E402

SETS = ("dev50", "train150", "dev100", "heldout100")
CONFIG = "configs/v2b-batch8.yaml"
BATCH = 8
ARCHIVE = REPO / "results" / "raw" / "preference-extraction-probe-v1.json"
USAGE = REPO / "results" / "raw" / "preference-extraction-probe-v1.usage.json"
ARMS = ("plain", "observer")
PREFERENCE_TYPES = {"preference", "profile"}
LIFT = 6
VOLUME = 0.30


def cohort() -> list[tuple[str, object]]:
    """(question id, gold session) for every non-test100 preference question."""
    from llm_long_term_memory.evaluation.datasets import longmemeval as lme

    instances = {i.question_id: i for i in lme.load("s", REPO / "data")}
    pairs = []
    for name in SETS:
        manifest = json.loads((REPO / "results" / "manifests" / f"{name}.json").read_text("utf-8"))
        for qid in manifest["question_ids"]:
            inst = instances[qid]
            if inst.question_type != "single-session-preference":
                continue
            for session in inst.sessions:
                if any(t.has_answer for t in session.turns):
                    pairs.append((qid, session))
    return pairs


def run() -> None:
    from llm_long_term_memory.config import ExperimentConfig, Settings
    from llm_long_term_memory.ingest.two_stage import TwoStageExtractor
    from llm_long_term_memory.llm import Limits, QuotaManager, UsageTracker
    from llm_long_term_memory.llm.client import GeminiClient

    cfg = ExperimentConfig.from_yaml(CONFIG)
    settings = Settings()
    quota = QuotaManager(
        state_dir=settings.store_dir / "quota",
        default=Limits(rpm=cfg.quota.rpm, tpm=cfg.quota.tpm, rpd=cfg.quota.rpd),
    )
    quota.load_learned()
    usage = UsageTracker()
    client = GeminiClient(settings.require_api_key(), quota=quota, usage=usage)
    pairs = cohort()
    sessions = [session for _, session in pairs]
    archive = {"config": CONFIG, "model": cfg.models.extractor, "batch": BATCH, "arms": {}}
    try:
        for arm in ARMS:
            extractor = TwoStageExtractor(
                client, cfg.models.extractor, preference_observer=arm == "observer"
            )
            memories = []
            for start in range(0, len(sessions), BATCH):
                memories += extractor.extract(sessions[start : start + BATCH]).memories
            archive["arms"][arm] = {
                "version": extractor.version,
                "memories": [
                    {
                        k: v
                        for k, v in asdict(m).items()
                        if isinstance(v, (str, int, float, type(None)))
                    }
                    for m in memories
                ],
            }
    finally:
        usage.save(USAGE, merge=True)
    ARCHIVE.write_text(json.dumps(archive, indent=1) + "\n", encoding="utf-8")


def analyse() -> dict:
    archive = json.loads(ARCHIVE.read_text(encoding="utf-8"))
    pairs = cohort()
    gold = {
        qid: {(session.session_id, i) for i, t in enumerate(session.turns) if t.has_answer}
        for qid, session in pairs
    }
    session_of = {session.session_id: qid for qid, session in pairs}
    result = {"questions": len(pairs), "arms": {}}
    for arm in ARMS:
        memories = archive["arms"][arm]["memories"]
        pref, anyc = set(), set()
        lines = []
        for m in memories:
            qid = session_of.get(m.get("source_session_id"))
            if qid is None or m.get("source_turn_index") is None:
                continue
            if (m["source_session_id"], m["source_turn_index"]) not in gold[qid]:
                continue
            anyc.add(qid)
            if m.get("type") in PREFERENCE_TYPES or m.get("scope") == "preference":
                pref.add(qid)
                lines.append({"question_id": qid, "content": m["content"]})
        result["arms"][arm] = {
            "version": archive["arms"][arm]["version"],
            "preference_covered": len(pref),
            "any_covered": len(anyc),
            "memories_per_session": len(memories) / max(1, len(pairs)),
            "preference_lines_on_gold_turns": lines,
        }
    plain, observer = result["arms"]["plain"], result["arms"]["observer"]
    lift = observer["preference_covered"] - plain["preference_covered"]
    growth = observer["memories_per_session"] / max(1e-9, plain["memories_per_session"]) - 1
    result["rules"] = {
        "preference_covered_up_by_6": lift >= LIFT,
        "volume_up_at_most_30pct": growth <= VOLUME,
    }
    result["lift"], result["volume_growth"] = lift, growth
    result["pursue"] = all(result["rules"].values())
    return result


def render(result: dict) -> str:
    p, o = result["arms"]["plain"], result["arms"]["observer"]
    lines = [
        "# Capturing preferences shown in passing — extraction probe",
        "",
        "> Extraction only; no answer is scored. Registered in",
        "> `results/prereg-preference-extraction-probe-v1.md`.",
        "",
        f"{result['questions']} preference questions (all non-test100), one gold session each.",
        "",
        "| arm | preference memory on a gold turn | any memory on a gold turn "
        "| memories / session |",
        "|---|---:|---:|---:|",
        f"| plain | {p['preference_covered']} | {p['any_covered']} "
        f"| {p['memories_per_session']:.1f} |",
        f"| observer | {o['preference_covered']} | {o['any_covered']} "
        f"| {o['memories_per_session']:.1f} |",
        "",
    ]
    lines += [f"- {'PASS' if ok else 'FAIL'} — `{k}`" for k, ok in result["rules"].items()]
    lines += [
        "",
        "**Decision: "
        + ("register an ingest-scale test" if result["pursue"] else "record, do not pursue")
        + "**",
        "",
        "## Observer preference lines on gold turns (read for invented preferences)",
        "",
    ]
    lines += [
        f"- `{x['question_id']}`: {x['content']}" for x in o["preference_lines_on_gold_turns"]
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="make the ~12 extractor calls")
    parser.add_argument("--analyse-only", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args()
    if not args.run and not args.analyse_only:
        pairs = cohort()
        batches = -(-len(pairs) // BATCH)
        print(
            f"plan: {len(pairs)} gold sessions from {len({q for q, _ in pairs})} questions; "
            f"{batches} batches x 2 stages x {len(ARMS)} arms = {batches * 2 * len(ARMS)} "
            "extractor calls. Nothing sent. Pass --run to make them."
        )
        return 0
    if args.run:
        run()
    result = analyse()
    print(write_report(result, render, json_out=args.json_out, md_out=args.md_out), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
