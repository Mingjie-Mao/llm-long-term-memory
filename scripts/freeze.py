"""Capture the system as frozen for the held-out run, and verify it later.

    python scripts/freeze.py --capture   # write the freeze record
    python scripts/freeze.py             # check the working tree still matches

The held-out set is worth exactly as much as the promise that nothing moved
between freezing and running it. A promise in a README is not checkable, and this
project has already been bitten once by a store whose `meta` claimed one extractor
while 68% of its rows came from another. So the freeze is a set of content hashes
over everything that can change an answer, and this script re-derives them.

What is covered, and why each one:

    extractor      version, both stage prompts, the schema, the model, batch size
                   — the ingest fingerprint, which is what wrote the store
    answerer       prompt version and the prompt text itself, since a version
                   string is edited by hand and the text is what the model sees
    judge          same, because a lenient judge is a silent accuracy gain
    retrieval      weights, top_k, candidate_limit, decay — four of the five
                   signals are weighted zero and that must stay true or the
                   held-out run measures a different system
    fallback       enabled, max_turns, max_chars, and the recover() source, since
                   its level-selection logic was rewritten mid-project
    store          row counts and the ingest fingerprint of two-stage-p10
    held-out set   the manifest hash, so it is provable the questions did not move

What is deliberately not covered: results files, scripts, documentation. They can
be corrected after the freeze without changing a single answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from llm_long_term_memory.config import ExperimentConfig, Settings  # noqa: E402
from llm_long_term_memory.ingest import fingerprint  # noqa: E402
from llm_long_term_memory.ingest.pipeline import resolved_sessions_per_request  # noqa: E402
from llm_long_term_memory.llm import Limits, QuotaManager  # noqa: E402

FREEZE = REPO / "results" / "frozen" / "p10-final" / "freeze.json"
STORE_NAME = "two-stage-p10"
CONFIG = "configs/fallback.yaml"
HELDOUT = REPO / "results" / "manifests" / "heldout100.json"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def capture() -> dict:
    from llm_long_term_memory.evaluation.judge import JUDGE_PROMPT_VERSION, JUDGE_SYSTEM
    from llm_long_term_memory.evaluation.runners.base import ANSWER_PROMPT_VERSION, ANSWER_SYSTEM

    settings = Settings()
    cfg = ExperimentConfig.from_yaml(str(REPO / CONFIG))
    quota = QuotaManager(
        state_dir=settings.store_dir / "quota",
        default=Limits(rpm=cfg.quota.rpm, tpm=cfg.quota.tpm, rpd=cfg.quota.rpd),
    )
    quota.load_learned()
    spr = resolved_sessions_per_request(
        cfg.ingest.sessions_per_request, quota.for_model(cfg.models.extractor).limits.tpm
    )

    db = settings.store_dir / f"{STORE_NAME}.db"
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        counts = {
            "memories": conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0],
            "namespaces": conn.execute("SELECT COUNT(DISTINCT user_id) FROM memories").fetchone()[
                0
            ],
            "sessions": conn.execute("SELECT COUNT(DISTINCT session_id) FROM turns").fetchone()[0],
            "turns": conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0],
            "scope_null": conn.execute(
                "SELECT COUNT(*) FROM memories WHERE scope IS NULL"
            ).fetchone()[0],
            "active": conn.execute(
                "SELECT COUNT(*) FROM memories WHERE status='active'"
            ).fetchone()[0],
        }
        stored_fp = conn.execute(
            "SELECT value FROM meta WHERE key='ingest_fingerprint'"
        ).fetchone()[0]
    finally:
        conn.close()

    return {
        "note": (
            "The system as frozen before the held-out run. Every hash here is "
            "re-derivable by scripts/freeze.py; a mismatch means the held-out "
            "result describes something other than what was frozen."
        ),
        "store": STORE_NAME,
        "config": CONFIG,
        "extractor": fingerprint.from_config(cfg, sessions_per_request=spr).as_dict(),
        "store_ingest_fingerprint": json.loads(stored_fp),
        "store_counts": counts,
        "answerer": {"version": ANSWER_PROMPT_VERSION, "prompt_sha": sha(ANSWER_SYSTEM)},
        "judge": {"version": JUDGE_PROMPT_VERSION, "prompt_sha": sha(JUDGE_SYSTEM)},
        "retrieval": {
            "weights": cfg.retrieval.weights.model_dump(),
            "top_k": cfg.retrieval.top_k,
            "candidate_limit": cfg.retrieval.candidate_limit,
            "rerank_enabled": cfg.retrieval.rerank.enabled,
            "temporal_resolution": cfg.temporal_resolution,
        },
        "fallback": {
            "enabled": cfg.fallback.enabled,
            "max_turns": cfg.fallback.max_turns,
            "max_chars": cfg.fallback.max_chars,
            "recover_sha": sha(
                (REPO / "src/llm_long_term_memory/retrieve/fallback.py").read_text(encoding="utf-8")
            ),
        },
        "heldout_manifest_sha": sha(HELDOUT.read_text(encoding="utf-8")),
        "heldout_questions": len(json.loads(HELDOUT.read_text(encoding="utf-8"))["question_ids"]),
    }


def differences(frozen: dict, current: dict, path: str = "") -> list[str]:
    out = []
    for key in sorted(set(frozen) | set(current)):
        if key == "note":
            continue
        here = f"{path}.{key}" if path else key
        a, b = frozen.get(key), current.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            out += differences(a, b, here)
        elif a != b:
            out.append(f"{here}: {a!r} -> {b!r}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true", help="write the freeze record")
    args = parser.parse_args()

    current = capture()
    if args.capture:
        FREEZE.parent.mkdir(parents=True, exist_ok=True)
        FREEZE.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"frozen -> {FREEZE}")
        for k in ("extractor", "answerer", "judge", "retrieval", "fallback"):
            print(f"  {k:14s} {json.dumps(current[k], sort_keys=True)[:96]}")
        print(f"  {'store':14s} {json.dumps(current['store_counts'], sort_keys=True)}")
        print(
            f"  {'held-out':14s} {current['heldout_questions']} questions, "
            f"sha {current['heldout_manifest_sha']}"
        )
        return 0

    if not FREEZE.exists():
        print(f"no freeze record at {FREEZE} — run with --capture first")
        return 1
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    moved = differences(frozen, current)
    if moved:
        print("\033[31mThe system has moved since it was frozen:\033[0m")
        for m in moved:
            print(f"  {m}")
        print("\nA held-out result measured now would not describe the frozen system.")
        return 1
    print("\033[32mUnchanged since the freeze.\033[0m")
    print(
        f"  extractor {frozen['extractor']['extractor_version']}, "
        f"store {frozen['store_counts']['memories']:,} memories, "
        f"held-out {frozen['heldout_questions']} questions unopened"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
