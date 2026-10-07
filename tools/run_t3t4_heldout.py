"""Run `results/prereg-raw-primary-t3-t4-heldout100-v1.md` to completion, then its gate.

    python tools/run_t3t4_heldout.py --execute

Nine runs in the registered order (v1, t3, t4 for rep1, then rep2, then rep3), each by the
unchanged `lltm eval run`, which resumes from its saved rows. A run that stops is
restarted: after a quota stop (fewer than 15 calls left for the answerer or the judge)
at the next Pacific midnight, otherwise after a minute. Twelve restarts in a row without
a new row stop the job; so does the registration's deadline. Never answers a saved row
again; state in `results/analysis/t3t4-heldout100-v1.progress.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
from continue_acceptance_job import next_reset  # noqa: E402

ANALYSIS = REPO / "results/analysis"
STATE = ANALYSIS / "t3t4-heldout100-v1.progress.json"
LOG = ANALYSIS / "t3t4-heldout100-v1.log"
DEADLINE = datetime(2026, 10, 13, 13, 0, tzinfo=UTC)  # 2026-10-14 00:00 Sydney
RUNS = [
    (variant, rep)
    for rep in (1, 2, 3)
    for variant in ("two_stage_raw_primary", "two_stage_raw_primary_t3", "two_stage_raw_primary_t4")
]
MAX_STALLS = 12
QUOTA_FLOOR = 15


def rows(variant: str, rep: int) -> int:
    path = REPO / f"results/raw/{variant}.heldout100-t3t4-rep{rep}.jsonl"
    return (
        sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)
        if path.exists()
        else 0
    )


def remaining() -> dict[str, int]:
    from llm_long_term_memory import cli

    _, _, runner, _, _ = cli._build(
        "two_stage_raw_primary", "configs/fallback.yaml", "heldout100", read_only_store=True
    )
    try:
        quota = runner.client.quota
        return {
            m: quota.for_model(m).remaining_today
            for m in ("gemini-3.5-flash-lite", "gemma-4-31b-it")
        }
    finally:
        runner.store.close()


def save(state: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    tmp.replace(STATE)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps({"provider_calls": 0, "runs": len(RUNS), "deadline": DEADLINE.isoformat()})
        )
        return 0
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONWARNDEFAULTENCODING": "1",
    }
    state = {
        "started_at": datetime.now(UTC).isoformat(),
        "quota_waits": 0,
        "restarts": 0,
        "status": "RUNNING",
    }
    with LOG.open("a", encoding="utf-8") as log:
        for variant, rep in RUNS:
            stalls = 0
            while rows(variant, rep) < 100:
                if datetime.now(UTC) > DEADLINE:
                    state.update(status="INCOMPLETE_DEADLINE")
                    save(state)
                    return 1
                before = rows(variant, rep)
                state.update(current=f"{variant} rep{rep}", rows=before)
                save(state)
                log.write(
                    f"\n== {datetime.now(UTC).isoformat()} {variant} rep{rep} rows {before}/100\n"
                )
                log.flush()
                rc = subprocess.call(
                    [
                        str(REPO / ".venv/bin/lltm"),
                        "eval",
                        "run",
                        variant,
                        "--config",
                        "configs/fallback.yaml",
                        "--store-name",
                        "heldout100",
                        "--questions",
                        "results/manifests/heldout100.json",
                        "--label",
                        f"heldout100-t3t4-rep{rep}",
                        "--experiment-class",
                        "regression",
                    ],
                    cwd=REPO,
                    env=env,
                    stdout=log,
                    stderr=log,
                )
                if rows(variant, rep) >= 100:
                    break
                state["restarts"] += 1
                stalls = 0 if rows(variant, rep) > before else stalls + 1
                if stalls >= MAX_STALLS:
                    state.update(status="STOPPED_NO_PROGRESS", returncode=rc)
                    save(state)
                    return 1
                left = remaining()
                if min(left.values()) < QUOTA_FLOOR:
                    target = next_reset(datetime.now(UTC)) + timedelta(seconds=60)
                    state["quota_waits"] += 1
                    state.update(status="QUOTA_WAIT", waiting_until=target.isoformat(), quota=left)
                    save(state)
                    while datetime.now(UTC) < target:
                        time.sleep(60)
                    state.update(status="RUNNING")
                    state.pop("waiting_until", None)
                else:
                    time.sleep(60)
        rc = subprocess.call(
            [
                str(REPO / ".venv/bin/python"),
                str(REPO / "tools/raw_primary_gate_t3t4.py"),
                "--json-out",
                str(ANALYSIS / "raw-primary-t3-t4-heldout100-gate.json"),
                "--md-out",
                str(ANALYSIS / "raw-primary-t3-t4-heldout100-gate.md"),
            ],
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=log,
        )
        state.update(
            status="GATE_WRITTEN", gate_returncode=rc, finished_at=datetime.now(UTC).isoformat()
        )
        save(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
