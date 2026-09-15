"""The three things a deployment holding real data needs and this one did not have.

`backup_restore.py` can take a snapshot, prune old ones and rehearse a restore, but
somebody has to remember to run it. That is the whole gap: a backup policy nobody
executes is a plan, not a backup. So this is one command a scheduler can call on a
timer, and it does the three jobs that were outstanding before the writeable service
could be deployed.

**Backup, then prune, in that order.** Pruning first would delete the copy the new
snapshot has not replaced yet.

**The erasure journal goes somewhere else.** A deletion record that lives beside the
store it describes is lost with the store, and then a restore replays nothing and
reports that the erased namespaces were correctly removed — the exact failure the
restore drill was built to catch, one layer up. `--journal-copy` writes it outside the
backup folder so the two do not share a disk.

**Budgets are checked against a threshold, not just enforced.** `AccountBudgets` refuses
a call once a cap is reached, which the caller learns as a 429 at the worst moment. This
reports who is approaching a cap while there is still time to act.

Exit codes are what a scheduler reads: 0 all clear, 1 something needs attention, 2 the
run could not be completed. Every run appends one JSON line to `--log`, so "did the
backup run last night" is answerable without reading a mailbox.

    python3 tools/operate.py --store stores/live.db --backups /var/backups/lltm \
        --journal-copy /mnt/offsite/lltm --keep 7 --alert-at 0.8
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))


def copy_journal(store: Path, destination: Path) -> dict:
    """Put the erasure journal somewhere the store's disk failing cannot take with it."""
    from llm_long_term_memory.store.erasure import journal_path_for

    source = journal_path_for(store)
    if not source.is_file():
        # Not an error: a store that has never served an erasure has no journal. Said
        # explicitly so that "no journal" is distinguishable from "copy failed".
        return {"copied": False, "reason": "no erasure journal yet", "source": str(source)}
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / source.name
    shutil.copy2(source, target)
    lines = sum(1 for line in target.read_text(encoding="utf-8").splitlines() if line.strip())
    return {"copied": True, "source": str(source), "target": str(target), "erasures": lines}


def budget_pressure(store_path: Path, threshold: float) -> dict:
    """Who is close to a cap, while there is still time to do something about it."""
    from llm_long_term_memory.api.budget import ANY, AccountBudgets, quota_day
    from llm_long_term_memory.store import SQLiteMemoryStore

    store = SQLiteMemoryStore(store_path, read_only=True)
    store.initialize()
    try:
        budgets = AccountBudgets(store)
        day = quota_day()
        approaching = []
        checked = 0
        # Accounts that hold a cap or have spent today, not accounts that hold memories:
        # an account can be burning quota with nothing written, and that is precisely the
        # one an operator needs named.
        store_user_ids = store.accounts_with_quota(day)
        for user_id in store_user_ids:
            spend = budgets.spent_today(user_id, day)
            # The tenant-wide cap, which is what `*`/`*` resolves to when no narrower
            # row exists. A `None` cap is "no cap", which is not a cap of zero.
            budget = budgets.budget_for(user_id, ANY, ANY)
            for dimension, spent, cap in (
                ("calls", spend.calls, budget.daily_calls),
                ("tokens", spend.tokens, budget.daily_tokens),
            ):
                if cap is None or cap <= 0:
                    continue
                checked += 1
                share = spent / cap
                if share >= threshold:
                    approaching.append(
                        {
                            "user_id": user_id,
                            "dimension": dimension,
                            "spent": spent,
                            "cap": cap,
                            "share": round(share, 3),
                        }
                    )
    finally:
        store.close()
    return {
        "day": day,
        "threshold": threshold,
        "accounts": len(store_user_ids),
        "capped_dimensions_checked": checked,
        "approaching": approaching,
    }


def run(args) -> tuple[int, dict]:
    import backup_restore

    started = datetime.now(UTC).isoformat()
    record: dict = {"started_at": started, "store": str(args.store)}
    problems: list[str] = []

    if not args.store.is_file():
        return 2, {**record, "error": f"{args.store} does not exist"}

    # 1. Snapshot first. Pruning before the new copy exists would delete the backup the
    #    new one has not replaced yet.
    try:
        manifest = backup_restore.snapshot(args.store, args.backups / started.replace(":", ""))
        record["backup"] = {"counts": manifest["counts"], "files": len(manifest["files"])}
    except (OSError, ValueError) as exc:
        return 2, {**record, "error": f"backup failed: {exc}"}

    # 2. The journal, off this disk.
    if args.journal_copy:
        try:
            record["journal"] = copy_journal(args.store, args.journal_copy)
        except OSError as exc:
            problems.append(f"journal copy failed: {exc}")
            record["journal"] = {"copied": False, "reason": str(exc)}

    # 3. Retention. Every backup kept is another copy an erased namespace survives in,
    #    so this is part of the erasure story rather than housekeeping.
    if args.keep:
        try:
            record["prune"] = backup_restore.prune(args.backups, args.keep)
        except (OSError, ValueError) as exc:
            problems.append(f"prune failed: {exc}")

    # 4. Who is about to be refused.
    try:
        pressure = budget_pressure(args.store, args.alert_at)
        record["budgets"] = pressure
        if pressure["approaching"]:
            problems.append(
                f"{len(pressure['approaching'])} account/dimension pair(s) at or above "
                f"{args.alert_at:.0%} of cap"
            )
    except Exception as exc:  # a reporting failure must not look like a backup failure
        problems.append(f"budget check failed: {type(exc).__name__}: {exc}")

    record["problems"] = problems
    record["finished_at"] = datetime.now(UTC).isoformat()
    return (1 if problems else 0), record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--backups", type=Path, required=True, help="folder to snapshot into")
    parser.add_argument(
        "--journal-copy",
        type=Path,
        help="folder for the erasure journal, on different storage from --backups",
    )
    parser.add_argument("--keep", type=int, default=7, help="backups to retain; 0 keeps all")
    parser.add_argument(
        "--alert-at",
        type=float,
        default=0.8,
        help="report an account at or above this share of its daily cap",
    )
    parser.add_argument("--log", type=Path, help="append one JSON line per run here")
    args = parser.parse_args()

    code, record = run(args)
    record["exit_code"] = code
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(record) + "\n")
    print(json.dumps(record, indent=2))
    for problem in record.get("problems", []):
        print(f"ATTENTION: {problem}", file=sys.stderr)
    if "error" in record:
        print(f"FAILED: {record['error']}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
