"""A simulated local trial of the product surface — no provider, no deployment, no users.

There is no staging environment and no trial group, so this stands in for one: scripted
accounts drive the real service objects (store, index, idempotency, budget ledger,
erasure journal, backup and restore) with a deterministic stub extractor in place of the
provider. Every check below is one thing a first outside user could hit on day one.

What it does show: whether the write path stays consistent under retry, injected failure,
budget refusal and restore, and how long those paths take on this machine.

What it cannot show: answer quality, extraction quality, real user behaviour, multi-process
contention, or anything about a hosted deployment. Zero provider calls, so nothing here is
evidence about accuracy or cost.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

from llm_long_term_memory.api.budget import BudgetExceeded  # noqa: E402
from llm_long_term_memory.api.service import (  # noqa: E402
    IdempotencyConflict,
    MemoryService,
    TurnExtractionOutcome,
)
from llm_long_term_memory.store import Memory  # noqa: E402

SCRIPT = [
    ("user", "I adopted a rescue cat called Pepper last March."),
    ("user", "I finally planted basil and mint on the balcony."),
    ("user", "We ate at Rosetta on Friday; the pasta was the best I have had."),
    ("user", "I sold the old road bike, so it is just the commuter bike now."),
    ("user", "I read Piranesi over the weekend and started Klara and the Sun."),
]


class StubEncoder:
    dim = 2

    def encode(self, texts, show_progress=False):
        return np.ones((len(texts), 2), dtype=np.float32)

    def encode_one(self, text):
        return np.ones(2, dtype=np.float32)


class ScriptedExtractor:
    """One memory per turn, derived from the turn itself. Deterministic, never a provider.

    Real extraction is a model call and is the thing this trial deliberately does not
    exercise; substituting it keeps the trial free and repeatable, and means nothing here
    can be read as evidence about extraction quality.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.fail_next = False

    def extract_turn(self, *, user_id, session_id, role, content, now, context_turns=()):
        self.calls += 1
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("simulated provider failure")
        memory = Memory(
            id=f"mem_{user_id}_{self.calls:04d}",
            user_id=user_id,
            type="episodic",
            content=content,
            token_count=max(1, len(content) // 4),
            subject="user",
            predicate="stated",
            object=content[:40],
            scope="event",
            source_session_id=session_id,
            source_turn_index=0,
        )
        return TurnExtractionOutcome([memory], {"requests": 1})


class Settings:
    def __init__(self, root: Path) -> None:
        self.store_dir = root
        self.data_dir = root
        self.results_dir = root
        self.gemini_api_key = ""

    @property
    def has_api_key(self) -> bool:
        return False


def build_service(root: Path) -> MemoryService:
    """The real service, pointed at a temporary store, with the provider stubbed out."""
    return MemoryService(
        config_path=REPO / "configs/fallback.yaml",
        store_name="trial",
        settings=Settings(root),
        encoder=StubEncoder(),
        extractor=ScriptedExtractor(),
    )


def state(service: MemoryService, user: str) -> dict:
    export = service.export_user(user)
    return {
        "memories": len(export["memories"]),
        "turns": sum(len(s["turns"]) for s in export["sessions"]),
        "sessions": len(export["sessions"]),
    }


def check(report: list, name: str, passed: bool, detail: object = None) -> None:
    report.append({"check": name, "passed": bool(passed), "detail": detail})


def run(root: Path, accounts: int) -> dict:
    service = build_service(root)
    latencies: list[float] = []
    checks: list[dict] = []
    users = [f"trial-{n:02d}" for n in range(1, accounts + 1)]

    # 1. the ordinary path: every account writes its whole script
    for user in users:
        for index, (role, text) in enumerate(SCRIPT):
            started = time.perf_counter()
            service.add_message(user, role, text, session_id=f"s{index // 2}")
            latencies.append((time.perf_counter() - started) * 1000)
    check(
        checks,
        "every scripted turn is stored once",
        all(state(service, user)["turns"] == len(SCRIPT) for user in users),
        {"turns_per_account": len(SCRIPT)},
    )

    # 2. a client retries with the same key: the first reply is replayed, nothing appended
    user = users[0]
    before = state(service, user)
    first = service.add_message(user, "user", "I booked the vet for Pepper.", "s9", "k-retry")
    replay = service.add_message(user, "user", "I booked the vet for Pepper.", "s9", "k-retry")
    after = state(service, user)
    check(
        checks,
        "a retried write replays instead of appending",
        replay.get("idempotent_replay") is True
        and (replay["session_id"], replay["turn_index"])
        == (first["session_id"], first["turn_index"])
        and after["turns"] == before["turns"] + 1,
        after,
    )

    # 3. the same key with different content must be refused, not silently accepted
    try:
        service.add_message(user, "user", "Something else entirely.", "s9", "k-retry")
        conflicted = False
    except IdempotencyConflict:
        conflicted = True
    check(checks, "a reused key with new content is refused", conflicted)

    # 4. the provider fails mid-write; the retry must succeed and leave no half state
    before = state(service, user)
    service.extractor.fail_next = True
    try:
        service.add_message(user, "user", "The cat now sleeps on the desk.", "s9", "k-fail")
        failed = False
    except RuntimeError:
        failed = True
    mid = state(service, user)
    service.add_message(user, "user", "The cat now sleeps on the desk.", "s9", "k-fail")
    after = state(service, user)
    # The failed attempt rolls the turn back entirely, so the account is where it was;
    # the retry then writes it exactly once.
    check(
        checks,
        "a failed write leaves nothing behind and its retry writes once",
        failed and mid == before and after["turns"] == before["turns"] + 1,
        {"before": before, "after_failure": mid, "after_retry": after},
    )

    # 5. an account over its cap is refused here, not by a provider error everyone shares
    capped = users[-1]
    service.budgets.set_budget(capped, daily_calls=0)
    try:
        service.add_message(capped, "user", "One more note.", "s9")
        refused = False
    except BudgetExceeded:
        refused = True
    check(checks, "an account over budget is refused before any provider call", refused)

    # 6. one account's data must never surface in another's reads
    leaks = [
        r
        for other in users[1:]
        for r in service.search(other, "Pepper", limit=5).memories
        if getattr(r, "user_id", other) != other
    ]
    check(checks, "search never returns another account's memory", not leaks, len(leaks))

    # 7. export, erase, and a journal entry a restore can replay
    exported = service.export_user(users[1])
    removed = service.erase_user(users[1])
    entries = [e for e in service.erasures.entries() if e.user_id == users[1]]
    check(
        checks,
        "erasure empties the account and is journalled",
        state(service, users[1]) == {"memories": 0, "turns": 0, "sessions": 0}
        and len(entries) == 1,
        {"exported_memories": len(exported["memories"]), "removed": removed},
    )

    # 8. back up, lose the database, restore, and confirm the erased account stays erased
    from backup_restore import restore, snapshot

    backup = root / "backup"
    journal = service.erasures.path
    store_path = service.store.path
    snapshot(store_path, backup)
    service.store.close()
    into = root / "restored"
    report = restore(backup=backup, into=into, rebuild_index=True, erasures=journal)
    restored = sqlite3.connect(into / store_path.name)
    survivors = {row[0] for row in restored.execute("SELECT DISTINCT user_id FROM memories")}
    restored.close()
    check(
        checks, "a restore brings back the live accounts", users[0] in survivors, sorted(survivors)
    )
    check(
        checks,
        "a restore does not resurrect an erased account",
        users[1] not in survivors,
        {"restore": {k: report[k] for k in ("ok", "erasures") if k in report}},
    )

    latencies.sort()
    return {
        "schema_version": 1,
        "kind": "simulated_local_trial",
        "provider_calls": 0,
        "ran_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "accounts": accounts,
        "turns_written": len(latencies),
        "write_latency_ms": {
            "median": round(latencies[len(latencies) // 2], 2),
            "p95": round(latencies[int(len(latencies) * 0.95) - 1], 2),
            "max": round(latencies[-1], 2),
        },
        "checks": checks,
        "passed": sum(1 for c in checks if c["passed"]),
        "failed": sum(1 for c in checks if not c["passed"]),
        "not_covered": [
            "answer quality and extraction quality (the extractor is a stub)",
            "real user behaviour, and anything about a hosted deployment",
            "multi-process contention on the same store",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accounts", type=int, default=5)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path, default=REPO / "results/validation/local-trial.json")
    args = parser.parse_args()
    root = args.root or Path(REPO / ".trial-tmp")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        report = run(root, args.accounts)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, ensure_ascii=False))
    for row in report["checks"]:
        print(("PASS  " if row["passed"] else "FAIL  ") + row["check"])
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
