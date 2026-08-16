"""Re-record the inspector's demonstration runs against the current store.

The recordings exist so a demo page costs nothing to open: a real run against a
real model, replayed and labelled as replayed, with a **Run live answer** button
for anyone who wants it executed again. What keeps that honest is the fingerprint
— of the store's contents and the prompt versions — which the service compares
against the live process and reports as stale when it no longer matches.

Which had happened without anyone noticing. The recordings on disk carried
`extractor: None`, from a store written before ingestion stamped its version, so
both demos read as stale against every store this repo now has. A stale demo is
not a small thing here: the whole argument for showing a recording rather than a
mock is that the reader can tell which one they are looking at.

This script did not exist. The recordings were made by hand, which is why nothing
noticed they had gone stale and why re-making them was not a command anyone could
run. It is checked in so the next store rebuild ends in a re-record rather than in
a stale banner nobody can clear.

    python scripts/record_golden.py            # show what would change
    python scripts/record_golden.py --write    # spend the calls and save

Each demo is run `--runs` times (3 by default). Runs that disagree on whether the
archive was needed at all are not recorded: a question that sometimes answers from
memory alone is not a demonstration of recovery, and one run is an anecdote. The
answer text is allowed to vary — it is generated prose — and so is the route, since
the first-pass verdict is not deterministic and `need_source` and `no_evidence`
reach the archive by different levels. When the route varies the recording says so
in its `note` rather than presenting one run's path as the only one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from llm_long_term_memory.api.golden import (  # noqa: E402
    GoldenRun,
    RecordedEvidence,
    fingerprint,
    load_runs,
    recorded_at_now,
    save_runs,
    store_state,
)
from llm_long_term_memory.api.service import (  # noqa: E402
    ANSWER_PROMPT_VERSION,
    JUDGE_PROMPT_VERSION,
    MemoryService,
)

# Named here rather than inherited from `MemoryService`'s defaults. They agree
# today, and stating them anyway means a change to the service default cannot
# silently re-record the demos against a store nobody meant to demo.
#
# The config is the product one: the baseline config leaves the fallback off, and a
# Mayo recording made under it could not reach the archive at all.
DEMO_STORE = "two-stage-p10"
DEMO_CONFIG = "configs/fallback.yaml"

# The four demos the README links. `timeline` and `collectibles` are not here:
# they are store views, rendered from rows at request time, and cost nothing to
# open because they never call a model. Only the two that answer a question need
# recording.
DEMOS = [
    ("mayo", "41275add", "what was the mayo clinic youtube video you recommended?"),
    (
        "battery",
        "09d032c9",
        "I've been having trouble with the battery life on my phone lately. Any tips?",
    ),
]


def resolve_evidence(service: MemoryService, refs: list[str]) -> list[RecordedEvidence]:
    """Turn the answerer's `session_id:turn_index` refs into quotable turns.

    The runner records references rather than text, which is right for a result row
    and not enough for a page that shows the reader the sentence the answer came
    from. Resolving them here rather than storing prose twice keeps the recording
    anchored to the archive: a turn that no longer exists is a recording that
    cannot be made, which is the honest outcome.
    """
    out: list[RecordedEvidence] = []
    for ref in refs:
        session_id, _, index = ref.rpartition(":")
        turns = {t.turn_index: t for t in service.store.turns_for_session(session_id)}
        turn = turns.get(int(index))
        if turn is None:
            raise LookupError(f"evidence turn {ref} is not in {service.store_name}")
        out.append(
            RecordedEvidence(
                session_id=session_id,
                turn_index=turn.turn_index,
                role=turn.role,
                text=turn.content,
            )
        )
    return out


def record_one(service: MemoryService, name: str, namespace: str, query: str, runs: int):
    """Run one demo `runs` times and return a GoldenRun, or a reason it was not."""
    results = []
    for i in range(runs):
        try:
            results.append(service.answer(namespace, query))
        except Exception as exc:  # reported, not swallowed
            return None, f"run {i + 1} failed: {type(exc).__name__}: {exc}"

    # Agreement is required on *whether* raw evidence was needed, not on which
    # level supplied it. The first-pass answerer decides between `need_source` and
    # `no_evidence` and is not deterministic about it, and the two route to
    # different levels because `need_source` passes the retrieved memories and
    # `no_evidence` passes none. Demanding one level would refuse to record a demo
    # whose every run recovers the same answer by a different route.
    #
    # Requiring agreement on "was the archive needed at all" is the property the
    # demo actually claims. A question that sometimes answers from memory and
    # sometimes does not is still not recordable.
    used = {r["fallback_level"] != "none" for r in results}
    if len(used) != 1:
        return None, (
            "runs disagreed on whether the archive was needed: "
            f"{sorted((r['answer_status'], r['fallback_level']) for r in results)}"
        )

    first = results[0]
    levels = {r["fallback_level"] for r in results}
    note = (
        ""
        if len(levels) == 1
        else (
            f"The first-pass verdict varies across runs, so this question reaches "
            f"the archive by more than one route ({', '.join(sorted(levels))}). "
            f"All {runs} runs recovered the same answer; the route shown is the "
            f"one recorded."
        )
    )
    return (
        GoldenRun(
            name=name,
            namespace=namespace,
            query=query,
            answer=first["answer"],
            answer_status=first["answer_status"],
            fallback_level=first["fallback_level"],
            fallback_reason=first["fallback_reason"],
            evidence=resolve_evidence(service, first.get("fallback_turns") or []),
            memories_selected=first["memories_selected"],
            candidates_considered=first["candidates_considered"],
            recorded_at=recorded_at_now(),
            runs=runs,
            note=note,
            fingerprint=fingerprint(
                namespace,
                query,
                {
                    "answer_prompt": ANSWER_PROMPT_VERSION,
                    "judge_prompt": JUDGE_PROMPT_VERSION,
                    "extractor": service.store.get_meta("extractor_version"),
                },
                store_state(service.store, namespace),
            ),
            versions={
                "answer_prompt": ANSWER_PROMPT_VERSION,
                "judge_prompt": JUDGE_PROMPT_VERSION,
                "extractor": service.store.get_meta("extractor_version"),
            },
        ),
        None,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="save; otherwise dry-run")
    parser.add_argument("--runs", type=int, default=3, help="independent runs per demo")
    parser.add_argument("--store-name", default=DEMO_STORE, help=f"default: {DEMO_STORE}")
    parser.add_argument("--config", default=DEMO_CONFIG, help=f"default: {DEMO_CONFIG}")
    args = parser.parse_args()

    service = MemoryService(store_name=args.store_name, config_path=args.config)
    print(f"recording against {service.store_name} · {service.config.name} · {args.runs} runs each")
    if not args.write:
        print("dry run — no calls are spent and nothing is written\n")

    existing = load_runs()
    for name, _, _ in DEMOS:
        run = existing.get(name)
        if run is not None:
            _, current = service.golden_run(name)
            print(f"  {name:10s} on disk: {'current' if current else 'STALE'}")

    if not args.write:
        print("\nRe-run with --write to record.")
        return 0

    print()
    recorded = dict(existing)
    failures = []
    for name, namespace, query in DEMOS:
        run, why = record_one(service, name, namespace, query, args.runs)
        if run is None:
            print(f"  \033[31mFAIL\033[0m  {name}: {why}")
            failures.append(name)
            continue
        recorded[name] = run
        print(
            f"  \033[32mOK\033[0m    {name}: status={run.answer_status} "
            f"fallback={run.fallback_level} evidence={len(run.evidence)} "
            f"memories={run.memories_selected}"
        )

    if failures:
        # Partial writes would leave the file describing two different stores, which
        # is the same class of mistake as the mixed store.
        print(f"\n\033[31m{len(failures)} demo(s) failed — nothing written.\033[0m")
        return 1

    path = save_runs(recorded)
    print(f"\n→ {path}")

    # Read back through the service, because "the fingerprint I just wrote matches
    # the fingerprint I just computed" proves nothing about the path the inspector
    # actually takes.
    stale = [n for n, _, _ in DEMOS if not service.golden_run(n)[1]]
    if stale:
        print(f"\033[31mstill stale after writing: {stale}\033[0m")
        return 1
    print("\033[32mall recordings current\033[0m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
