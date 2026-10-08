"""Public train150 page-selector development gate; dry-run first, resumable, no dev gold."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
from analysis_io import rows_by_question, sha256_file  # noqa: E402
from grade_saved_recovering import grade_pending  # noqa: E402
from grounded_reader_diagnostic import save_reader  # noqa: E402

QIDS = ("b46e15ed", "9d25d4e0", "gpt4_731e37d7", "gpt4_a1b77f9c", "51a45a95", "eaca4986")
PUBLIC_SHA = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"


def main():
    from llm_long_term_memory import cli
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.evaluation.datasets.longmemeval import load
    from llm_long_term_memory.evaluation.harness import _exclusive
    from llm_long_term_memory.evaluation.recovery import RecoveringJudge
    from llm_long_term_memory.llm.usage import UsageTracker
    from llm_long_term_memory.runtime.evidence_pages import archive_pages
    from llm_long_term_memory.runtime.grounded_answering import personal_archive_review
    from llm_long_term_memory.store import external_session_id

    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--grade", action="store_true")
    parser.add_argument("--label", default="prerequisite-reader-v18-v1")
    args = parser.parse_args()
    if not args.label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in args.label):
        raise ValueError("invalid namespace")
    settings = Settings()
    dataset = (settings.data_dir / "longmemeval_s_cleaned.json").resolve()
    if sha256_file(dataset) != PUBLIC_SHA:
        raise ValueError("public dataset changed")
    train_path = REPO / "results/manifests/train150.json"
    allowed = set(json.loads(train_path.read_text(encoding="utf-8"))["question_ids"])
    if not set(QIDS) <= allowed:
        raise ValueError("gate must stay within exposed train150")
    instances = {i.question_id: i for i in load("s", settings.data_dir) if i.question_id in QIDS}
    gate = json.loads(
        (REPO / "results/analysis/role-paged-growth-v3.train150.json").read_text(encoding="utf-8")
    )
    for factor in ("1", "2", "4"):
        measured = gate["by_factor"][factor]
        if measured["covered"] != 146 or measured["complete_archives"] != 150:
            raise ValueError("registered offline growth gate failed")
    cfg, _, runner, judge, usage = cli._build(
        "two_stage_raw_primary_grounded_v18",
        "configs/fallback.yaml",
        "train150",
        client_override=None if args.execute else object(),
        read_only_store=True,
    )
    try:
        if args.execute:
            runner.client.max_retries = 2
            runner.client.max_transport_retries = 2
        rows = []
        for qid, instance in instances.items():
            originals = {
                (s.session_id, i): (t.role, t.content)
                for s in instance.sessions
                for i, t in enumerate(s.turns)
            }
            raw_count = 0
            for sid in runner.store.session_ids_for_user(qid):
                for turn in runner.store.turns_for_session(sid):
                    if originals.get((external_session_id(sid), turn.turn_index)) != (
                        turn.role,
                        turn.content,
                    ):
                        raise ValueError("raw history is not the exact public same-user corpus")
                    raw_count += 1
            public_sessions = {s.session_id for s in instance.sessions}
            memories = runner.store.iter_all(qid)
            for memory in memories:
                if (
                    memory.user_id != qid
                    or not memory.source_session_id
                    or external_session_id(memory.source_session_id) not in public_sessions
                ):
                    raise ValueError("derived memory lacks same-user public provenance")
            routed = personal_archive_review(instance.question)
            pages = (
                archive_pages(runner.store, qid, roles=("user",), split_oversized=True)
                if routed
                else None
            )
            if pages and not pages.complete:
                raise ValueError("incomplete archive preflight")
            rows.append(
                {
                    "question_id": qid,
                    "public_raw_turns": raw_count,
                    "public_derived_memories": len(memories),
                    "user_review": routed,
                    "pages": len(pages.pages) if pages else 0,
                    "review_tokens": sum(int(len(p.render()) / 4.6) for p in pages.pages)
                    if pages
                    else 0,
                }
            )
        expected = sum(r["pages"] for r in rows) + 2 * len(rows)
        if expected > 40:
            raise ValueError("registered answerer call ceiling exceeded")
        paths = [
            *list((REPO / "src").rglob("*.py")),
            Path(__file__),
            train_path,
            REPO / "tools/grounded_reader_diagnostic.py",
            REPO / "tools/grade_saved_recovering.py",
            REPO / "results/prereg-prerequisite-reader-v18-v1.md",
            REPO / "configs/fallback.yaml",
            REPO / "results/analysis/role-paged-growth-v3.train150.json",
            dataset,
            REPO / "stores/train150.db",
        ]
        if args.label.endswith("-v9"):
            paths.append(REPO / "results/prereg-prerequisite-reader-v18-v9.md")
        if args.label.endswith(("-v8", "-v9")):
            paths.append(REPO / "results/prereg-prerequisite-reader-v18-v8.md")
        if args.label.endswith(("-v7", "-v8", "-v9")):
            paths.append(REPO / "results/prereg-prerequisite-reader-v18-v7.md")
        if args.label.endswith(("-v6", "-v7", "-v8", "-v9")):
            paths.append(REPO / "results/prereg-prerequisite-reader-v18-v6.md")
        if args.label.endswith(("-v5", "-v6", "-v7", "-v8", "-v9")):
            paths.append(REPO / "results/prereg-prerequisite-reader-v18-v5.md")
        if args.label.endswith(("-v4", "-v5", "-v6", "-v7", "-v8", "-v9")):
            paths.append(REPO / "results/prereg-prerequisite-reader-v18-v4.md")
        if args.label.endswith(("-v3", "-v4", "-v5", "-v6", "-v7", "-v8", "-v9")):
            paths.append(REPO / "results/prereg-prerequisite-reader-v18-v3.md")
        if args.label.endswith(("-v2", "-v3", "-v4", "-v5", "-v6", "-v7", "-v8", "-v9")):
            paths.append(REPO / "results/prereg-prerequisite-reader-v18-v2.md")
        for stem in ("train150-index", "train150-turn-key-index"):
            paths.extend(
                REPO / "stores" / f"{stem}{suffix}"
                for suffix in (".npy", ".ids.json", ".manifest.json")
                if suffix != ".manifest.json" or (REPO / "stores" / f"{stem}{suffix}").exists()
            )
        identity = {
            "experiment_class": "exposed-train150-mechanism-development",
            "question_ids": list(QIDS),
            "baseline": "historical v15; no paired accuracy estimate",
            "models": cfg.models.model_dump(),
            "prompt_version": runner.answer_prompt_version,
            "sample_count": len(QIDS),
            "temperature": 0,
            "seed": "provider unsupported",
            "python": platform.python_version(),
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "files": {str(p.relative_to(REPO)): sha256_file(p) for p in paths},
            "preflight": rows,
            "max_nominal_reader_calls": expected,
            "judge_calls": len(rows),
            "recovery": {"max_recoveries": 3, "max_wait_seconds": 180},
            "provider_retry_policy": {
                "max_retries": 2,
                "max_transport_retries": 2,
                "timeout_seconds": 180,
            },
        }
        source_snapshot = REPO / f"results/analysis/{args.label}.source-snapshot.json"
        snapshot = {
            "files": {
                p: {"sha256": sha, "text": (REPO / p).read_text(encoding="utf-8")}
                for p, sha in identity["files"].items()
                if Path(p).suffix in {".py", ".yaml", ".md"}
            }
        }
        if source_snapshot.exists():
            if json.loads(source_snapshot.read_text(encoding="utf-8")) != snapshot:
                raise ValueError("source snapshot changed")
        elif args.execute:
            raise ValueError("create frozen source preview before execution")
        else:
            source_snapshot.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
        identity["source_snapshot_sha256"] = sha256_file(source_snapshot)
        base = REPO / f"results/raw/{args.label}"
        inventory_path = base.with_suffix(".execution.json")
        readers_path = base.with_suffix(".readers.jsonl")
        grades_path = base.with_suffix(".grades.jsonl")
        usage_path = base.with_suffix(".usage.json")
        if not args.execute:
            preview = REPO / f"results/analysis/{args.label}.preview.json"
            if preview.exists():
                raise ValueError("preview namespace exists")
            preview.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
            print(
                json.dumps(
                    {"provider_calls": 0, "preflight": rows, "max_nominal_reader_calls": expected}
                )
            )
            return 0
        preview = json.loads(
            (REPO / f"results/analysis/{args.label}.preview.json").read_text(encoding="utf-8")
        )
        if identity != preview:
            raise ValueError("preview identity changed")
        with _exclusive(readers_path):
            if inventory_path.exists():
                if json.loads(inventory_path.read_text(encoding="utf-8")) != identity:
                    raise ValueError("execution identity changed")
            else:
                if any(p.exists() for p in (readers_path, grades_path, usage_path)):
                    raise ValueError("unbound namespace")
                inventory_path.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
            digest = sha256_file(inventory_path)
            readers = rows_by_question(readers_path) if readers_path.exists() else {}
            grades = rows_by_question(grades_path) if grades_path.exists() else {}
            if not set(readers) <= set(QIDS) or any(
                r["inventory_sha256"] != digest for r in readers.values()
            ):
                raise ValueError("foreign readers")
            if (readers or grades) and not usage_path.exists():
                raise ValueError("missing usage provenance")
            if usage_path.exists():
                usage.records[:] = UsageTracker._load_records(usage_path)
            started = time.time()
            try:
                if args.grade:
                    if set(readers) != set(QIDS):
                        raise ValueError("complete readers before grading")
                    wrapped = RecoveringJudge(
                        judge, usage, usage_path, base.with_suffix(".recovery.json"), digest
                    )
                    with grades_path.open("a", encoding="utf-8") as sink:
                        grade_pending(
                            wrapped,
                            instances,
                            readers,
                            grades,
                            sink,
                            lambda: usage.save(usage_path),
                        )
                else:
                    with readers_path.open("a", encoding="utf-8") as sink:
                        for qid in QIDS:
                            if qid not in readers:
                                readers[qid] = save_reader(
                                    runner, instances[qid], sink, digest, "prerequisite-pages"
                                )
                                usage.save(usage_path)
                                print(
                                    json.dumps(
                                        {
                                            "reader_completed": len(readers),
                                            "total": len(QIDS),
                                            "question_id": qid,
                                        }
                                    ),
                                    flush=True,
                                )
            finally:
                usage.save(usage_path)
            if any(sha256_file(REPO / name) != sha for name, sha in identity["files"].items()):
                raise ValueError("source changed during execution")
            print(
                json.dumps(
                    {
                        "readers": len(readers),
                        "grades": len(grades),
                        "elapsed_seconds": time.time() - started,
                        "usage": usage.summary(),
                    }
                ),
                flush=True,
            )
    finally:
        runner.store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
