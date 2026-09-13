"""Prepare, rehearse and resume the source-bound 404-row v4.2 development comparison."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import tarfile
import time
from contextlib import ExitStack, closing, contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from v42_protocol import (
    ARMS,
    MAX_OBSERVED_TOKENS,
    MAX_REQUESTS,
    analyse,
    append_json,
    atomic_json,
    load_rows,
    schedule,
    sha,
    validate_rows,
)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_FREEZE = REPO / "results/frozen/v4.2-development-20260912"
CONFIG = "configs/v3-phase5-compact.yaml"
PROBES = "results/analysis/synthesis-probes.json"
SPLIT = "results/manifests/v4-probe-split.json"
ADDENDUM = "results/prereg-v4.2-execution-addendum.md"


def runtime_versions():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "google-genai",
                "numpy",
                "torch",
                "sentence-transformers",
                "transformers",
                "pydantic",
            )
        },
    }


def development():
    from probe_safety import select_probes

    raw = (REPO / PROBES).read_bytes()
    spec = json.loads(raw)
    return spec, select_probes(spec, raw, REPO / SPLIT, "development")


def prepare(root: Path) -> None:
    from huggingface_hub import snapshot_download
    from synthesis_probes import _fingerprint

    from llm_long_term_memory.config import Settings

    if root.exists():
        raise ValueError("freeze directory already exists; a freeze is never overwritten")
    settings = Settings()
    spec, probes = development()
    cells = schedule(probes)
    private = settings.store_dir.resolve() / root.name
    if private.exists():
        raise ValueError("private freeze directory already exists")
    private.mkdir(parents=True)
    source_db = (settings.store_dir / "train150.db").resolve()
    with closing(sqlite3.connect(source_db.as_uri() + "?mode=ro", uri=True)) as original:
        if _fingerprint(original) != spec["store_fingerprint"]:
            raise ValueError("probe store identity mismatch")
        with closing(sqlite3.connect(private / "train150.db")) as copied:
            original.backup(copied)
            # A WAL-mode source produces transient -wal/-shm sidecars. Finish the
            # snapshot as one closed database before hashing; connection context
            # managers alone commit but do not close sqlite connections.
            copied.execute("PRAGMA journal_mode=DELETE").fetchone()
            if copied.execute("pragma quick_check").fetchone()[0] != "ok":
                raise ValueError("database integrity check failed")
            db_ids = {r[0] for r in copied.execute("select id from memories")}
    for suffix in ("-index.ids.json", "-index.npy"):
        source = settings.store_dir / f"train150{suffix}"
        before = sha(source)
        shutil.copyfile(source, private / source.name)
        if before != sha(source) or before != sha(private / source.name):
            raise ValueError("source index changed during snapshot")
    ids = json.loads((private / "train150-index.ids.json").read_text(encoding="utf-8"))
    import numpy as np

    matrix = np.load(private / "train150-index.npy", mmap_mode="r")
    if set(ids) != db_ids or len(ids) != len(set(ids)) or matrix.shape != (len(ids), 384):
        raise ValueError("database/index mismatch")
    cached = Path(
        snapshot_download("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
    )
    shutil.copytree(cached, private / "encoder")
    root.mkdir(parents=True)
    paths = sorted(
        {
            *REPO.joinpath("src").rglob("*.py"),
            *REPO.joinpath("src").rglob("*.sql"),
            *REPO.joinpath("tools").glob("*.py"),
            REPO / CONFIG,
            REPO / PROBES,
            REPO / SPLIT,
            REPO / ADDENDUM,
            REPO / "results/prereg-v4.2-cited-enumeration.md",
            REPO / "results/analysis/predicate-map.csv",
            REPO / "pyproject.toml",
            REPO / "uv.lock",
        }
    )
    sources = [{"path": str(p.relative_to(REPO)), "sha256": sha(p)} for p in paths]
    with tarfile.open(root / "source.tar.gz", "w:gz") as archive:
        for p in paths:
            archive.add(p, arcname=str(p.relative_to(REPO)))
    data = [
        {"path": str(p.relative_to(private)), "bytes": p.stat().st_size, "sha256": sha(p)}
        for p in sorted(private.rglob("*"))
        if p.is_file()
    ]
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "working_tree_bound_by_hashes": True,
        "sources": sources,
        "source_archive_sha256": sha(root / "source.tar.gz"),
        "private_data": str(private),
        "data": data,
        "encoder_revision": cached.name,
        "device": "cpu",
        "runtime": runtime_versions(),
        "quota_state_dir": str((settings.store_dir / "quota").resolve()),
        "max_requests": MAX_REQUESTS,
        "max_observed_tokens": MAX_OBSERVED_TOKENS,
        "arms": ARMS,
        "cells": cells,
        "development_probe_count": len(probes),
        "held_out_count": 0,
    }
    atomic_json(root / "freeze.json", payload)
    print(
        json.dumps(
            {"prepared": str(root), "cells": len(cells), "freeze_id": sha(root / "freeze.json")}
        )
    )


def verify(root: Path) -> tuple[dict, str]:
    manifest = json.loads((root / "freeze.json").read_text(encoding="utf-8"))
    for item in manifest["sources"]:
        if sha(REPO / item["path"]) != item["sha256"]:
            raise ValueError(f"frozen source changed: {item['path']}")
    if sha(root / "source.tar.gz") != manifest["source_archive_sha256"]:
        raise ValueError("source archive changed")
    private = Path(manifest["private_data"])
    for item in manifest["data"]:
        if sha(private / item["path"]) != item["sha256"]:
            raise ValueError(f"frozen data changed: {item['path']}")
    if manifest["runtime"] != runtime_versions():
        raise ValueError("frozen runtime changed")
    if manifest["cells"] != schedule(development()[1]):
        raise ValueError("frozen schedule changed")
    return manifest, sha(root / "freeze.json")


class BudgetStopped(RuntimeError):
    pass


class Meter:
    """Reserve every SDK attempt durably, including retries, before it is sent."""

    def __init__(self, path: Path, max_requests: int, token_limit: int):
        self.path, self.max_requests, self.token_limit = path, max_requests, token_limit
        self.cell_id = ""
        self.events = load_rows(path)
        self.records = []
        starts = {e["attempt"] for e in self.events if e["event"] == "start"}
        ends = {e["attempt"] for e in self.events if e["event"] == "end"}
        if starts != ends:
            raise BudgetStopped("unresolved in-flight attempt; do not resend automatically")

    @property
    def attempts(self):
        return sum(e["event"] == "start" for e in self.events)

    @property
    def tokens(self):
        return sum(e.get("input_tokens", 0) + e.get("output_tokens", 0) for e in self.events)

    @contextmanager
    def measure(self, role: str, model: str):
        if self.attempts >= self.max_requests or self.tokens >= self.token_limit:
            raise BudgetStopped("request or observed-token budget reached")
        if self.summary()["failed_attempts"] >= 20:
            raise BudgetStopped("failed-attempt budget reached")
        number = self.attempts + 1
        start = {
            "event": "start",
            "attempt": number,
            "cell_id": self.cell_id,
            "role": role,
            "model": model,
            "at": datetime.now(UTC).isoformat(),
        }
        append_json(self.path, start)
        self.events.append(start)
        box = {"input_tokens": 0, "output_tokens": 0}
        began = time.perf_counter()
        error = None
        try:
            yield box
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            end = {
                "event": "end",
                "attempt": number,
                "cell_id": self.cell_id,
                **box,
                "ok": error is None,
                "error_type": error,
                "latency_ms": (time.perf_counter() - began) * 1000,
            }
            append_json(self.path, end)
            self.events.append(end)

    def summary(self):
        return {
            "attempts": self.attempts,
            "observed_tokens": self.tokens,
            "failed_attempts": sum(e["event"] == "end" and not e["ok"] for e in self.events),
        }


class DurableClient:
    """Replay already completed calls when a row was interrupted during its fallback."""

    def __init__(self, provider, path: Path, meter: Meter):
        self.provider, self.path, self.meter = provider, path, meter
        self.cache = {}
        for rec in load_rows(path):
            key = (rec["cell_id"], rec["call_index"])
            if key in self.cache:
                raise ValueError("duplicate durable completion")
            self.cache[key] = rec
        self.cell = None
        self.call_index = 0
        self.runner = None

    def begin(self, cell, runner):
        self.cell, self.runner, self.call_index = cell, runner, 0
        self.meter.cell_id = cell["cell_id"]

    def generate(self, **kwargs):
        from llm_long_term_memory.llm.client import Completion

        index = self.call_index
        self.call_index += 1
        payload = dict(kwargs)
        schema = payload.get("schema")
        if schema is not None:
            payload["schema"] = schema.model_json_schema()
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()
        key = (self.cell["cell_id"], index)
        if key in self.cache:
            rec = self.cache[key]
            if rec["request_sha256"] != digest:
                raise ValueError("cached request changed on resume")
            return Completion(**rec["completion"])
        if self.provider is None:
            kind = self.cell["kind"]
            reply = {"status": "answer", "answer": "3", "operation": kind}
            if kind == "count":
                reply["items"] = ["a book", "a scarf", "2 tickets"]
                if self.cell["arm"] == "candidate":
                    reply["member_labels"] = sorted(self.runner._context_labels)[:3]
            elif kind == "duration":
                reply.update(start_date="2022-01-01", end_date="2022-01-19")
            elif kind == "comparison":
                reply.update(
                    start_date="2023-05-20", end_date="2023-03-10", answer="B happened first."
                )
            else:
                reply["answer"] = "the current value"
            with self.meter.measure("answerer", "fake-provider"):
                result = Completion(json.dumps(reply), "fake-provider", 0, 0, 0, 1)
        else:
            result = self.provider.generate(**kwargs)
        rec = {
            "cell_id": key[0],
            "call_index": index,
            "request_sha256": digest,
            "completion": asdict(result),
        }
        append_json(self.path, rec)
        self.cache[key] = rec
        return result


def output_row(cell, probe, answer, flags, freeze_id, mode):
    from run_synthesis_probes import grade

    found, needed = set(answer.retrieved_ids), set(flags["required_ids"])
    graded = grade(probe, answer.text, answer.notes.get("answer_status"), flags["stale_values"])
    keys = (
        "answer_status",
        "reasoning_kind",
        "synthesis_operation",
        "synthesis_computation",
        "synthesis_missing_field",
        "answer_was_raw_structure",
        "missing_field_was_narration",
        "scan_route",
        "fallback_level",
    )
    return {
        **cell,
        "freeze_id": freeze_id,
        "mode": mode,
        "question": probe["question"],
        "namespace": probe["namespace"],
        "gold": probe["answer"],
        "text": answer.text,
        "verdict": graded.verdict,
        "parsed": graded.parsed,
        "detail": graded.detail,
        **{k: answer.notes.get(k) for k in keys},
        "retrieved_ids": answer.retrieved_ids,
        "evidence_found": len(found & needed),
        "evidence_needed": len(needed),
        "context_complete": needed <= found,
        "context_tokens": answer.context_tokens,
    }


def execute(root: Path, mode: str, stop_after: int = 0):
    from run_synthesis_probes import build_flags

    from llm_long_term_memory.cli import _build
    from llm_long_term_memory.config import ExperimentConfig, Settings
    from llm_long_term_memory.embed import Encoder
    from llm_long_term_memory.evaluation.datasets.longmemeval import Instance
    from llm_long_term_memory.llm import Limits, QuotaManager
    from llm_long_term_memory.llm.client import DailyQuotaExhausted, GeminiClient
    from llm_long_term_memory.locking import exclusive

    manifest, freeze_id = verify(root)
    folder = root / mode
    with exclusive(folder / "run", what="v4.2 comparison"), ExitStack() as stack:
        binding = folder / "binding.json"
        expected_binding = {"freeze_id": freeze_id, "mode": mode}
        if binding.exists() and json.loads(binding.read_text(encoding="utf-8")) != expected_binding:
            raise ValueError("output belongs to another freeze/mode")
        atomic_json(binding, expected_binding)
        if mode == "live":
            ready = json.loads((root / "rehearsal-ready.json").read_text(encoding="utf-8"))
            if ready["freeze_id"] != freeze_id or ready["rows"] != 404 or ready["gate_0"] != "PASS":
                raise ValueError("full rehearsal has not passed for this freeze")
            if sha(root / "rehearsal/conclusion.json") != ready["conclusion_sha256"]:
                raise ValueError("rehearsal conclusion changed")
        spec, probes = development()
        cells = manifest["cells"]
        rows_path = folder / "rows.jsonl"
        rows = load_rows(rows_path)
        done = validate_rows(rows, cells, probes, freeze_id, mode)
        meter = Meter(
            folder / "requests.jsonl", manifest["max_requests"], manifest["max_observed_tokens"]
        )
        settings = Settings()
        cfg = ExperimentConfig.from_yaml(REPO / CONFIG)
        provider = None
        if mode == "live" and len(done) < len(cells):
            quota = QuotaManager(
                state_dir=Path(manifest["quota_state_dir"]),
                default=Limits(rpm=cfg.quota.rpm, tpm=cfg.quota.tpm, rpd=cfg.quota.rpd),
            )
            quota.load_learned()
            limiter = quota.for_model(cfg.models.answerer)
            # Respect both the configured budget and any lower limit learned from 429s.
            limiter.limits = Limits(
                rpm=min(cfg.quota.rpm, limiter.limits.rpm),
                tpm=min(cfg.quota.tpm, limiter.limits.tpm),
                rpd=min(cfg.quota.rpd, limiter.limits.rpd),
            )
            provider = GeminiClient(
                settings.require_api_key(),
                quota=quota,
                usage=meter,
                max_retries=3,
                max_transport_retries=3,
            )
            stack.callback(provider._client.close)
        client = DurableClient(provider, folder / "completions.jsonl", meter)
        private = Path(manifest["private_data"])
        frozen_settings = settings.model_copy(update={"store_dir": private})
        encoder = Encoder(str(private / "encoder"), device="cpu")
        runners = {}
        for arm, variant in ARMS.items():
            _, _, runner, _, _ = _build(
                variant,
                str(REPO / CONFIG),
                store_name="train150",
                client_override=client,
                settings_override=frozen_settings,
                read_only_store=True,
            )
            runner.encoder = encoder
            runners[arm] = runner
            stack.callback(runner.store.close)
        flags = build_flags(probes, private / "train150.db")
        by_id = {p["probe_id"]: p for p in probes}
        status_path = folder / "status.json"
        status = {
            "freeze_id": freeze_id,
            "mode": mode,
            "rows": len(rows),
            "expected_rows": 404,
            "state": "running",
            "started_or_resumed_at": datetime.now(UTC).isoformat(),
        }
        atomic_json(status_path, {**status, **meter.summary()})
        try:
            for cell in cells[len(done) :]:
                p = by_id[cell["probe_id"]]
                runner = runners[cell["arm"]]
                client.begin(cell, runner)
                instance = Instance(
                    question_id=p["namespace"],
                    question_type=p["kind"],
                    question=p["question"],
                    answer=str(p["answer"]),
                    question_date=spec.get("question_date", "2026-09-06"),
                    sessions=[],
                    answer_session_ids=[],
                )
                ans = runner.answer(instance)
                row = output_row(cell, p, ans, flags[p["probe_id"]], freeze_id, mode)
                # Do not persist an invalid row into the resumable prefix.
                validate_rows([*rows, row], cells, probes, freeze_id, mode)
                append_json(rows_path, row)
                rows.append(row)
                status["rows"] = len(rows)
                atomic_json(status_path, {**status, **meter.summary()})
                if len(rows) % 10 == 0:
                    print(
                        json.dumps({"mode": mode, "rows": len(rows), **meter.summary()}), flush=True
                    )
                if stop_after and len(rows) >= stop_after:
                    status["state"] = "paused_by_row_limit"
                    break
            else:
                status["state"] = "complete"
        except (DailyQuotaExhausted, BudgetStopped) as exc:
            status.update(state="paused", reason=type(exc).__name__)
            if isinstance(exc, DailyQuotaExhausted):
                status["retry_after_seconds"] = exc.wait.seconds
        except Exception as exc:
            status.update(state="stopped", reason=type(exc).__name__)
            raise
        finally:
            atomic_json(status_path, {**status, **meter.summary()})
        if status["state"] == "complete":
            verify(root)
            # Read from the frozen copy, whose hash the manifest binds, so the
            # enumeration reading is computed against the same facts the run answered
            # over rather than against whatever the live store holds later.
            from diagnose_v4_probes import _gold_texts

            summary = analyse(
                rows, cells, probes, freeze_id, mode, _gold_texts(private / "train150.db")
            )
            summary["usage"] = meter.summary()
            summary["artifacts"] = [
                {"path": str(p.relative_to(root)), "sha256": sha(p)}
                for p in (rows_path, folder / "requests.jsonl", folder / "completions.jsonl")
            ]
            atomic_json(folder / "conclusion.json", summary)
            if mode == "rehearsal":
                if not summary["mechanism_observed"]:
                    raise ValueError("rehearsal did not exercise cited enumeration")
                atomic_json(
                    root / "rehearsal-ready.json",
                    {
                        "freeze_id": freeze_id,
                        "rows": len(rows),
                        "gate_0": "PASS",
                        "conclusion_sha256": sha(folder / "conclusion.json"),
                    },
                )
        print(json.dumps({**status, **meter.summary()}), flush=True)
        return 0 if status["state"] == "complete" else 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "rehearse", "run", "verify"))
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--stop-after", type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.stop_after < 0:
        parser.error("--stop-after must be nonnegative")
    if args.action == "prepare":
        prepare(args.freeze.resolve())
        return 0
    if args.action == "verify":
        _, fid = verify(args.freeze.resolve())
        print(json.dumps({"verified": fid}))
        return 0
    return execute(
        args.freeze.resolve(), "live" if args.action == "run" else "rehearsal", args.stop_after
    )


if __name__ == "__main__":
    raise SystemExit(main())
