"""Turn a complete train150 store into one reproducible v2 candidate config.

This script makes no LLM calls and refuses partial ingestion.  It runs the final
zero-yield audit, evaluates a fixed train-only context grid, applies the selection
rule in ``evaluation.context_selection``, writes ``configs/v2.yaml`` once, then
reruns the selected configuration as a verification pass.

The grid is intentionally small and was recorded before all 150 questions were
available.  It covers the provisional winner and nearby controls without fitting
new knobs to individual failures after the full result appears.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from llm_long_term_memory.config import ExperimentConfig, Settings  # noqa: E402
from llm_long_term_memory.evaluation.context_selection import (  # noqa: E402
    ContextCandidate,
    NoEligibleContext,
    choose_context_candidate,
)
from llm_long_term_memory.evaluation.datasets import longmemeval as lme  # noqa: E402
from llm_long_term_memory.evaluation.ingest_state import inspect_ingest_state  # noqa: E402
from llm_long_term_memory.evaluation.manifest import load_manifest  # noqa: E402
from llm_long_term_memory.evaluation.reproducibility import sha256_file  # noqa: E402
from llm_long_term_memory.ingest import fingerprint, namespaced_sessions  # noqa: E402
from llm_long_term_memory.ingest.pipeline import (  # noqa: E402
    IngestProgress,
    _key,
    resolved_sessions_per_request,
)
from llm_long_term_memory.llm import Limits, QuotaManager  # noqa: E402

GRID = (
    ("max-whole-cap20", "max", None, 20),
    ("mean-whole-cap20", "mean", None, 20),
    ("mean-r1-cap20", "mean", 1, 20),
    ("mean-r1-cap30", "mean", 1, 30),
    ("mean-r1-cap40", "mean", 1, 40),
    ("mean-r2-cap30", "mean", 2, 30),
    ("max-r1-cap30", "max", 1, 30),
)


@dataclass(slots=True)
class StagedConfig:
    """A candidate config that is invisible at its final path until verified."""

    candidate_path: Path
    output_path: Path
    needs_publish: bool

    def publish(self) -> None:
        if self.needs_publish:
            self.candidate_path.replace(self.output_path)
            self.needs_publish = False

    def cleanup(self) -> None:
        if self.needs_publish:
            self.candidate_path.unlink(missing_ok=True)


def stage_config(config: ExperimentConfig, output_path: Path) -> StagedConfig:
    """Stage a new config, or validate an already-published identical config."""
    if output_path.exists():
        existing = ExperimentConfig.from_yaml(output_path)
        if existing.model_dump() != config.model_dump():
            raise ValueError(
                f"{output_path} already exists with a different resolved config; "
                "automatic refreezing is forbidden"
            )
        return StagedConfig(output_path, output_path, needs_publish=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.stem}-candidate-",
        suffix=output_path.suffix,
    )
    os.close(descriptor)
    candidate_path = Path(temporary)
    try:
        config.to_yaml(candidate_path)
    except Exception:
        candidate_path.unlink(missing_ok=True)
        raise
    return StagedConfig(candidate_path, output_path, needs_publish=True)


def _run(command: list[str], environment: dict[str, str]) -> None:
    result = subprocess.run(command, cwd=REPO, env=environment)
    if result.returncode:
        raise RuntimeError(f"command stopped with exit {result.returncode}: {' '.join(command)}")


def _candidate(name: str, aggregate: str, radius: int | None, cap: int, payload: dict):
    assembled = payload["assembled_by_aggregate"][aggregate]
    return ContextCandidate(
        name=name,
        aggregate=aggregate,
        window_radius=radius,
        max_total_memories=cap,
        rank_top3=float(payload["recall_by_aggregate"][aggregate]["3"]),
        assembled_recall=float(assembled["recall"]),
        median_context_tokens=float(assembled["median_tokens"]),
        flat_median_tokens=float(payload["flat_median_tokens"]),
        truncated_questions=int(assembled["truncated_questions"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default="train150")
    parser.add_argument("--manifest-name", default="train150.json")
    parser.add_argument("--base-config", type=Path, default=Path("configs/fallback.yaml"))
    parser.add_argument("--output-config", type=Path, default=Path("configs/v2.yaml"))
    args = parser.parse_args()
    if args.store != "train150" or args.manifest_name != "train150.json":
        print("STOP: final selection is restricted to train150", file=sys.stderr)
        return 2

    settings = Settings()
    manifest_path = REPO / "results" / "manifests" / args.manifest_name
    manifest = load_manifest(manifest_path)
    by_id = {item.question_id: item for item in lme.load(manifest.variant, settings.data_dir)}
    instances = [by_id[qid] for qid in manifest.question_ids]
    pairs = namespaced_sessions(instances)
    expected = {_key(namespace, session) for namespace, session in pairs}
    progress = IngestProgress.load(settings.store_dir / f"{args.store}-ingest.json")
    terminal = progress.done_sessions | progress.blocked_sessions
    if terminal != expected or progress.done_sessions & progress.blocked_sessions:
        print(
            f"STOP: train150 is not complete: {len(terminal):,}/{len(expected):,} terminal; "
            "no final artifact was written",
            file=sys.stderr,
        )
        return 2

    base_config = ExperimentConfig.from_yaml(REPO / args.base_config)
    if base_config.context.window_radius is not None:
        print(
            "STOP: the fixed grid assumes the base config's whole-session radius is null",
            file=sys.stderr,
        )
        return 2

    quota = QuotaManager(
        state_dir=settings.store_dir / "quota",
        default=Limits(
            rpm=base_config.quota.rpm,
            tpm=base_config.quota.tpm,
            rpd=base_config.quota.rpd,
        ),
    )
    quota.load_learned()
    batch_size = resolved_sessions_per_request(
        base_config.ingest.sessions_per_request,
        quota.for_model(base_config.models.extractor).limits.tpm,
    )
    expected_fingerprint = fingerprint.from_config(
        base_config, sessions_per_request=batch_size
    ).as_dict()
    try:
        ingest_state = inspect_ingest_state(
            store_dir=settings.store_dir,
            store_name=args.store,
            pairs=pairs,
            expected_fingerprint=expected_fingerprint,
            batch_size=batch_size,
            embedding_dim=base_config.models.embedding_dim,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"STOP: train150 integrity check failed: {exc}", file=sys.stderr)
        return 2
    if not ingest_state.complete:
        details = "; ".join(ingest_state.issues) or "store is not fully terminal"
        print(f"STOP: train150 integrity check failed: {details}", file=sys.stderr)
        return 2

    analysis = REPO / "results" / "analysis"
    grid_dir = analysis / "train150-context-grid-final"
    analysis.mkdir(parents=True, exist_ok=True)
    grid_dir.mkdir(parents=True, exist_ok=True)
    ingest_state_path = analysis / "train150-ingest-state.final.json"
    ingest_state_path.write_text(
        json.dumps(ingest_state.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["HF_HUB_OFFLINE"] = "1"

    zero_path = analysis / "train150-zero-yield.final.json"
    zero_markdown = analysis / "train150-zero-yield.final.md"
    pilot_path = REPO / "results" / "raw" / "batch-position-pilot.json"
    try:
        _run(
            [
                sys.executable,
                str(REPO / "scripts" / "audit_zero_yield.py"),
                "--store",
                args.store,
                "--questions",
                str(REPO / "results" / "manifests" / args.manifest_name),
                "--output",
                str(zero_path),
                "--markdown",
                str(zero_markdown),
                "--pilot",
                str(pilot_path),
                "--require-complete",
            ],
            environment,
        )
    except RuntimeError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2
    zero_report = json.loads(zero_path.read_text(encoding="utf-8"))
    if not zero_report.get("complete"):
        print("STOP: zero-yield audit is not complete", file=sys.stderr)
        return 2

    candidates: list[ContextCandidate] = []
    artifacts: dict[str, Path] = {}
    for name, aggregate, radius, cap in GRID:
        artifact = grid_dir / f"{name}.json"
        command = [
            sys.executable,
            str(REPO / "scripts" / "session_recall.py"),
            "--store",
            args.store,
            "--questions",
            args.manifest_name,
            "--config",
            str(REPO / args.base_config),
            "--aggregate",
            aggregate,
            "--max-total-memories",
            str(cap),
            "--out",
            str(artifact),
        ]
        if radius is not None:
            command.extend(["--window-radius", str(radius)])
        try:
            _run(command, environment)
        except RuntimeError as exc:
            print(f"STOP: {exc}", file=sys.stderr)
            return 2
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        if not payload.get("complete") or payload.get("evaluated_questions") != 150:
            print(f"STOP: grid arm {name} is not a complete 150-question result", file=sys.stderr)
            return 2
        candidates.append(_candidate(name, aggregate, radius, cap, payload))
        artifacts[name] = artifact

    try:
        selected = choose_context_candidate(candidates)
    except NoEligibleContext as exc:
        selection_path = analysis / "train150-context-selection.final.json"
        selection_path.write_text(
            json.dumps(
                {
                    "status": "STOP",
                    "reason": str(exc),
                    "candidates": [asdict(candidate) for candidate in candidates],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"STOP: {exc}; no v2 config was written", file=sys.stderr)
        return 2

    output_config = REPO / args.output_config
    config = base_config
    config.name = "v2"
    config.description = "train150-selected v2 candidate; frozen before dev100"
    config.context.aggregate = selected.aggregate
    config.context.window_radius = selected.window_radius
    config.context.max_total_memories = selected.max_total_memories
    try:
        staged = stage_config(config, output_config)
    except (OSError, TypeError, ValueError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2

    verification = analysis / "train150-session-recall.final.json"
    try:
        _run(
            [
                sys.executable,
                str(REPO / "scripts" / "session_recall.py"),
                "--store",
                args.store,
                "--questions",
                args.manifest_name,
                "--config",
                str(staged.candidate_path),
                "--out",
                str(verification),
            ],
            environment,
        )
    except RuntimeError as exc:
        staged.cleanup()
        print(f"STOP: selected-config verification failed: {exc}", file=sys.stderr)
        return 2
    try:
        verified = json.loads(verification.read_text(encoding="utf-8"))
        verified_top3 = float(verified["recall_at"]["3"])
        verified_assembled = float(verified["assembled_by_aggregate"][selected.aggregate]["recall"])
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        staged.cleanup()
        print(f"STOP: selected-config verification artifact is invalid: {exc}", file=sys.stderr)
        return 2
    if verified_top3 != selected.rank_top3 or verified_assembled != selected.assembled_recall:
        staged.cleanup()
        print("STOP: selected-config verification disagrees with its grid arm", file=sys.stderr)
        return 2
    try:
        staged.publish()
    except OSError as exc:
        staged.cleanup()
        print(f"STOP: verified config could not be published: {exc}", file=sys.stderr)
        return 2

    selection_path = analysis / "train150-context-selection.final.json"
    selection_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "selection_rule": {
                    "min_top3": 0.80,
                    "min_assembled_recall": 0.80,
                    "max_median_context_ratio": 1.50,
                    "tie_break": [
                        "higher assembled recall",
                        "higher Top-3 recall",
                        "smaller hard memory cap",
                        "smaller window radius",
                        "lower median context tokens",
                    ],
                },
                "grid_declared_in_source": [list(item) for item in GRID],
                "candidates": [asdict(candidate) for candidate in candidates],
                "selected": asdict(selected),
                "artifacts": {
                    name: {
                        "path": str(path.relative_to(REPO)),
                        "sha256": sha256_file(path),
                    }
                    for name, path in artifacts.items()
                },
                "zero_yield_audit": {
                    "path": str(zero_path.relative_to(REPO)),
                    "sha256": sha256_file(zero_path),
                },
                "zero_yield_summary": {
                    "path": str(zero_markdown.relative_to(REPO)),
                    "sha256": sha256_file(zero_markdown),
                },
                "batch_position_pilot": {
                    "path": str(pilot_path.relative_to(REPO)),
                    "sha256": sha256_file(pilot_path),
                },
                "ingest_state": {
                    "path": str(ingest_state_path.relative_to(REPO)),
                    "sha256": sha256_file(ingest_state_path),
                },
                "verification": {
                    "path": str(verification.relative_to(REPO)),
                    "sha256": sha256_file(verification),
                },
                "output_config": {
                    "path": str(output_config.relative_to(REPO)),
                    "sha256": sha256_file(output_config),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"PASS: selected {selected.name}: Top-3 {selected.rank_top3:.1%}, "
        f"assembled {selected.assembled_recall:.1%}, median "
        f"{selected.median_context_tokens:.0f} tokens"
    )
    print(f"  config: {output_config}")
    print(f"  record: {selection_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
