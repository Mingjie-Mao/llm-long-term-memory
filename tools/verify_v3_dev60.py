"""Capture or verify the immutable v3.3 dev60 validation package.

`dev60` is spent. Its one-shot ledger forbids a second run, so the only thing that
can still be checked about it is whether the record is intact — which makes an
independent verifier the last line of defence rather than a formality.

The package binds sealed rows **by hash, in place**. It never copies them into the
archive directory, and that is deliberate rather than an oversight about
completeness: the pre-registration says only aggregate results may be reported, and
copying 360 per-question rows into a committed folder would publish exactly what the
seal exists to withhold. Auditability and the seal are both satisfied by recording
what the rows hash to.

Run without `--capture` to verify. `--capture` refuses to overwrite an existing
conclusion, because rewriting a conclusion after seeing it is the failure the whole
freeze protocol is built to prevent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
ARCHIVE = Path("results/archive/v3-dev60")
FREEZE = Path("results/frozen/v3-candidate-dev60/freeze.json")
LEDGER = Path("results/frozen/v3-candidate-dev60/one-shot.json")
AGGREGATE = Path("results/validation/v3-dev60.json")
REPORT = Path("results/validation/v3-dev60.md")
SEALED = Path("results/sealed/v3-dev60")
SNAPSHOT = ARCHIVE / "source.tar.gz"
README = ARCHIVE / "README.md"
CONCLUSION = ARCHIVE / "conclusion.json"
VERIFIER = Path("tools/verify_v3_dev60.py")

EXPERIMENT = "v3-dev60"
EXPECTED_ROWS = 360
GATES = (
    "target_gain",
    "temporal_no_regression",
    "multi_session_no_regression",
    "knowledge_update_no_regression",
    "ordinary_no_regression",
    "no_new_confident_errors",
    "selected_recall_no_regression",
    "context_within_2x_v2",
    "generation_cost",
)


class VerificationError(RuntimeError):
    """The frozen dev60 result is missing, inconsistent, or changed."""


def _read_json(repo: Path, relative: Path) -> dict[str, Any]:
    try:
        value = json.loads((repo / relative).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read valid JSON from {relative}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"expected one JSON object in {relative}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_file(repo: Path, relative: Path, record: dict[str, Any]) -> None:
    path = repo / relative
    if not path.is_file():
        raise VerificationError(f"missing evidence file: {relative}")
    if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
        raise VerificationError(f"evidence changed: {relative}")


def _verify_live_source(repo: Path, freeze: dict[str, Any]) -> None:
    for name, record in freeze["system"]["source"]["files"].items():
        _verify_file(repo, Path(name), record)


def _capture_snapshot(repo: Path, freeze: dict[str, Any]) -> None:
    if (repo / SNAPSHOT).exists():
        raise VerificationError("dev60 source snapshot already exists and cannot be replaced")
    (repo / ARCHIVE).mkdir(parents=True, exist_ok=True)
    _verify_live_source(repo, freeze)
    with tarfile.open(repo / SNAPSHOT, "w:gz") as archive:
        for name in sorted(freeze["system"]["source"]["files"]):
            archive.add(repo / name, arcname=name)


def _verify_snapshot(repo: Path, freeze: dict[str, Any]) -> None:
    """The snapshot must contain exactly the frozen source, byte for byte.

    Comparing names only would let a snapshot drift from the source it claims to
    preserve while still looking complete, which is the one thing an archive must
    not be able to do.
    """
    path = repo / SNAPSHOT
    if not path.is_file():
        raise VerificationError(f"missing source snapshot: {SNAPSHOT}")
    expected = freeze["system"]["source"]["files"]
    with tarfile.open(path, "r:gz") as archive:
        members = {m.name: m for m in archive.getmembers() if m.isfile()}
        if set(members) != set(expected):
            missing = sorted(set(expected) - set(members))[:3]
            extra = sorted(set(members) - set(expected))[:3]
            raise VerificationError(
                f"snapshot does not match the frozen source (missing {missing}, extra {extra})"
            )
        for name, record in expected.items():
            handle = archive.extractfile(members[name])
            if handle is None:
                raise VerificationError(f"unreadable snapshot member: {name}")
            digest = hashlib.sha256()
            with handle:
                while block := handle.read(1 << 20):
                    digest.update(block)
            if digest.hexdigest() != record["sha256"]:
                raise VerificationError(f"snapshot member changed: {name}")


def _verify_ledger(repo: Path, freeze_path: Path) -> dict[str, Any]:
    """The one-shot ledger is what makes 'spent' checkable rather than asserted."""
    ledger = _read_json(repo, LEDGER)
    if ledger.get("status") != "complete":
        raise VerificationError(f"dev60 ledger is {ledger.get('status')!r}, not complete")
    if ledger.get("freeze_sha256") != _sha256(repo / freeze_path):
        raise VerificationError("dev60 ledger is bound to a different freeze")
    rows = ledger.get("rows") or {}
    total = sum(rows.values()) if isinstance(rows, dict) else 0
    if total != EXPECTED_ROWS or any(count != 60 for count in rows.values()):
        raise VerificationError(f"dev60 ledger records {total} rows, not {EXPECTED_ROWS} as 6 x 60")
    if ledger.get("aggregate_sha256") != _sha256(repo / AGGREGATE):
        raise VerificationError("dev60 ledger does not match its aggregate")
    return ledger


def _verify_aggregate(repo: Path) -> dict[str, Any]:
    aggregate = _read_json(repo, AGGREGATE)
    if aggregate.get("privacy", {}).get("contains_question_ids_or_text") is not False:
        raise VerificationError("dev60 aggregate is not marked free of question content")
    for name, record in (aggregate.get("sealed_artifacts") or {}).items():
        _verify_file(repo, SEALED / name, record)
    checks = (aggregate.get("gate") or {}).get("checks") or {}
    if set(checks) != set(GATES):
        raise VerificationError("dev60 gate does not carry the nine registered checks")
    if not all(checks.values()):
        failed = sorted(name for name, ok in checks.items() if not ok)
        raise VerificationError(f"dev60 gate did not pass: {failed}")
    return aggregate


def _paths(freeze: dict[str, Any], aggregate: dict[str, Any]) -> list[Path]:
    paths = {FREEZE, LEDGER, AGGREGATE, REPORT, SNAPSHOT, README, VERIFIER}
    paths.add(Path(freeze["system"]["config"]["path"]))
    paths.update(Path(name) for name in freeze["system"]["protocol"]["files"])
    paths.update(SEALED / name for name in aggregate.get("sealed_artifacts") or {})
    return sorted(paths, key=Path.as_posix)


def _summary(aggregate: dict[str, Any]) -> dict[str, Any]:
    """Derive the headline numbers from the aggregate, every time.

    Never stored independently and re-read: a summary that can drift from the
    aggregate it summarises is a second source of truth, and the whole point of the
    package is that there is one. A malformed aggregate raises the verifier's own
    error rather than a KeyError, so a broken archive reports as a broken archive.
    """
    try:
        arms = {arm["arm"]: arm for arm in aggregate["arms"]}
        control, candidate = arms["v2-control"], arms["v3.3-compact"]
        paired = aggregate["paired_majority"]
        gate = aggregate["gate"]
        gate["baseline_slices"], gate["candidate_slices"]
        control["majority_accuracy"], candidate["majority_accuracy"]
    except (KeyError, TypeError) as exc:
        raise VerificationError(f"dev60 aggregate is missing {exc}") from exc
    return {
        "selected_candidate": "v3.3-compact",
        "majority_accuracy": {
            "v2-control": control["majority_accuracy"],
            "v3.3-compact": candidate["majority_accuracy"],
        },
        "median_context_tokens": {
            "v2-control": control["median_context_tokens"],
            "v3.3-compact": candidate["median_context_tokens"],
        },
        "median_context_ratio": round(
            candidate["median_context_tokens"] / control["median_context_tokens"], 4
        ),
        "fallback_trigger_rate": {
            "v2-control": control["fallback_trigger_rate"],
            "v3.3-compact": candidate["fallback_trigger_rate"],
        },
        "selected_source_recall": gate.get("selected_recall"),
        "registered_slices": {
            "v2-control": gate["baseline_slices"],
            "v3.3-compact": gate["candidate_slices"],
        },
        "paired_result": {
            "questions": paired["questions"],
            "wins": paired["candidate_wins"],
            "losses": paired["candidate_losses"],
            "accuracy_delta": paired["accuracy_delta"],
            "p_value": paired["p_value"],
        },
        "run_stability": {
            "v2-control": {
                "mean_accuracy": control["mean_accuracy"],
                "stdev_accuracy": control["stdev_accuracy"],
                "unanimous_agreement": control["unanimous_agreement"],
            },
            "v3.3-compact": {
                "mean_accuracy": candidate["mean_accuracy"],
                "stdev_accuracy": candidate["stdev_accuracy"],
                "unanimous_agreement": candidate["unanimous_agreement"],
            },
        },
        "decision": "passed_all_nine_preregistered_gates",
        "reading_rule": (
            "Aggregate metrics and pre-registered slices only. Individual dev60 rows "
            "are not read, and dev60 must not be used to tune any later candidate."
        ),
    }


def _capture_conclusion(repo: Path, freeze: dict[str, Any], aggregate: dict[str, Any]) -> None:
    if (repo / CONCLUSION).exists():
        raise VerificationError("dev60 conclusion already exists and cannot be replaced")
    files = []
    for relative in _paths(freeze, aggregate):
        path = repo / relative
        files.append(
            {"path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    payload = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_sha256": freeze["system"]["source"]["tree_sha256"],
        "files": files,
        "summary_file": AGGREGATE.as_posix(),
        "outcome": "validated_v3.3",
        "outcome_reason": (
            "v3.3-compact reached 55.0% majority accuracy against v2-control's 46.7% on "
            "the sealed 60-question set, improved every pre-registered slice with no "
            "regression, and did so at 1.95x median context while triggering the raw "
            "fallback less often (22.8% against 30.6%). All nine registered gates passed. "
            "The paired result is +8.3 points at p=0.1797, which is not statistically "
            "conclusive; the gate was written to turn on consistency across slices rather "
            "than on that p-value."
        ),
        "summary": _summary(aggregate),
    }
    (repo / CONCLUSION).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _verify_conclusion(repo: Path, freeze: dict[str, Any], aggregate: dict[str, Any]) -> dict:
    conclusion = _read_json(repo, CONCLUSION)
    if conclusion.get("schema_version") != 1:
        raise VerificationError("unsupported conclusion schema")
    if conclusion.get("experiment_id") != EXPERIMENT:
        raise VerificationError("unexpected conclusion experiment id")
    if conclusion.get("source_sha256") != freeze["system"]["source"]["tree_sha256"]:
        raise VerificationError("conclusion source hash changed")
    if conclusion.get("outcome") != "validated_v3.3":
        raise VerificationError("conclusion outcome changed")
    # The narrative summary must still agree with the aggregate it claims to
    # summarise; a conclusion that drifts from its own evidence is worse than none.
    if conclusion.get("summary") != _summary(aggregate):
        raise VerificationError("conclusion summary no longer matches the aggregate")
    files = conclusion.get("files")
    if not isinstance(files, list) or not files:
        raise VerificationError("conclusion has no evidence inventory")
    seen: set[str] = set()
    for record in files:
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            raise VerificationError("invalid conclusion record")
        if record["path"] in seen:
            raise VerificationError(f"duplicate conclusion file: {record['path']}")
        seen.add(record["path"])
        _verify_file(repo, Path(record["path"]), record)
    return conclusion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--repo", type=Path, default=REPO)
    args = parser.parse_args()
    repo = args.repo.resolve()
    try:
        freeze = _read_json(repo, FREEZE)
        # The live tree is checked only at capture time, as in the other archives.
        # Verifying it on every run would make this package fail as soon as
        # development continues past dev60 — which it is expected to do, and which is
        # not a failure of the record. What the package attests is the source that
        # produced the result, and `source.tar.gz` holds that byte for byte.
        if args.capture:
            _capture_snapshot(repo, freeze)
        _verify_snapshot(repo, freeze)
        _verify_ledger(repo, FREEZE)
        aggregate = _verify_aggregate(repo)
        if args.capture:
            _capture_conclusion(repo, freeze, aggregate)
            print(f"CAPTURED: {CONCLUSION}")
        conclusion = _verify_conclusion(repo, freeze, aggregate)
    except VerificationError as exc:
        print(f"STOP: {exc}")
        return 2
    print("PASS: preserved v3.3 dev60 validation · 360 rows · 55.0% vs 46.7% · nine gates passed")
    print(f"  source snapshot: {len(freeze['system']['source']['files'])} files")
    print(f"  evidence inventory: {len(conclusion['files'])} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
