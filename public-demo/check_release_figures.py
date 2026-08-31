"""Verify that `release.json`'s engineering figures are the measured ones.

Run: `.venv/bin/python public-demo/check_release_figures.py`
Update after a real change: add `--update`.

**Why this is separate from `check_site.py`.** That script validates the page's
*shape* — the deploy surface, labels, translations, and that the manifest matches its
schema — and it deliberately runs with no project dependencies so CI can execute it in
seconds. It also asserts that a figure like "560 tests" is not written into the HTML,
which is right and was the original fix.

But that only moved the drift. Nothing checked whether the manifest's own numbers were
still true, so `tests: 560` was as hand-maintained as the HTML string had been — the
page simply now had a tidier place to be wrong. This runs the suite and compares.

**Why it is not a test in `tests/`.** A test asserting "the manifest says N tests"
changes N by existing. The check has to run after the suite, not inside it.

`source_commit` and `verified_at` are *recorded*, not asserted: they say when the
figures were last confirmed, and that is legitimately older than HEAD. `--update`
refreshes them together with the figures, so they can never drift apart from each
other even though they may lag the branch.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MANIFEST = HERE / "site" / "release.json"
PYTHON = REPO / ".venv" / "bin" / "python"


def measure() -> dict[str, int]:
    python = str(PYTHON) if PYTHON.exists() else sys.executable
    run = subprocess.run(
        # No `-q` here: pyproject's addopts already supplies one, and a second makes
        # it `-qq`, which suppresses the "N passed" summary this reads — the coverage
        # table then becomes the last line and the count silently disappears.
        [
            python,
            "-m",
            "pytest",
            "--no-header",
            "-p",
            "no:cacheprovider",
            "--cov=src/llm_long_term_memory",
            "--cov-report=term",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    passed = re.search(r"(\d+) passed", run.stdout)
    if not passed:
        print("FAIL: could not read a pass count from pytest")
        print(run.stdout[-1500:])
        raise SystemExit(1)
    total = re.search(r"^TOTAL\s+\d+\s+\d+\s+(\d+)%", run.stdout, re.M)
    if not total:
        print("FAIL: could not read a coverage total")
        print(run.stdout[-1500:])
        raise SystemExit(1)
    return {"tests": int(passed.group(1)), "line_coverage": int(total.group(1))}


def head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO
    ).stdout.strip()[:10]


def main() -> int:
    update = "--update" in sys.argv
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    claimed = manifest["engineering"]
    actual = measure()

    drift = {k: (claimed.get(k), v) for k, v in actual.items() if claimed.get(k) != v}

    if update:
        from datetime import UTC, datetime

        claimed.update(actual)
        claimed["source_commit"] = head()
        claimed["verified_at"] = datetime.now(UTC).date().isoformat()
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"updated: {actual} at {claimed['source_commit']} on {claimed['verified_at']}")
        return 0

    for key, (was, now) in drift.items():
        print(f"FAIL: release.json says {key}={was}, measured {now}")
    if drift:
        print("\nRun: python public-demo/check_release_figures.py --update")
        return 1

    print(
        f"release figures OK: {actual['tests']} tests, {actual['line_coverage']}% coverage "
        f"(recorded at {claimed.get('source_commit')} on {claimed.get('verified_at')})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
