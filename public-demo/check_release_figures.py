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

import hashlib
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
        encoding="utf-8",
        cwd=REPO,
    )
    if run.returncode != 0:
        print(f"FAIL: pytest exited with status {run.returncode}; figures were not verified")
        print(run.stdout[-1500:])
        print(run.stderr[-1500:])
        raise SystemExit(1)
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
    # The public label is "automated tests", not "passed tests". Four optional
    # artifact replays require a private store absent from CI. Count those collected
    # tests too, while requiring a successful suite exit above on every platform.
    skipped = re.search(r"(\d+) skipped", run.stdout)
    return {
        "tests": int(passed.group(1)) + (int(skipped.group(1)) if skipped else 0),
        "line_coverage": int(total.group(1)),
    }


# What the figures are a measurement *of*. Deliberately excludes `release.json` itself:
# a digest that covered the file it is written into could never be stable, which is the
# whole defect this replaces.
MEASURED = ("src/**/*.py", "tests/**/*.py", "pyproject.toml")


def measured_files() -> list[str]:
    """The measured files, as git sees them.

    Listed by `git ls-files` rather than by globbing the working tree. A glob picks up
    whatever happens to be on this machine, so an untracked file makes the digest differ
    from every other checkout by construction — which is exactly what happened here:
    `tests/data/*.py` is matched by the `data/` line in `.gitignore`, exists locally, has
    never been committed, and is imported by nothing. The digest has to describe what was
    published, not what is lying around.
    """
    listed = subprocess.run(
        ["git", "ls-files", "-z", *MEASURED],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=REPO,
    )
    if listed.returncode != 0:
        print("FAIL: could not list the measured files with git")
        print(listed.stderr[-500:])
        raise SystemExit(1)
    return sorted(name for name in listed.stdout.split("\0") if name)


def measured_tree() -> str:
    """A digest over the files that decide the test count and the coverage.

    `source_commit` cannot be trusted to stay reachable. It records the commit the
    figures were confirmed at, and writing it into the manifest changes that commit's
    SHA — so an amend, a squash or a rebase leaves it pointing at an object nobody can
    check out. It happened three times in one afternoon here.

    A content digest has no such problem. It is stable under any history rewrite, it can
    be recomputed by anyone with the tree, and it says the thing that actually matters:
    whether the code these numbers were measured on is the code in front of you.
    """
    digest = hashlib.sha256()
    for name in measured_files():
        # The name is hashed alongside the bytes so that moving a file changes the
        # digest; otherwise a rename that swaps two files would go unnoticed.
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((REPO / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=REPO,
    ).stdout.strip()[:10]


def main() -> int:
    update = "--update" in sys.argv
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    claimed = manifest["engineering"]
    actual = measure()
    tree = measured_tree()

    drift = {k: (claimed.get(k), v) for k, v in actual.items() if claimed.get(k) != v}
    # Checked separately from the figures so the message can say which one moved: a
    # changed tree with unchanged counts is a real state, and it means the numbers are
    # stale even though they still happen to match.
    stale_tree = claimed.get("measured_tree_sha256") != tree

    if update:
        from datetime import UTC, datetime

        claimed.update(actual)
        claimed["measured_tree_sha256"] = tree
        # Kept as a human pointer, and labelled as one. It is the commit that was checked
        # out when the figures were taken, which is useful to read and unsafe to rely on.
        claimed["source_commit_for_reference"] = head()
        claimed.pop("source_commit", None)
        claimed["verified_at"] = datetime.now(UTC).date().isoformat()
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"updated: {actual} on {claimed['verified_at']}")
        print(f"  measured tree: {tree[:16]}…")
        print(f"  reference commit: {claimed['source_commit_for_reference']}")
        return 0

    for key, (was, now) in drift.items():
        print(f"FAIL: release.json says {key}={was}, measured {now}")
    if stale_tree:
        print("FAIL: the measured source tree has changed since these figures were taken")
        print(f"  recorded: {claimed.get('measured_tree_sha256')}")
        print(f"  measured: {tree}")
    if drift or stale_tree:
        print("\nRun: python public-demo/check_release_figures.py --update")
        return 1

    print(
        f"release figures OK: {actual['tests']} tests, {actual['line_coverage']}% coverage "
        f"(tree {tree[:12]}…, verified {claimed.get('verified_at')})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
