"""The release manifest must record something a history rewrite cannot orphan.

`source_commit` was the record of record, and it could not work: writing the commit into
the manifest changes the commit, so an amend, a squash or a rebase leaves it pointing at
an object nobody can check out. That happened three times in one afternoon before it was
replaced by a digest over the files the figures are a measurement of.

The two properties that make the digest work are opposites, and both are easy to break by
widening or narrowing the file set: it must ignore the manifest itself, and it must not
ignore the code.

**Freshness is deliberately not tested here.** `check_release_figures.py` already says
why — "a test asserting 'the manifest says N tests' changes N by existing" — and a digest
over `tests/**` makes that worse than the original: a test asserting the digest is current
runs *inside* the suite that `--update` has to pass before it may write the new digest, so
the manifest can never be refreshed again. That deadlock was reached once, here, by
ignoring the warning. Whether the manifest is stale is the tool's job, after the suite.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "public-demo"))

figures = pytest.importorskip("check_release_figures")

MANIFEST = REPO / "public-demo/site/release.json"


def test_the_digest_ignores_the_manifest_it_is_written_into():
    """The whole point. A digest covering its own file could never be stable, which is
    the defect this replaces rather than reproduces."""
    before = figures.measured_tree()
    original = MANIFEST.read_text(encoding="utf-8")
    payload = json.loads(original)
    payload["engineering"]["verified_at"] = "1999-01-01"
    try:
        MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        assert figures.measured_tree() == before
    finally:
        MANIFEST.write_text(original, encoding="utf-8")


def test_the_digest_moves_when_measured_code_moves():
    """A digest that ignored the code would be stable and worthless."""
    before = figures.measured_tree()
    probe = REPO / "src/llm_long_term_memory/config.py"
    original = probe.read_bytes()
    try:
        probe.write_bytes(original + b"\n# digest probe\n")
        assert figures.measured_tree() != before
    finally:
        probe.write_bytes(original)
    assert figures.measured_tree() == before


def test_the_digest_lists_files_through_git_not_the_working_tree():
    """A glob picks up whatever is on this machine. `tests/data/*.py` exists here, is
    matched by `.gitignore`, has never been committed and is imported by nothing — under
    a glob it made the digest differ from every clean checkout by construction."""
    listed = figures.measured_files()

    assert listed, "no measured files"
    assert all(not name.startswith("tests/data/") for name in listed), (
        "the digest is covering files git does not track"
    )
    assert listed == sorted(listed), "listing order must be stable across platforms"


def test_the_digest_covers_source_tests_and_the_test_configuration():
    """`pyproject.toml` carries pytest's addopts and the coverage settings, so it decides
    the recorded numbers as surely as the code does."""
    assert set(figures.MEASURED) == {"src/**/*.py", "tests/**/*.py", "pyproject.toml"}


def test_a_rename_changes_the_digest():
    """Bytes alone would let two files swap names unnoticed, so the path is hashed too."""
    import hashlib

    def digest(entries):
        h = hashlib.sha256()
        for name, data in entries:
            h.update(name.encode("utf-8"))
            h.update(b"\0")
            h.update(data)
            h.update(b"\0")
        return h.hexdigest()

    assert digest([("a.py", b"x"), ("b.py", b"y")]) != digest([("a.py", b"y"), ("b.py", b"x")])


def test_the_committed_manifest_records_a_digest_and_not_a_bare_commit():
    if not MANIFEST.is_file():
        pytest.skip("no release manifest in this checkout")
    engineering = json.loads(MANIFEST.read_text(encoding="utf-8"))["engineering"]

    assert re.fullmatch(r"[0-9a-f]{64}", engineering["measured_tree_sha256"])
    # A bare `source_commit` invites reliance on a SHA a rewrite can orphan. The
    # reference field says in its own name that it is not load-bearing.
    assert "source_commit" not in engineering
    reference = engineering.get("source_commit_for_reference")
    assert reference is None or re.fullmatch(r"[0-9a-f]{7,40}", reference)
