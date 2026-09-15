"""No document may render with its own emphasis markers still visible.

Chinese prose hits a CommonMark rule that English prose almost never does. A closing `**`
has to be "right-flanking": if the character before it is punctuation, the character after
it must be whitespace or punctuation. `**未晋级。**结果` fails that test — `。` before,
`结` after — so the whole run is left as literal text and the reader sees the asterisks.

The mistake is invisible to the author, because the sentence is correct and the intent is
obvious; it only shows up once rendered. `docs/history/REPORT.zh-CN.md` carried 58 of them.
The fix is always the same: put the punctuation outside the emphasis, `**未晋级**。结果`.

This renders every tracked Markdown file and fails on a surviving asterisk, so the next
Chinese document cannot reintroduce it silently.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SKIP = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache"}

MarkdownIt = pytest.importorskip("markdown_it").MarkdownIt

# Code spans and blocks legitimately contain asterisks (`configs/*.yaml`, a shell glob),
# so they are removed before the check rather than excused case by case.
_CODE = re.compile(r"<code>.*?</code>|<pre.*?</pre>", re.S)


def documents() -> list[Path]:
    """Every Markdown file **git tracks**, which is what the docstring above always
    claimed and the implementation did not do.

    Walking the working tree instead swept up whatever happened to be on disk: six
    `README.md` files inside downloaded encoder directories under the ignored `stores/`
    made this file contribute six more tests locally than in CI. That is not a cosmetic
    difference — `release.json` records the suite's size, and a figure that moves when
    somebody downloads a model is not a measurement of anything.

    Falls back to the walk when git is unavailable, so a source tarball still checks its
    own documents.
    """
    listing = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if listing.returncode == 0 and listing.stdout.strip():
        return sorted(REPO / name for name in listing.stdout.split())
    return sorted(
        path for path in REPO.rglob("*.md") if not SKIP & set(path.relative_to(REPO).parts)
    )


def test_there_are_documents_to_check():
    """A path typo that matched nothing would make every assertion below vacuous."""
    assert len(documents()) > 20


@pytest.mark.parametrize("document", documents(), ids=lambda p: p.relative_to(REPO).as_posix())
def test_no_emphasis_marker_survives_rendering(document: Path):
    renderer = MarkdownIt("commonmark", {"html": True}).enable("table").enable("strikethrough")
    rendered = _CODE.sub("", renderer.render(document.read_text(encoding="utf-8")))

    if "*" not in rendered:
        return
    # Name the offending line, because a file-level failure on a 1,700-line report is not
    # actionable. Rendered per paragraph: emphasis may legally span lines inside one.
    offenders = []
    paragraph: list[str] = []
    start = 1
    lines = document.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate([*lines, ""], 1):
        if line.strip():
            if not paragraph:
                start = number
            paragraph.append(line)
            continue
        if paragraph and "*" in _CODE.sub("", renderer.render("\n".join(paragraph))):
            offenders.append(f"{document.relative_to(REPO)}:{start}: {paragraph[0][:90]}")
        paragraph = []

    assert not offenders, (
        "emphasis markers reached the reader. Move the punctuation outside the markers "
        "(`**文字**。` not `**文字。**`):\n" + "\n".join(offenders)
    )


def test_the_document_list_does_not_depend_on_what_is_lying_around():
    """`release.json` records the suite's size. Walking the working tree made that figure
    move when somebody downloaded an encoder, because the model directories under the
    ignored `stores/` carry their own `README.md`. Six of them, six phantom tests, and a
    recorded number that CI could never reproduce."""
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8",
    )  # fmt: skip
    if tracked.returncode != 0:
        pytest.skip("not a git checkout")
    assert {p.relative_to(REPO).as_posix() for p in documents()} == set(tracked.stdout.split())
    assert not any("stores/" in p.as_posix() for p in documents())
