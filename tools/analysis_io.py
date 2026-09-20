"""Reading evidence and writing a report, once.

Every analyser in this directory had grown its own JSONL reader. They agreed, but each
was a separate place to forget `encoding="utf-8"` — which is not cosmetic here: the
answer rows carry non-ASCII text, CI runs with `PYTHONWARNDEFAULTENCODING=1`, and a
Windows machine reading them through the locale codec mis-decodes an evaluation
artifact rather than failing.

Import it the way the other cross-tool modules in this directory are imported:

    sys.path.insert(0, str(REPO / "tools"))
    from analysis_io import read_jsonl, write_report
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path


def read_json(path: Path):
    """One JSON document."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    """The rows of a JSONL artifact, in file order, blank lines skipped."""
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def rows_by_question(path: Path) -> dict[str, dict]:
    """A run's rows keyed by question id.

    A duplicate id is an error: it means a resumed run recorded the same question
    twice, and every count taken from the result would be silently wrong.
    """
    rows: dict[str, dict] = {}
    for row in read_jsonl(path):
        qid = row["question_id"]
        if qid in rows:
            raise ValueError(f"duplicate question_id {qid!r} in {path}")
        rows[qid] = row
    return rows


def sha256_file(path: Path) -> str:
    """The digest of a file's bytes, read in blocks so a large artifact is streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_report(
    result: dict,
    render: Callable[[dict], str],
    *,
    json_out: Path | None = None,
    md_out: Path | None = None,
) -> str:
    """Write an analysis as both its data and its prose, and return the prose.

    Both files come from the same `result`, so the document can never describe a run
    the JSON does not. Parent directories are created; nothing else is written.
    """
    markdown = render(result)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if md_out is not None:
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(markdown, encoding="utf-8")
    return markdown
