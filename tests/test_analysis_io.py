"""The shared reader for evidence artifacts.

Each analyser used to carry its own copy, which is one more place to forget
`encoding="utf-8"` — not cosmetic here, because the answer rows carry non-ASCII text
and a machine reading them through the locale codec mis-decodes an evaluation artifact
instead of failing.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _module():
    path = REPO / "tools/analysis_io.py"
    spec = importlib.util.spec_from_file_location("analysis_io_for_tests", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rows_are_read_in_file_order_and_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"question_id": "b"}\n\n{"question_id": "a"}\n', encoding="utf-8")

    assert [row["question_id"] for row in _module().read_jsonl(path)] == ["b", "a"]


def test_non_ascii_evidence_survives_whatever_the_machine_locale_is(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"hypothesis": "一杯拿铁 · café"}\n', encoding="utf-8")

    assert _module().read_jsonl(path)[0]["hypothesis"] == "一杯拿铁 · café"


def test_a_duplicated_question_is_an_error_not_a_silently_overwritten_row(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"question_id": "q1", "correct": true}\n{"question_id": "q1", "correct": false}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate question_id"):
        _module().rows_by_question(path)


def test_the_digest_streams_a_file_and_matches_hashing_it_whole(tmp_path):
    import hashlib

    path = tmp_path / "artifact.bin"
    payload = b"x" * (3 << 20)
    path.write_bytes(payload)

    assert _module().sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_the_report_and_its_data_are_written_from_one_result(tmp_path):
    module = _module()
    result = {"questions": 16, "refused": 9}

    markdown = module.write_report(
        result,
        lambda r: f"# report\n\n{r['refused']} of {r['questions']} refused\n",
        json_out=tmp_path / "nested" / "out.json",
        md_out=tmp_path / "nested" / "out.md",
    )

    assert json.loads((tmp_path / "nested" / "out.json").read_text(encoding="utf-8")) == result
    assert (tmp_path / "nested" / "out.md").read_text(encoding="utf-8") == markdown
    assert "9 of 16 refused" in markdown
