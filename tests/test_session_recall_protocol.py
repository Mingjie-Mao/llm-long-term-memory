from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "session_recall.py"
    spec = importlib.util.spec_from_file_location("session_recall_protocol_for_tests", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


session_recall = _load_script()


def test_final_gate_requires_every_manifest_question_in_the_denominator():
    assert session_recall.gate_is_complete(
        ingest_complete=True,
        manifest_questions=150,
        scored_questions=150,
        skipped_without_gold=0,
    )
    assert not session_recall.gate_is_complete(
        ingest_complete=True,
        manifest_questions=150,
        scored_questions=149,
        skipped_without_gold=1,
    )
    assert not session_recall.gate_is_complete(
        ingest_complete=False,
        manifest_questions=150,
        scored_questions=103,
        skipped_without_gold=0,
    )
