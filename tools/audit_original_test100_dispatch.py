"""Zero-call completion audit for the separate scoped-dispatch original test100."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
import audit_original_test100 as audit  # noqa: E402
from analysis_io import sha256_file  # noqa: E402
from run_original_test100_dispatch import ANALYSIS, STEM, configured  # noqa: E402
from t4_savings_regression import immutable_json  # noqa: E402

if __name__ == "__main__":
    with configured():
        result = audit.audit()
    result["profile_audit_generator_sha256"] = sha256_file(Path(__file__))
    immutable_json(ANALYSIS / f"{STEM}.completion-audit-v1.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "provider_calls",
                    "questions",
                    "reader_rows",
                    "grade_rows",
                    "gate_reproduces",
                    "accuracy_percent",
                )
            }
        )
    )
