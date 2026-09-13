"""The lightweight schema path must not load provider-only dependencies."""

from __future__ import annotations

import subprocess
import sys


def test_importing_ingest_schemas_does_not_load_the_provider_sdk():
    probe = (
        "import sys; "
        "from llm_long_term_memory.ingest.schemas import SINGLE_VALUED_PREDICATES; "
        "assert SINGLE_VALUED_PREDICATES; "
        "assert 'google.genai' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stderr
