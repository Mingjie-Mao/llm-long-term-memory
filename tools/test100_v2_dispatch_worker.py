"""Namespace-only launcher of the isolated, frozen historical v2 reader."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test100_v2_worker as worker

worker.STEM = "original-test100-v2-t4-dispatch-v1"

if __name__ == "__main__":
    raise SystemExit(worker.main())
