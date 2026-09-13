"""Import `app.py` in a throwaway environment built from `requirements.txt` alone.

Run before deploying: `.venv/bin/python demo-api/check_imports.py`
Takes about a minute and needs network — it creates a virtualenv and installs into it.

**Why this exists.** The first Render deploy failed on line 55 of `app.py`, importing
`llm_long_term_memory.ingest.schemas` for a list of four predicate names. The package
initializer used to reach the extractor and its provider SDK. The initializer is now
lazy, so this check also proves the LLM-free service does not need `google-genai`.

The defect was not the missing line. It was that `smoke.py` ran in a development
checkout with every optional extra installed, so all 27 of its assertions passed
against an import graph the deployment would never have.

**Why a real environment rather than a simulated one.** The first version of this check
blocked module names not named in `requirements.txt` and allow-listed the transitive
ones. That is whack-a-mole: it failed on `annotated_types`, then `annotated_doc`, and
the set of packages pip happens to pull is not the same on this machine as on the
target anyway. Installing the declared set into an empty interpreter reproduces the
deployment instead of guessing at it, and costs a minute.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

PROBE = """
import sys
sys.path.insert(0, {demo!r})
sys.path.insert(0, {src!r})
import app
assert "google.genai" not in sys.modules, "demo import unexpectedly loaded Google SDK"
print("imported app.py")
print("  routes:", len([r for r in app.app.routes if getattr(r, "methods", None)]))
print("  Google SDK loaded: no")
"""


def run(*cmd: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def main() -> int:
    uv = shutil.which("uv")
    if not uv:
        print("SKIP: uv not found; cannot build an isolated environment")
        return 0

    tmp = Path(tempfile.mkdtemp(prefix="lltm-deploy-env-"))
    venv = tmp / "venv"
    try:
        print(f"building an environment from demo-api/requirements.txt in {tmp} …")
        r = run(uv, "venv", str(venv), "--python", "3.12")
        if r.returncode:
            print(r.stderr.strip()[-800:])
            return 1

        r = run(
            uv,
            "pip",
            "install",
            "--python",
            str(venv / "bin" / "python"),
            "-r",
            str(HERE / "requirements.txt"),
        )
        if r.returncode:
            print("FAIL: the declared dependencies do not install")
            print(r.stderr.strip()[-1500:])
            return 1
        installed = [
            line for line in r.stderr.splitlines() if "Installed" in line or "Prepared" in line
        ]
        print("  " + ("; ".join(installed) if installed else "installed"))

        probe = tmp / "probe.py"
        probe.write_text(PROBE.format(demo=str(HERE), src=str(REPO / "src")), encoding="utf-8")
        r = run(str(venv / "bin" / "python"), str(probe))
        if r.returncode:
            print("\nFAIL: app.py does not import under the declared dependencies\n")
            print(r.stdout.strip())
            print(r.stderr.strip()[-2000:])
            print("\nAdd whatever it names to demo-api/requirements.txt.")
            return 1

        print("\nOK — " + r.stdout.strip().replace("\n", "\n     "))
        print(
            "\nThis is what Render will do. A pass here means the build will not fail on imports."
        )
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
