"""The judge's version is shared with the product; its text must not drift from it."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from llm_long_term_memory.evaluation import judge
from llm_long_term_memory.prompt_versions import JUDGE_PROMPT_VERSION

# Recorded with the version. Changing any judge prompt changes this digest; when that
# happens, bump JUDGE_PROMPT_VERSION in `prompt_versions.py` and record the new pair.
JUDGE_PROMPTS = {"lme-type-aware-v2": "cf22600e2057d738"}


def test_the_judge_prompt_text_matches_its_version():
    text = "\x00".join(
        [judge.JUDGE_SYSTEM, judge._QA_PROMPT, judge._ABSTENTION_PROMPT, judge._PREFERENCE_PROMPT]
    )
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    assert JUDGE_PROMPTS.get(JUDGE_PROMPT_VERSION) == digest, (
        "the judge prompt changed without a new JUDGE_PROMPT_VERSION"
    )


def test_the_judge_re_exports_the_shared_version():
    assert judge.JUDGE_PROMPT_VERSION == JUDGE_PROMPT_VERSION


def test_the_product_does_not_import_evaluation():
    """The service and the answer engine must run without the benchmark package."""
    src = Path(__file__).resolve().parent.parent / "src" / "llm_long_term_memory"
    product = [src / "api" / "service.py", *sorted((src / "runtime").glob("*.py"))]
    pattern = re.compile(r"^\s*(from|import)\s+llm_long_term_memory\.evaluation\b", re.M)
    offenders = [p.name for p in product if pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == []
