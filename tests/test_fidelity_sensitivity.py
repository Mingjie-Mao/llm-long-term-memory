"""The lenient normaliser used to measure the recall ruler's literal-match bias."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from fidelity_sensitivity import normalise


def test_number_words_become_digits():
    assert normalise("Three months") == "3 months"
    assert normalise("twenty-five miles") == "25 miles"
    assert normalise("forty two") == "42"


def test_thousands_separators_and_trailing_punctuation_go():
    assert normalise("$1,200,") == "$1200"
    assert normalise("2022.") == "2022"


def test_a_word_containing_a_number_word_is_left_alone():
    assert normalise("someone often") == "someone often"
