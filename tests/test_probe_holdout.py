"""The held-out probes are the only unseen check v4 can have, and they can be spent once.

LongMemEval-S is exhausted, so `v4-hidden` cannot be cut from it. These 87 probes are the
substitute — narrower, because both halves come from the same store, but genuinely unseen
by the development loop. Every invariant below exists because breaking it would spend the
split without anyone noticing.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "results/manifests/v4-probe-split.json"
PROBES = REPO / "results/analysis/synthesis-probes.json"


@pytest.fixture(scope="module")
def split():
    if not MANIFEST.is_file():
        pytest.skip("no probe split recorded in this checkout")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def probes():
    return json.loads(PROBES.read_text(encoding="utf-8"))


def test_the_halves_are_disjoint_and_complete(split, probes):
    dev, held = set(split["development"]), set(split["held_out"])
    assert not dev & held, "a probe in both halves is in neither"
    assert dev | held == {p["probe_id"] for p in probes["probes"]}


def test_the_split_belongs_to_the_probe_set_it_names(split, probes):
    """A split carried to a regenerated set would silently reassign ids: they are
    positional, so `count_0000` survives regeneration as a different question."""
    assert split["store_fingerprint"] == probes["store_fingerprint"]
    import hashlib

    assert split["probe_set_sha256"] == hashlib.sha256(PROBES.read_bytes()).hexdigest()


def test_nothing_already_answered_is_held_out(split, probes):
    """The point of the held-out half is that the development loop has not seen it."""
    seen = set()
    for pattern in (
        "results/raw/probes*.jsonl",
        "results/archive/synthesis-probes-v1/probes*.jsonl",
    ):
        for path in glob.glob(str(REPO / pattern)):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        question = json.loads(line).get("question")
                        if question:
                            seen.add(question)
    by_id = {p["probe_id"]: p for p in probes["probes"]}
    leaked = [i for i in split["held_out"] if by_id[i]["question"] in seen]
    assert not leaked, f"{len(leaked)} held-out probes have already been answered"


def test_each_held_out_stratum_is_big_enough_to_report(split, probes):
    """A stratum of four questions produces a number nobody should act on, so the tool
    keeps such a kind whole rather than manufacturing one."""
    by_id = {p["probe_id"]: p for p in probes["probes"]}
    counts: dict[str, int] = {}
    for identifier in split["held_out"]:
        kind = by_id[identifier]["kind"]
        counts[kind] = counts.get(kind, 0) + 1
    for kind, count in counts.items():
        assert count >= 20, f"{kind} held out only {count}"


def test_the_manifest_says_what_it_is_not(split):
    """The caveat travels with the data. Both halves share a store, so this measures
    fitting the probe set, not fitting the store."""
    assert "not accuracy" in split["not_a_benchmark"]
    assert "once" in split["rule"]
