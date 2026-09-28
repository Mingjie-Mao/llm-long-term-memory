"""How much of the recall ruler's "lost" is spelling, and how much is task sessions. No calls.

    python tools/fidelity_sensitivity.py stores/v2b-gate16.db results/manifests/v2b-gate16.json

The recall ruler (`ingest.fidelity`) matches literally: "three months" stored as
"3 months" counts as lost, and so does "1,200" stored as "1200". That was a deliberate
choice — a paired comparison on one ruler is valid whatever the ruler's bias — but it
leaves the absolute level unknown. This measures the bias instead of changing the
ruler: every published figure keeps its literal definition.

Two readings per store, over the same stated specifics:

* **literal** — the ruler as published;
* **normalised** — both the specific and the memories pass through the same lenient
  normaliser (number words to digits, thousands separators and trailing punctuation
  removed, whitespace collapsed) before matching.

Each is also split by session kind, because `batch8-session-kind-attribution-v1`
found that ShareGPT task sessions contribute a third of the specifics and that most of
them are task parameters rather than facts about the user.

What it still cannot see: paraphrase ("my sister's wedding" kept as "a family
wedding"), synonyms, and invented content that shares no literal with the source.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import read_json, write_report  # noqa: E402

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}  # fmt: skip
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}  # fmt: skip
_NUMBER_WORD = re.compile(
    r"\b(?:("
    + "|".join(_TENS)
    + r")(?:[\s-]("
    + "|".join(k for k in _UNITS if k != "zero")
    + r"))?|("
    + "|".join(_UNITS)
    + r"))\b"
)
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_TRAILING = re.compile(r"[\s.,;:!?]+$")
_SPACE = re.compile(r"\s+")


def _number(match: re.Match) -> str:
    tens, unit, alone = match.groups()
    if alone:
        return str(_UNITS[alone])
    return str(_TENS[tens] + (_UNITS[unit] if unit else 0))


def normalise(text: str) -> str:
    """Lenient, symmetric: applied to the specific and to the memories alike."""
    text = text.lower()
    text = _NUMBER_WORD.sub(_number, text)
    text = _THOUSANDS.sub("", text)
    text = _SPACE.sub(" ", text)
    return _TRAILING.sub("", text).strip()


def measure(store_path: Path, manifest_path: Path) -> dict:
    from batch8_session_kind_attribution import session_kind

    from llm_long_term_memory.evaluation.datasets import longmemeval as lme
    from llm_long_term_memory.ingest.fidelity import _extract_facets, user_assertions
    from llm_long_term_memory.ingest.pipeline import namespaced_sessions
    from llm_long_term_memory.store import SQLiteMemoryStore, scoped_session_id

    manifest = read_json(manifest_path)
    instances = {i.question_id: i for i in lme.load(manifest["variant"], REPO / "data")}
    pairs = namespaced_sessions([instances[q] for q in manifest["question_ids"]])
    store = SQLiteMemoryStore(store_path, read_only=True)
    store.initialize()
    try:
        by_scoped: dict[str, list] = defaultdict(list)
        for namespace in {namespace for namespace, _ in pairs}:
            for memory in store.iter_all(namespace):
                if memory.source_session_id:
                    by_scoped[memory.source_session_id].append(memory)
    finally:
        store.close()

    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    recovered_examples: list[dict] = []
    for namespace, session in pairs:
        memories = by_scoped.get(scoped_session_id(namespace, session.session_id), [])
        blob = " || ".join(f"{m.content} {m.object or ''}" for m in memories).lower()
        loose_blob = normalise(blob)
        kind = "task" if session_kind(session.session_id) == "sharegpt" else "personal"
        for facet, values in _extract_facets(user_assertions(session)).items():
            for value in values:
                literal = value in blob
                loose = literal or normalise(value) in loose_blob
                for key in ((kind, facet), (kind, "all"), ("all", facet), ("all", "all")):
                    counts[key]["stated"] += 1
                    counts[key]["literal"] += literal
                    counts[key]["normalised"] += loose
                if loose and not literal and len(recovered_examples) < 12:
                    recovered_examples.append({"facet": facet, "stated": value})

    table = {
        f"{kind}|{facet}": {
            "stated": c["stated"],
            "literal": c["literal"] / c["stated"],
            "normalised": c["normalised"] / c["stated"],
        }
        for (kind, facet), c in sorted(counts.items())
        if c["stated"]
    }
    return {
        "store": str(store_path.relative_to(REPO))
        if store_path.is_relative_to(REPO)
        else str(store_path),
        "manifest": str(manifest_path.relative_to(REPO)),
        "table": table,
        "recovered_examples": recovered_examples,
    }


def render(result: dict) -> str:
    table = result["table"]
    lines = [
        f"# Recall ruler sensitivity on `{result['store']}`",
        "",
        "> Zero calls. The published ruler is unchanged; this measures its literal-match",
        "> bias and the effect of ShareGPT task sessions. It still cannot see paraphrase.",
        "",
        "| sessions | specifics | literal (published) | normalised |",
        "|---|---:|---:|---:|",
    ]
    for kind, label in (
        ("all", "all"),
        ("personal", "personal (non-ShareGPT)"),
        ("task", "ShareGPT task"),
    ):
        row = table.get(f"{kind}|all")
        if row:
            lines.append(
                f"| {label} | {row['stated']:,} | {row['literal']:.1%} | {row['normalised']:.1%} |"
            )
    lines += [
        "",
        "## Personal sessions by facet",
        "",
        "| facet | specifics | literal | normalised |",
        "|---|---:|---:|---:|",
    ]
    for key, row in table.items():
        kind, facet = key.split("|")
        if kind == "personal" and facet != "all":
            lines.append(
                f"| {facet} | {row['stated']:,} | {row['literal']:.1%} | {row['normalised']:.1%} |"
            )
    if result["recovered_examples"]:
        lines += [
            "",
            "Examples counted lost literally and kept after normalisation: "
            + ", ".join(f"`{e['stated']}`" for e in result["recovered_examples"])
            + ".",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("store", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args()
    store = args.store if args.store.is_absolute() else (REPO / args.store).resolve()
    manifest = args.manifest if args.manifest.is_absolute() else (REPO / args.manifest).resolve()
    result = measure(store, manifest)
    print(write_report(result, render, json_out=args.json_out, md_out=args.md_out), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
