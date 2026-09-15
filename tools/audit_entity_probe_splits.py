"""Which of the 60 entity-count probes have a gold answer the splitter invented.

`entity_count_probes.py` fixed the old generator's "one row is not one member" defect by
splitting a stored object on commas and `and`. That is right for a list — "tofu, tempeh,
seitan" really is three foods — and wrong for a name or a trailing date, and the two are
indistinguishable to a regex. The generator's own comment names `Pride and Prejudice` as
the risk. In this store that title never appears; what appears instead is a date or a city
pasted onto the end of an object, which the splitter turns into a member of the set:

    'MoMA, Dec 2023'                        -> ['MoMA', 'Dec 2023']
    'TechFest, San Francisco, February 2023' -> ['TechFest', 'San Francisco', 'February 2023']

Both probes then ask "how many distinct places has the user visited?" and count a month as
a place. So the risk is not hypothetical and not mainly about titles: it is about objects
that carry their own provenance.

This reads the emitted probe set and the store it came from and reports, per probe, every
member the splitter produced that does not look like a member of the asked-about set, and
every unsplit object that is a date or carries a record's date, time or price. It decides
nothing. It writes no correction, does not touch `entity-count-probes.json`, and
makes no provider call — the categories below are regex guesses meant to put a short,
checkable list in front of a human, exactly as `count-gold-review` did for the paid set.

Run: `.venv/bin/python tools/audit_entity_probe_splits.py`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from entity_count_probes import entities_of

REPO = Path(__file__).resolve().parent.parent
PROBES = REPO / "results/analysis/entity-count-probes.json"
STORES = REPO / "stores"
OUT = REPO / "results/analysis/entity-probe-split-audit.json"

# Whole month names. Matching any word that began with a month's three letters made
# 'decorating' and 'Marketing' dates.
_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b\.?"
)
# A member that is a date, not a thing. Written as a whole-string match: "Dec 2023" is a
# date, "Dec 2023 conference" is a name that happens to start with one.
_DATE = re.compile(
    rf"^(?:{_MONTH}\s*\d{{0,2}},?\s*(?:19|20)\d{{2}}|(?:19|20)\d{{2}}|(?:19|20)\d0s|\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}(?:\s*(?:19|20)\d{{2}})?|{_MONTH})$",
    re.IGNORECASE,
)
# A member that is the tail of a sentence rather than a noun: the separator cut a clause.
_CLAUSE = re.compile(
    r"^(attended|held|hosted|visited|located|organi[sz]ed|purchased|bought|read|on|in|at|"
    r"for|since|from|during|with|to|by)\b",
    re.IGNORECASE,
)
# A member that is a real thing carrying a date the split cut in half: "Walk for Hunger on
# May 21" lost its year to the comma. It is a member, so dropping it would undercount; it
# is also not the name of the thing, so it needs repair, not deletion. Kept separate from
# the drop categories for exactly that reason.
_TRAILING_DATE = re.compile(rf"\bon\s+{_MONTH}\s*\d{{1,2}}(st|nd|rd|th)?$", re.IGNORECASE)
# An unsplit object that records an occasion rather than naming a member: a day or a year
# after a month, an ISO date, a clock time or a price. The first version of this audit read
# only members a separator produced, and so reported zero flagged members while the set it
# read counted 'January 15 2023' as a place visited and one pair of sneakers twice through
# two receipts. A leading model year ('1995 Honda Civic') is not flagged.
_RECORD = re.compile(
    rf"\b{_MONTH}\s*\d{{1,2}}(?:st|nd|rd|th)?\b|\b{_MONTH},?\s*(?:19|20)\d{{2}}\b|"
    r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b|\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*(?:am|pm)\b|[$€£¥]\s?\d",
    re.IGNORECASE,
)


def _shown(path: Path) -> str:
    """A repository-relative path where there is one, otherwise the path as given."""
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


# Categories where the member is not a member of the asked-about set at all.
_DROP = {"date", "clause_fragment"}


def classify(member: str) -> str | None:
    """What is wrong with this member, or None if nothing visibly is."""
    text = member.strip()
    if _DATE.match(text):
        return "date"
    if _CLAUSE.match(text):
        return "clause_fragment"
    if _TRAILING_DATE.search(text):
        return "truncated_date"
    return None


def audit(probes_file: Path, stores_dir: Path) -> dict[str, Any]:
    payload = json.loads(probes_file.read_text(encoding="utf-8"))
    store = stores_dir / payload["store"]
    connection = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row

    findings: list[dict[str, Any]] = []
    split_probes = 0
    for probe in payload["probes"]:
        splits: list[dict[str, Any]] = []
        suspect: list[dict[str, str]] = []
        for memory_id in probe["evidence_memory_ids"]:
            row = connection.execute(
                "SELECT object FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                continue
            members = entities_of(row["object"])
            if len(members) < 2:
                whole = (row["object"] or "").strip()
                reason = None
                if _DATE.match(whole):
                    reason = "date"
                elif _RECORD.search(whole):
                    reason = "record"
                if reason:
                    suspect.append(
                        {
                            "member": whole,
                            "reason": reason,
                            "from_object": row["object"],
                            "memory_id": memory_id,
                            "unsplit": True,
                        }
                    )
                continue
            splits.append({"memory_id": memory_id, "object": row["object"], "members": members})
            for member in members:
                reason = classify(member)
                if reason:
                    suspect.append(
                        {
                            "member": member,
                            "reason": reason,
                            "from_object": row["object"],
                            "memory_id": memory_id,
                        }
                    )
        if splits:
            split_probes += 1
        elif not suspect:
            continue
        # An object that ends in a date is a "name, where, when" record, so its other
        # members are suspect for a reason no regex here can state: 'San Francisco' is a
        # plausible place and an implausible conference. Listed, not flagged.
        dated_objects = {s["from_object"] for s in suspect if s["reason"] == "date"}
        co_members = [
            member
            for split in splits
            if split["object"] in dated_objects
            for member in split["members"]
            if classify(member) is None
        ]
        findings.append(
            {
                "probe_id": probe["probe_id"],
                "question": probe["question"],
                "gold_answer": probe["answer"],
                "gold_entities": probe["entities"],
                # What the count would be if every non-member were dropped. Shown so a
                # reviewer can see the size of the disagreement, NOT as a corrected gold:
                # dropping a member is itself a judgement this tool is not making, and a
                # `truncated_date` member stays counted because it is a real thing whose
                # name is merely cut short.
                "answer_if_non_members_dropped": len(
                    [e for e in probe["entities"] if classify(e) not in _DROP]
                ),
                "suspect_members": suspect,
                "co_members_from_a_dated_object": sorted(set(co_members)),
                "splits_in_evidence": splits,
            }
        )
    findings.sort(key=lambda f: (not f["suspect_members"], f["probe_id"]))

    connection.close()
    return {
        "schema_version": 2,
        "provider_calls": 0,
        "review_status": "machine_flagged_pending_human_review",
        "decides_nothing": (
            "Every entry is a regex guess about one member. No gold answer is changed, no "
            "probe file is rewritten, and 'answer_if_non_members_dropped' is an "
            "arithmetic consequence of the flags, not a corrected answer."
        ),
        "scope": (
            "Members the splitter produced from a multi-entity object, and unsplit objects "
            "that are a date or carry a record's date, time or price. A gold answer can "
            "still be wrong for reasons this cannot see: a name split into two real-looking "
            "halves ('Daisy Jones and The Six'), an alias counted twice, a member that is "
            "not the asked-about kind of thing, or a member the extractor lost before the "
            "probe was built."
        ),
        "inputs": [
            {
                "path": _shown(probes_file),
                "sha256": hashlib.sha256(probes_file.read_bytes()).hexdigest(),
            },
            {
                "path": _shown(Path(__file__)),
                "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
        ],
        "store": payload["store"],
        "store_fingerprint_claimed": payload["store_fingerprint"],
        "probes": len(payload["probes"]),
        "probes_whose_gold_depends_on_a_split": split_probes,
        "probes_with_a_flagged_member": sum(1 for f in findings if f["suspect_members"]),
        "probes_with_a_flagged_unsplit_object": sum(
            1 for f in findings if any(s.get("unsplit") for s in f["suspect_members"])
        ),
        "flagged_members": sum(len(f["suspect_members"]) for f in findings),
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes", type=Path, default=PROBES)
    parser.add_argument("--stores", type=Path, default=STORES)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    report = audit(args.probes, args.stores)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        f"{report['probes_whose_gold_depends_on_a_split']}/{report['probes']} probes have a "
        f"gold answer that depends on splitting an object"
    )
    print(
        f"{report['probes_with_a_flagged_member']} of them contain "
        f"{report['flagged_members']} member(s) a regex can already tell is not a member:"
    )
    for finding in report["findings"]:
        if not finding["suspect_members"]:
            continue
        members = ", ".join(f"{s['member']!r} ({s['reason']})" for s in finding["suspect_members"])
        print(
            f"  {finding['probe_id']}  gold={finding['gold_answer']} "
            f"-> {finding['answer_if_non_members_dropped']}   {members}"
        )
        if finding["co_members_from_a_dated_object"]:
            print(f"      also from a dated object: {finding['co_members_from_a_dated_object']}")
    print(
        f"\nAll {report['probes_whose_gold_depends_on_a_split']} split-dependent probes are in "
        f"{_shown(args.out)} for review · no gold answer was changed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
