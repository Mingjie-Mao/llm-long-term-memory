"""Export one BEAM half from the pinned parquet files to the JSON the loader reads.

    python3 tools/beam_export.py --half dev

Run it with a Python that has pyarrow; the project environment does not install it.

Only the conversations `results/manifests/beam-split.json` assigns to the half are
written, so the final-test conversations stay off disk until the final run exports them,
and exporting that half has to be asked for by name. The output quotes the benchmark, so
it lands under the ignored `data/beam/`. This prints counts, never text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import beam_split  # noqa: E402  (the pinned files, revision and licence)

KEEP = ("id", "role", "content", "time_anchor")
COLUMNS = ["conversation_id", "conversation_seed", "chat", "probing_questions"]


def export(half: str, repo: Path = REPO) -> dict:
    import pyarrow.parquet as pq  # the only code path that needs pyarrow

    record = json.loads((repo / "results/manifests/beam-split.json").read_text(encoding="utf-8"))
    pinned = {scale: digest for scale, (_, digest) in beam_split.FILES.items()}
    recorded = {scale: entry["sha256"] for scale, entry in record["files"].items()}
    if record["revision"] != beam_split.REVISION or recorded != pinned:
        raise ValueError("the split record and the pinned files disagree")
    wanted = set(record[half]["conversations"])
    conversations = []
    for scale, (relative, expected) in beam_split.FILES.items():
        path = repo / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"{relative} is not the pinned revision")
        for row in pq.read_table(path, columns=COLUMNS).to_pylist():
            if f"{scale}-{row['conversation_id']}" not in wanted:
                continue
            conversations.append(
                {
                    "scale": scale,
                    "conversation_id": row["conversation_id"],
                    "seed_id": row["conversation_seed"]["id"],
                    "chat": [
                        [{key: message.get(key) for key in KEEP} for message in session]
                        for session in row["chat"]
                    ],
                    "probing_questions": beam_split.parse_probing(row["probing_questions"]),
                }
            )
    found = {f"{c['scale']}-{c['conversation_id']}" for c in conversations}
    if found != wanted:
        raise ValueError(f"{len(wanted - found)} conversations of the {half} half are missing")
    conversations.sort(key=lambda c: (c["scale"], len(c["conversation_id"]), c["conversation_id"]))
    return {
        "name": "beam-export",
        "half": half,
        "source": beam_split.SOURCE,
        "revision": beam_split.REVISION,
        "licence": beam_split.LICENCE,
        "conversations": conversations,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--half", choices=("dev", "test"), required=True)
    parser.add_argument(
        "--final-run",
        action="store_true",
        help="required to export the test half, which is answered once, at the end",
    )
    args = parser.parse_args()
    if args.half == "test" and not args.final_run:
        print("STOP: the test half is exported only for the final run; pass --final-run then")
        return 2
    try:
        payload = export(args.half)
    except (OSError, ValueError, KeyError) as exc:
        print(f"STOP: {exc}")
        return 2
    out = REPO / "data" / "beam" / f"beam-{args.half}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    questions = sum(
        len(items)
        for conversation in payload["conversations"]
        for items in conversation["probing_questions"].values()
    )
    sessions = sum(len(conversation["chat"]) for conversation in payload["conversations"])
    print(
        json.dumps(
            {
                "half": args.half,
                "conversations": len(payload["conversations"]),
                "sessions": sessions,
                "questions": questions,
                "written": str(out.relative_to(REPO)),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
