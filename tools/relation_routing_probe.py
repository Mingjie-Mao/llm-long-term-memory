"""Can a question be routed to the relation an exhaustive scan would have to scan?

`results/analysis/exhaustive-scan.json` measured the ceiling: scanning one relation
returns every member of a set — 100% complete coverage at 68.5 median tokens, against
top-20 similarity's 58.3% at 439. That is what would lift the `count` result, where 43%
of probes cannot be answered from the retrieved context at all.

But the scan arm was **told which relation to scan by the probe's ground truth**. The
routing that a real system would need does not exist, and without it the ceiling is not
reachable. This measures whether it could be built, using only the local encoder — no
provider call, so the answer is free and available before v4.1 is scoped.

**Two arms, because the obvious one is rigged.**

`templated` routes the probe's own question. That question is generated *from* the
relation name (`visited` becomes "places has the user visited"), so matching it back is
close to matching a string against itself. It is reported as an **upper bound** and
nothing else: if routing fails even here, it cannot work on real questions either.

`natural` routes real LongMemEval questions from `train150`, which nobody wrote with a
relation vocabulary in mind. It has no ground-truth relation label, so it cannot be
scored automatically; what it reports is the *margin* between the best and second-best
relation, plus a sample for reading. A router that is confidently wrong looks exactly
like a router that is confidently right until someone reads it.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

# One sentence per set-arity relation, written to describe what a *question* about it
# would ask, not what the relation is called. Naming them "visited / attended / read"
# would route by string identity and measure nothing.
RELATION_GLOSS = {
    "acquired": "buying, getting or receiving a new possession or item",
    "ate_at": "eating at a restaurant, cafe or food place",
    "attended": "going to a concert, wedding, workshop or other event",
    "completed": "finishing a task, project or course",
    "cooked": "preparing or making a dish or meal at home",
    "financial_activity": "an investment, payment, saving or other money transaction",
    "grows": "keeping or growing a plant or garden",
    "knows_person": "a friend, relative, colleague or other person the user knows",
    "listened_to": "listening to music, an album, a song or a podcast",
    "owns": "something the user already possesses",
    "practises_hobby": "a hobby, sport or leisure activity the user does",
    "read": "reading a book, article or publication",
    "used_service": "using an app, subscription or delivery service",
    "visited": "going to a place, city, country, museum or shop",
    "watched": "watching a film, television series or video",
}

_SCOPE_CLAUSE = re.compile(r"\s*Count only facts recorded under:.*$", re.IGNORECASE | re.DOTALL)
_AGGREGATION = re.compile(r"\b(?:how many|total|count|different|all (?:the|my))\b", re.IGNORECASE)


def _encoder():
    from llm_long_term_memory.embed import Encoder

    return Encoder()


def _unit(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def _route(encoder, questions: list[str], relations: list[str]) -> list[tuple[str, float, float]]:
    """Nearest relation gloss per question, with the margin over the runner-up."""
    gloss = _unit(
        np.asarray(encoder.encode([RELATION_GLOSS[r] for r in relations]), dtype=np.float32)
    )
    asked = _unit(np.asarray(encoder.encode(questions), dtype=np.float32))
    scores = asked @ gloss.T
    out = []
    for row in scores:
        order = np.argsort(-row)
        best, second = int(order[0]), int(order[1])
        out.append((relations[best], float(row[best]), float(row[best] - row[second])))
    return out


def _truth(probe: dict) -> str | None:
    found = re.search(r"relation_type is '([a-z_]+)'", probe["derivation"])
    return found.group(1) if found else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probes", type=Path, default=REPO / "results/analysis/synthesis-probes.json"
    )
    parser.add_argument("--dataset", type=Path, default=REPO / "data/longmemeval_s_cleaned.json")
    parser.add_argument("--manifest", type=Path, default=REPO / "results/manifests/train150.json")
    parser.add_argument("--out", type=Path, default=REPO / "results/analysis/relation-routing.json")
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()

    relations = sorted(RELATION_GLOSS)
    encoder = _encoder()

    # --- upper bound: the probe's own templated question, scope clause removed -------
    spec = json.loads(args.probes.read_text(encoding="utf-8"))
    probes = [p for p in spec["probes"] if p["kind"] == "count" and _truth(p)]
    asked = [_SCOPE_CLAUSE.sub("", p["question"]).strip() for p in probes]
    gold = [_truth(p) for p in probes]
    routed = _route(encoder, asked, relations)

    hits = sum(1 for (pred, _, _), want in zip(routed, gold, strict=True) if pred == want)
    per_relation: dict[str, list[int]] = defaultdict(list)
    confusion: Counter[tuple[str, str]] = Counter()
    for (pred, _, _), want in zip(routed, gold, strict=True):
        per_relation[want].append(int(pred == want))
        if pred != want:
            confusion[(want, pred)] += 1

    print(f"templated arm (UPPER BOUND) — {len(probes)} count probes")
    print(f"  routing accuracy: {hits}/{len(probes)} = {hits / len(probes):.1%}")
    print(f"  {'relation':<20}{'n':>4}{'correct':>9}")
    for relation in sorted(per_relation, key=lambda r: -len(per_relation[r])):
        marks = per_relation[relation]
        print(f"    {relation:<18}{len(marks):>4}{sum(marks):>9}")
    if confusion:
        print("  most common confusions (truth -> predicted):")
        for (want, pred), count in confusion.most_common(5):
            print(f"    {want} -> {pred}  x{count}")

    # --- reality check: real questions, no label, margins only ----------------------
    ids = set(json.loads(args.manifest.read_text(encoding="utf-8"))["question_ids"])
    dataset = [
        q
        for q in json.loads(args.dataset.read_text(encoding="utf-8"))
        if q["question_id"] in ids and _AGGREGATION.search(q["question"])
    ]
    natural = _route(encoder, [q["question"] for q in dataset], relations) if dataset else []
    margins = [margin for _, _, margin in natural]

    print(f"\nnatural arm (no ground truth) — {len(dataset)} real aggregation questions")
    if margins:
        ordered = sorted(margins)
        print(
            f"  margin over runner-up: median {ordered[len(ordered) // 2]:.3f}  "
            f"min {ordered[0]:.3f}  max {ordered[-1]:.3f}"
        )
        print(
            f"  routed with a margin under 0.05: "
            f"{sum(1 for m in margins if m < 0.05)}/{len(margins)}"
        )
        print("  samples for reading (a confident wrong route looks like a right one):")
        for question, (pred, score, margin) in list(zip(dataset, natural, strict=True))[
            : args.samples
        ]:
            print(
                f"    -> {pred:<18} sim {score:.2f} margin {margin:.2f}  "
                f"{question['question'][:78]}"
            )

    payload = {
        "schema_version": 1,
        "relations": relations,
        "templated_arm": {
            "is_an_upper_bound": (
                "The probe question is generated from the relation name, so this is close "
                "to matching a string against itself. Real questions are harder."
            ),
            "probes": len(probes),
            "accuracy": hits / len(probes) if probes else None,
            "per_relation": {r: {"n": len(v), "correct": sum(v)} for r, v in per_relation.items()},
            "confusions": [
                {"truth": w, "predicted": p, "count": c} for (w, p), c in confusion.most_common()
            ],
        },
        "natural_arm": {
            "has_no_ground_truth": (
                "Real LongMemEval questions carry no relation label. Only the margin and a "
                "readable sample are reported; accuracy here would be invented."
            ),
            "questions": len(dataset),
            "median_margin": sorted(margins)[len(margins) // 2] if margins else None,
            "low_margin_share": (
                sum(1 for m in margins if m < 0.05) / len(margins) if margins else None
            ),
            "routes": [
                {"question": q["question"], "predicted": p, "similarity": s, "margin": m}
                for q, (p, s, m) in zip(dataset, natural, strict=True)
            ],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
