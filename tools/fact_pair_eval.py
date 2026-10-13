"""Stage D of research/ROADMAP.md, measured on research/fact-pairs-v1.json.

    python tools/fact_pair_eval.py candidates   # zero calls: candidate recall@K
    python tools/fact_pair_eval.py judge        # model calls: relation decisions

`candidates` asks, for each annotated pair, whether the old fact is among the new
fact's top-K candidates (`temporal.transition.candidate_predecessors`, vectors already in
the store's index), under several pool restrictions, against today's rule (only an
identical `(subject, predicate)` key is ever compared). `judge` lives below.

Development evidence on exposed questions; the annotation is the assistant's draft.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from llm_long_term_memory.store import NumpyFlatIndex, SQLiteMemoryStore  # noqa: E402
from llm_long_term_memory.temporal.transition import candidate_predecessors  # noqa: E402

PAIRS = REPO / "research/fact-pairs-v1.json"
OUT = REPO / "results/analysis/fact-pairs-v1"
KS = (1, 3, 5, 10)
VARIANTS = {
    "all earlier memories": {},
    "same subject": {"same_subject": True},
    "user-stated only": {"roles": ("user",)},
}


def _load(split: str):
    store = SQLiteMemoryStore(REPO / "stores" / f"{split}-anchor-v2.db", read_only=True)
    index = NumpyFlatIndex(REPO / "stores" / f"{split}-anchor-v2-index", dim=384)
    return store, index


def pair_contexts():
    """(question id, pair, new memory, the user's memories, vector lookup) per pair."""
    data = json.loads(PAIRS.read_text(encoding="utf-8"))["questions"]
    opened: dict[str, tuple] = {}
    try:
        for qid, entry in data.items():
            if not entry["pairs"]:
                continue
            split = entry["split"]
            if split not in opened:
                opened[split] = _load(split)
            store, index = opened[split]
            memories = store.iter_all(qid)
            by_id = {m.id: m for m in memories}
            vectors = index.vectors_for(by_id)
            for pair in entry["pairs"]:
                yield qid, pair, by_id[pair["new"]], memories, vectors.get
    finally:
        for store, _ in opened.values():
            store.close()


def candidates() -> dict:
    result = {"variants": {}, "pairs": []}
    for name in VARIANTS:
        result["variants"][name] = {
            "replace": {f"recall@{k}": 0 for k in KS},
            "coexist_in_top5": 0,
        }
    counts = {"replace": 0, "coexist": 0, "replace_same_key": 0}
    for qid, pair, new, memories, vector_of in pair_contexts():
        counts[pair["relation"]] += 1
        old = next(m for m in memories if m.id == pair["old"])
        same_key = (old.subject, old.predicate) == (new.subject, new.predicate)
        if pair["relation"] == "replace":
            counts["replace_same_key"] += same_key
        ranks = {}
        for name, options in VARIANTS.items():
            ranked = candidate_predecessors(new, memories, vector_of, k=len(memories), **options)
            ids = [c.memory.id for c in ranked]
            rank = ids.index(pair["old"]) + 1 if pair["old"] in ids else None
            ranks[name] = rank
            v = result["variants"][name]
            if pair["relation"] == "replace":
                for k in KS:
                    v["replace"][f"recall@{k}"] += rank is not None and rank <= k
            elif rank is not None and rank <= 5:
                v["coexist_in_top5"] += 1
        result["pairs"].append(
            {
                "question_id": qid,
                "relation": pair["relation"],
                "same_key": same_key,
                "pool": sum(1 for m in memories if m.id != new.id),
                "rank": ranks,
            }
        )
    result["counts"] = counts
    return result


K_JUDGE = 5
SAMPLES = 3


def _client(sample: str):
    from llm_long_term_memory.config import ExperimentConfig, Settings
    from llm_long_term_memory.llm import Limits, QuotaManager, UsageTracker
    from llm_long_term_memory.llm.cache import CachingClient
    from llm_long_term_memory.llm.client import GeminiClient

    settings = Settings()
    cfg = ExperimentConfig.from_yaml(REPO / "configs/fallback.yaml")
    quota = QuotaManager(
        state_dir=settings.store_dir / "quota",
        default=Limits(rpm=cfg.quota.rpm, tpm=cfg.quota.tpm, rpd=cfg.quota.rpd),
    )
    client = GeminiClient(settings.require_api_key(), quota=quota, usage=UsageTracker())
    cached = CachingClient(
        client, REPO / "stores/cache/fact-pairs-v1-judge.db", sample=sample, scope={"k": K_JUDGE}
    )
    return cached, cfg.models.extractor


def judge() -> dict:
    """Relation decisions for every annotated new fact against its top-5 candidates."""
    from llm_long_term_memory.temporal.transition import TRANSITION_PROMPT_VERSION, decide

    contexts: dict[str, tuple] = {}
    labels: dict[tuple[str, str], str] = {}
    for qid, pair, new, memories, vector_of in pair_contexts():
        labels[(new.id, pair["old"])] = pair["relation"]
        if new.id not in contexts:
            top = candidate_predecessors(new, memories, vector_of, k=K_JUDGE)
            contexts[new.id] = (qid, new, top)
    samples = []
    for n in range(1, SAMPLES + 1):
        client, model = _client(f"rep{n}")
        decisions = {}
        for new_id, (_, new, top) in contexts.items():
            decisions[new_id] = decide(client, model, new, top)
        samples.append(decisions)
    result = {
        "prompt_version": TRANSITION_PROMPT_VERSION,
        "model": model,
        "k": K_JUDGE,
        "samples": SAMPLES,
        "pairs": [],
        "unlabelled_replacements": [],
    }
    for (new_id, old_id), relation in labels.items():
        qid, new, top = contexts[new_id]
        result["pairs"].append(
            {
                "question_id": qid,
                "label": relation,
                "in_candidates": old_id in {c.memory.id for c in top},
                "decisions": [s[new_id].get(old_id, "not_a_candidate") for s in samples],
            }
        )
    labelled = set(labels)
    for new_id, (qid, new, top) in contexts.items():
        for c in top:
            if (new_id, c.memory.id) in labelled:
                continue
            votes = [s[new_id].get(c.memory.id) for s in samples]
            if any(v in {"replace", "terminate"} for v in votes):
                result["unlabelled_replacements"].append(
                    {
                        "question_id": qid,
                        "new": new.content,
                        "earlier": c.memory.content,
                        "decisions": votes,
                    }
                )
    return result


def render_judge(result: dict) -> str:
    pairs = result["pairs"]
    replace = [p for p in pairs if p["label"] == "replace"]
    coexist = [p for p in pairs if p["label"] == "coexist"]
    closes = {"replace", "terminate"}
    n = result["samples"]
    found = sum(sum(d == "replace" for d in p["decisions"]) for p in replace)
    false = sum(sum(d in closes for d in p["decisions"]) for p in coexist)
    labelled_closes = sum(sum(d in closes for d in p["decisions"]) for p in pairs)
    correct_closes = sum(sum(d == "replace" for d in p["decisions"]) for p in replace)
    unlabelled = sum(
        sum(d in closes for d in u["decisions"]) for u in result["unlabelled_replacements"]
    )
    lines = [
        "# Fact pairs v1: relation decisions (stage D)",
        "",
        f"> Generated by `tools/fact_pair_eval.py judge`. Model `{result['model']}`, prompt",
        f"> `{result['prompt_version']}`, top-{result['k']} candidates, {n} samples per new fact.",
        "> Development evidence; the annotation is the assistant's draft.",
        "",
        "| metric | value |",
        "|---|---|",
        f"| update recall (replace pairs judged replace) | {found}/{n * len(replace)} |",
        "| false supersession (coexist pairs judged replace/terminate) |"
        f" {false}/{n * len(coexist)} |",
        f"| update precision on labelled pairs | {correct_closes}/{labelled_closes} |",
        f"| replace/terminate on unlabelled candidates (to review) | {unlabelled} |",
        "",
        "| question | label | decisions |",
        "|---|---|---|",
    ]
    for p in pairs:
        lines.append(f"| `{p['question_id']}` | {p['label']} | {', '.join(p['decisions'])} |")
    if result["unlabelled_replacements"]:
        lines += ["", "## Closes proposed on unlabelled candidates", ""]
        for u in result["unlabelled_replacements"]:
            lines.append(
                f"- `{u['question_id']}` {u['decisions']}: NEW {u['new'][:90]} / EARLIER"
                f" {u['earlier'][:90]}"
            )
    return "\n".join(lines) + "\n"


def _copy_store(split: str, target: Path) -> SQLiteMemoryStore:
    import sqlite3

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    source = REPO / "stores" / f"{split}-anchor-v2.db"
    with (
        sqlite3.connect(f"file:{source}?mode=ro", uri=True) as src,
        sqlite3.connect(target) as dst,
    ):
        src.backup(dst)
    return SQLiteMemoryStore(target)


def _state(store, memory_id: str) -> dict:
    m = store.get(memory_id)
    return {
        "status": m.status,
        "superseded_by": m.superseded_by,
        "valid_to": m.valid_to.isoformat() if m.valid_to else None,
        "key": [m.subject, m.predicate],
    }


def _timeline_ok(store, pair: dict) -> bool:
    from llm_long_term_memory.temporal import as_of

    old, new = store.get(pair["old"]), store.get(pair["new"])
    if pair["relation"] == "coexist":
        return old.status == "active" and new.status == "active"
    if not (new.status == "active" and old.status == "superseded" and old.valid_to):
        return False
    between = old.occurred_at + (new.occurred_at - old.occurred_at) / 2
    return [m.id for m in as_of([old, new], between)] == [old.id] and [
        m.id for m in as_of([old, new], new.occurred_at)
    ] == [new.id]


def apply(work: Path) -> dict:
    """Judge each arriving fact, apply confirmed replacements, check the timelines.

    `forward`: the later fact of each pair arrives with the earlier one in the store;
    `reverse`: the earlier fact arrives last. `before`: the copy as the extractor and
    exact-key resolver left it. All on copies under `work`; the stores are untouched.
    """
    from collections import Counter

    from llm_long_term_memory.temporal import TemporalResolver
    from llm_long_term_memory.temporal.transition import apply_update, decide

    data = json.loads(PAIRS.read_text(encoding="utf-8"))["questions"]
    clients = [_client(f"rep{n}") for n in range(1, SAMPLES + 1)]
    model = clients[0][1]
    outcome: dict = {"orders": {}, "applied": {}}
    states: dict[str, dict] = {}
    for order in ("before", "forward", "reverse"):
        ok = Counter()
        applied = []
        states[order] = {}
        for split in sorted({e["split"] for e in data.values()}):
            store = _copy_store(split, work / f"{split}-{order}.db")
            index = NumpyFlatIndex(REPO / "stores" / f"{split}-anchor-v2-index", dim=384)
            resolver = TemporalResolver(store)
            try:
                for qid, entry in data.items():
                    if entry["split"] != split or not entry["pairs"]:
                        continue
                    if order != "before":
                        arrivals = {
                            (p["new"] if order == "forward" else p["old"]) for p in entry["pairs"]
                        }
                        for arriving_id in sorted(arrivals):
                            memories = store.iter_all(qid)
                            vectors = index.vectors_for({m.id for m in memories})
                            arriving = store.get(arriving_id)
                            top = candidate_predecessors(
                                arriving, memories, vectors.get, k=K_JUDGE, earlier_only=False
                            )
                            votes = [decide(c, model, arriving, top) for c, _ in clients]
                            for cand in top:
                                tally = Counter(v.get(cand.memory.id) for v in votes)
                                relation, n = tally.most_common(1)[0]
                                if relation == "replace" and n * 2 > SAMPLES:
                                    apply_update(
                                        store, resolver, arriving, store.get(cand.memory.id)
                                    )
                                    applied.append([qid, arriving_id, cand.memory.id])
                    for pair in entry["pairs"]:
                        ok[(pair["relation"], _timeline_ok(store, pair))] += 1
                        for mid in (pair["old"], pair["new"]):
                            states[order][mid] = _state(store, mid)
            finally:
                store.close()
        outcome["orders"][order] = {
            "replace_correct": ok[("replace", True)],
            "replace_total": ok[("replace", True)] + ok[("replace", False)],
            "coexist_correct": ok[("coexist", True)],
            "coexist_total": ok[("coexist", True)] + ok[("coexist", False)],
        }
        outcome["applied"][order] = applied
    differ = [mid for mid in states["forward"] if states["forward"][mid] != states["reverse"][mid]]
    outcome["order_invariant"] = {"memories": len(states["forward"]), "differ": differ}
    return outcome


def pairwise(work: Path) -> dict:
    """v3: gated candidates (similarity >= 0.65, user-stated), one judgment per pair shown
    earlier-then-later, majority of 3 samples, on copies in both arrival orders."""
    from collections import Counter

    from llm_long_term_memory.temporal import TemporalResolver
    from llm_long_term_memory.temporal.transition import (
        PAIR_PROMPT_VERSION,
        apply_update,
        judge_pair,
        related_candidates,
    )

    data = json.loads(PAIRS.read_text(encoding="utf-8"))["questions"]
    clients = [_client(f"rep{n}") for n in range(1, SAMPLES + 1)]
    model = clients[0][1]
    outcome: dict = {"prompt_version": PAIR_PROMPT_VERSION, "orders": {}, "applied": {}}
    labelled = {
        frozenset((p["old"], p["new"])): p["relation"] for e in data.values() for p in e["pairs"]
    }
    pair_votes: dict[frozenset, list] = {}
    states: dict[str, dict] = {}
    for order in ("before", "forward", "reverse"):
        ok = Counter()
        applied = []
        states[order] = {}
        for split in sorted({e["split"] for e in data.values()}):
            store = _copy_store(split, work / f"{split}-v3-{order}.db")
            index = NumpyFlatIndex(REPO / "stores" / f"{split}-anchor-v2-index", dim=384)
            resolver = TemporalResolver(store)
            try:
                for qid, entry in data.items():
                    if entry["split"] != split or not entry["pairs"]:
                        continue
                    if order != "before":
                        arrivals = {
                            (p["new"] if order == "forward" else p["old"]) for p in entry["pairs"]
                        }
                        for arriving_id in sorted(arrivals):
                            memories = store.iter_all(qid)
                            vectors = index.vectors_for({m.id for m in memories})
                            arriving = store.get(arriving_id)
                            for cand in related_candidates(arriving, memories, vectors.get):
                                other = store.get(cand.memory.id)
                                votes = [judge_pair(c, model, arriving, other) for c, _ in clients]
                                key = frozenset((arriving.id, other.id))
                                pair_votes[key] = votes
                                relation, n = Counter(votes).most_common(1)[0]
                                if relation == "replace" and n * 2 > SAMPLES:
                                    apply_update(store, resolver, arriving, other)
                                    applied.append([qid, arriving.id, other.id])
                    for pair in entry["pairs"]:
                        ok[(pair["relation"], _timeline_ok(store, pair))] += 1
                        for mid in (pair["old"], pair["new"]):
                            states[order][mid] = _state(store, mid)
            finally:
                store.close()
        outcome["orders"][order] = {
            "replace_correct": ok[("replace", True)],
            "replace_total": ok[("replace", True)] + ok[("replace", False)],
            "coexist_correct": ok[("coexist", True)],
            "coexist_total": ok[("coexist", True)] + ok[("coexist", False)],
        }
        outcome["applied"][order] = applied
    differ = [mid for mid in states["forward"] if states["forward"][mid] != states["reverse"][mid]]
    outcome["order_invariant"] = {"memories": len(states["forward"]), "differ": differ}
    judged = {k: v for k, v in pair_votes.items() if k in labelled}
    outcome["labelled"] = {
        "replace_judged": sum(1 for k in judged if labelled[k] == "replace"),
        "replace_votes": sum(
            v.count("replace") for k, v in judged.items() if labelled[k] == "replace"
        ),
        "coexist_judged": sum(1 for k in judged if labelled[k] == "coexist"),
        "coexist_false_votes": sum(
            sum(x in {"replace", "terminate"} for x in v)
            for k, v in judged.items()
            if labelled[k] == "coexist"
        ),
    }
    outcome["unlabelled_closes"] = [
        sorted(k)
        for k, v in pair_votes.items()
        if k not in labelled and Counter(v).most_common(1)[0][0] in {"replace", "terminate"}
    ]
    outcome["pairs_judged"] = len(pair_votes)
    return outcome


def render_apply(o: dict) -> str:
    lines = [
        "# Fact pairs v1: timelines after applying confirmed updates (stage D)",
        "",
        "> Generated by `tools/fact_pair_eval.py apply`; copies of the anchor-v2 stores,",
        "> judge `transition-judge-v2`, majority of 3 samples, top-5 candidates of any date.",
        "> Development evidence; the annotation is the assistant's draft.",
        "",
        "| state | replace pairs with a correct timeline | coexist pairs left intact |",
        "|---|---:|---:|",
    ]
    for order, r in o["orders"].items():
        lines.append(
            f"| {order} | {r['replace_correct']}/{r['replace_total']} |"
            f" {r['coexist_correct']}/{r['coexist_total']} |"
        )
    inv = o["order_invariant"]
    lines += [
        "",
        f"Order invariance: {inv['memories'] - len(inv['differ'])}/{inv['memories']} annotated"
        " memories end in the same state (status, successor, valid_to, key) whether the"
        " later or the earlier fact arrives last.",
        "",
        f"Updates applied: forward {len(o['applied']['forward'])},"
        f" reverse {len(o['applied']['reverse'])}.",
    ]
    if inv["differ"]:
        lines += ["", "Differing memories: " + ", ".join(f"`{m}`" for m in inv["differ"])]
    return "\n".join(lines) + "\n"


def render(result: dict) -> str:
    c = result["counts"]
    lines = [
        "# Fact pairs v1: does the old fact reach the new fact's candidates? (stage D)",
        "",
        "> Generated by `tools/fact_pair_eval.py candidates`; zero model calls. Pairs from",
        "> `research/fact-pairs-v1.json` (assistant's draft); exposed development questions.",
        "",
        f"Replace pairs: {c['replace']} (same key today: {c['replace_same_key']}, so the exact-key"
        f" rule can reach at most {c['replace_same_key']}/{c['replace']}). Coexist pairs:"
        f" {c['coexist']}.",
        "",
        "| candidate pool | "
        + " | ".join(f"recall@{k}" for k in KS)
        + " | coexist pairs in top 5 |",
        "|---|" + "---:|" * (len(KS) + 1),
    ]
    for name, v in result["variants"].items():
        cells = " | ".join(f"{v['replace'][f'recall@{k}']}/{c['replace']}" for k in KS)
        lines.append(f"| {name} | {cells} | {v['coexist_in_top5']}/{c['coexist']} |")
    pools = sorted(p["pool"] for p in result["pairs"])
    lines += ["", f"Pool size per new fact: median {pools[len(pools) // 2]}, max {pools[-1]}."]
    misses = [
        p
        for p in result["pairs"]
        if p["relation"] == "replace" and (p["rank"]["all earlier memories"] or 99) > 5
    ]
    if misses:
        lines += ["", "Replace pairs outside the top 5 of the full pool:", ""]
        for p in misses:
            lines.append(f"- `{p['question_id']}`: rank {p['rank']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> None:
    if argv[:1] == ["candidates"]:
        result = candidates()
        OUT.with_suffix(".candidates.json").write_text(
            json.dumps(result, indent=1) + "\n", encoding="utf-8"
        )
        OUT.with_suffix(".candidates.md").write_text(render(result), encoding="utf-8")
        print(render(result))
    elif argv[:1] == ["apply"]:
        work = Path(argv[1]) if len(argv) > 1 else REPO / "stores" / "scratch-fact-pairs"
        outcome = apply(work)
        OUT.with_suffix(".apply.json").write_text(
            json.dumps(outcome, indent=1) + "\n", encoding="utf-8"
        )
        OUT.with_suffix(".apply.md").write_text(render_apply(outcome), encoding="utf-8")
        print(render_apply(outcome))
    elif argv[:1] == ["pairwise"]:
        work = Path(argv[1]) if len(argv) > 1 else REPO / "stores" / "scratch-fact-pairs"
        outcome = pairwise(work)
        lab = outcome["labelled"]
        text = (
            render_apply(outcome)
            .replace(
                "judge `transition-judge-v2`, majority of 3 samples, top-5 candidates of any date.",
                "pairwise judge `transition-pair-v3`, majority of 3 samples, top-5 user-stated"
                " candidates of any date with similarity >= 0.65.",
            )
            .replace("timelines after applying confirmed updates", "pairwise judge (v3)")
            .replace("`tools/fact_pair_eval.py apply`", "`tools/fact_pair_eval.py pairwise`")
        )
        text += (
            f"\nPairs judged: {outcome['pairs_judged']}. Labelled replace pairs judged"
            f" {lab['replace_judged']}/21 (replace votes {lab['replace_votes']});"
            f" labelled coexist pairs judged {lab['coexist_judged']}/6 (close votes"
            f" {lab['coexist_false_votes']}). Closes on unlabelled pairs:"
            f" {len(outcome['unlabelled_closes'])}.\n"
        )
        OUT.with_suffix(".pairwise.json").write_text(
            json.dumps(outcome, indent=1) + "\n", encoding="utf-8"
        )
        OUT.with_suffix(".pairwise.md").write_text(text, encoding="utf-8")
        print(text)
    elif argv[:1] == ["judge"]:
        result = judge()
        OUT.with_suffix(".judge.json").write_text(
            json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        OUT.with_suffix(".judge.md").write_text(render_judge(result), encoding="utf-8")
        print(render_judge(result))
    else:
        raise SystemExit("usage: fact_pair_eval.py candidates | judge | apply | pairwise [workdir]")


if __name__ == "__main__":
    main(sys.argv[1:])
