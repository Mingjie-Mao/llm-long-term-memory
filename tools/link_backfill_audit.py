"""Zero-call audit of completed d1 users; never changes a benchmark store.

    python tools/link_backfill_audit.py dev100 --out results/analysis/d1-dev100-audit

Exports actual changed closures with source turns, checks existing draft gold pairs,
and replays the unchanged linker in both traversal orders using ONLY cached replies.
Missing cache entries make a replay incomplete, never trigger a provider request.
Traversal reversal checks batch-link order; it is not an ingestion-order experiment.
"""

# Chinese report punctuation is intentional.
# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from llm_long_term_memory.config import ExperimentConfig  # noqa: E402
from llm_long_term_memory.llm.cache import request_key  # noqa: E402
from llm_long_term_memory.llm.client import Completion  # noqa: E402
from llm_long_term_memory.store import NumpyFlatIndex, SQLiteMemoryStore  # noqa: E402
from llm_long_term_memory.temporal import TemporalResolver, as_of  # noqa: E402
from llm_long_term_memory.temporal.transition import UpdateLinker  # noqa: E402

FIELDS = (
    "status",
    "superseded_by",
    "valid_from",
    "valid_to",
    "subject",
    "predicate",
    "update_op",
    "target_object",
    "replaces_previous",
)


def state(memory):
    return {key: getattr(memory, key) for key in FIELDS}


def backup(source: Path, target: Path):
    with (
        sqlite3.connect(f"file:{source}?mode=ro", uri=True) as src,
        sqlite3.connect(target) as dst,
    ):
        src.backup(dst)


class CacheMiss(RuntimeError):
    pass


class CacheOnly:
    def __init__(self, split):
        self.split = split
        self.db = sqlite3.connect(
            f"file:{REPO / 'stores/cache/link-backfill-d1.db'}?mode=ro", uri=True
        )
        self.hits = 0
        self.question_id = None
        self.responses = []

    def generate(self, **request):
        key = request_key(sample="rep1", scope={"split": self.split}, **request)
        row = self.db.execute("select completion from responses where key=?", (key,)).fetchone()
        if row is None:
            raise CacheMiss(key)
        self.hits += 1
        stored = json.loads(row[0])
        self.responses.append(
            {
                "question_id": self.question_id,
                "request_key": key,
                "prompt": request["prompt"],
                "completion": stored,
            }
        )
        return Completion(**{**stored, "attempts": 0, "api_latency_ms": 0.0})


def source(memory, anchors):
    anchored = anchors.get(memory.id) or memory
    if not anchored.source_session_id or anchored.source_turn_index is None:
        return []
    return [
        {"turn_index": t.turn_index, "role": t.role, "content": t.content}
        for t in anchors.turns_for_session(anchored.source_session_id)
        if abs(t.turn_index - anchored.source_turn_index) <= 1
    ]


def fact(memory, anchors):
    return {
        "id": memory.id,
        "content": memory.content,
        "scope": memory.scope,
        "event_time": memory.event_time,
        "observed_at": memory.observed_at,
        "event_time_expression": memory.event_time_expression,
        "state": state(memory),
        "source_turns_anchor_v2": source(memory, anchors),
    }


def rebuild_check(snapshot, qids, folder):
    path = folder / "rebuild.db"
    backup(snapshot, path)
    store = SQLiteMemoryStore(path)
    resolver = TemporalResolver(store)
    differences = {"first_pass": [], "second_pass": []}
    try:
        for qid in qids:
            before = {m.id: state(m) for m in store.iter_all(qid)}
            for name in differences:
                resolver.resolve_all(qid)
                after = {m.id: state(m) for m in store.iter_all(qid)}
                for mid in before:
                    if before[mid] != after[mid]:
                        differences[name].append(
                            {
                                "question_id": qid,
                                "memory": mid,
                                "before": before[mid],
                                "after": after[mid],
                            }
                        )
                before = after
    finally:
        store.close()
    return differences


def replay(split, qids, index, folder):
    model = ExperimentConfig.from_yaml(REPO / "configs/fallback.yaml").models.extractor
    results = {}
    snapshots = {}
    for direction in ("forward", "reverse"):
        path = folder / f"{direction}.db"
        backup(REPO / f"stores/{split}.db", path)
        store = SQLiteMemoryStore(path)
        cache = CacheOnly(split)
        resolver = TemporalResolver(store)
        linker = UpdateLinker(cache, model)
        completed, missing = [], []
        snapshots[direction] = {}
        try:
            for qid in qids:
                cache.question_id = qid
                memories = sorted(
                    store.iter_all(qid),
                    key=lambda m: (m.occurred_at is None, m.occurred_at, m.id),
                    reverse=direction == "reverse",
                )
                try:
                    linker.link(store, index, resolver, memories)
                except CacheMiss as exc:
                    missing.append({"question_id": qid, "request_key": str(exc)})
                    continue
                completed.append(qid)
                snapshots[direction][qid] = {m.id: state(m) for m in store.iter_all(qid)}
        finally:
            cache.db.close()
            store.close()
        results[direction] = {
            "users_completed": len(completed),
            "cache_hits": cache.hits,
            "users_with_missing_cache": missing,
            "cached_judgments": cache.responses,
        }
    shared = set(snapshots["forward"]) & set(snapshots["reverse"])
    differences = []
    same, total = 0, 0
    for qid in sorted(shared):
        a, b = snapshots["forward"][qid], snapshots["reverse"][qid]
        for mid in sorted(a.keys() | b.keys()):
            total += 1
            if a.get(mid) == b.get(mid):
                same += 1
            else:
                differences.append(
                    {
                        "question_id": qid,
                        "memory": mid,
                        "forward": a.get(mid),
                        "reverse": b.get(mid),
                    }
                )
    results["comparison"] = {
        "users_compared": len(shared),
        "memories_compared": total,
        "memories_equal": same,
        "differences": differences,
        "scope": "cached batch traversal order, not out-of-order ingestion",
    }
    return results, snapshots["forward"]


def audit(split):
    progress = json.loads((REPO / f"stores/{split}-d1-link.json").read_text())
    qids = list(progress["done"])
    labels = json.loads((REPO / "research/fact-pairs-v1.json").read_text())["questions"]
    baseline = SQLiteMemoryStore(REPO / f"stores/{split}.db", read_only=True)
    anchors = SQLiteMemoryStore(REPO / f"stores/{split}-anchor-v2.db", read_only=True)
    index = NumpyFlatIndex(REPO / f"stores/{split}-index", dim=384)
    closures, changed, gold, isolation, invariant, replay_match = [], [], [], [], [], []
    with tempfile.TemporaryDirectory(prefix="lltm-d1-audit-") as temp:
        folder = Path(temp)
        backup(REPO / f"stores/{split}-d1.db", folder / "snapshot.db")
        target = SQLiteMemoryStore(folder / "snapshot.db", read_only=True)
        try:
            order, forward = replay(split, qids, index, folder)
            rebuild = rebuild_check(folder / "snapshot.db", qids, folder)
            for qid in qids:
                before = {m.id: m for m in baseline.iter_all(qid)}
                after = {m.id: m for m in target.iter_all(qid)}
                if before.keys() != after.keys():
                    invariant.append({"question_id": qid, "issue": "memory ids changed"})
                for mid in before.keys() & after.keys():
                    a, b = before[mid], after[mid]
                    allowed = set(FIELDS)
                    unexpected = [
                        k for k, v in asdict(a).items() if k not in allowed and v != asdict(b)[k]
                    ]
                    if unexpected:
                        invariant.append(
                            {
                                "question_id": qid,
                                "memory": mid,
                                "unexpected_changed_fields": unexpected,
                            }
                        )
                    if state(a) != state(b):
                        changed.append(
                            {
                                "question_id": qid,
                                "memory": mid,
                                "before": state(a),
                                "after": state(b),
                            }
                        )
                    if b.superseded_by:
                        successor = target.get(b.superseded_by)
                        if successor is None or successor.user_id != qid:
                            isolation.append(
                                {"question_id": qid, "memory": mid, "successor": b.superseded_by}
                            )
                    if b.status == "superseded" and (
                        a.status != b.status or a.superseded_by != b.superseded_by
                    ):
                        successor = target.get(b.superseded_by) if b.superseded_by else None
                        closures.append(
                            {
                                "question_id": qid,
                                "old": fact(a, anchors),
                                "new": fact(successor, anchors) if successor else None,
                                "after": state(b),
                                "review": "unreviewed",
                            }
                        )
                if qid in forward:
                    mismatches = [
                        mid for mid, m in after.items() if state(m) != forward[qid].get(mid)
                    ]
                    if mismatches:
                        replay_match.append({"question_id": qid, "memories": mismatches})
                for pair in labels.get(qid, {}).get("pairs", []):
                    old, new = after[pair["old"]], after[pair["new"]]
                    if pair["relation"] == "replace":
                        closed = old.status == "superseded" and old.superseded_by == new.id
                        historical = old.occurred_at is not None and old.id in {
                            m.id for m in as_of(list(after.values()), old.occurred_at)
                        }
                        current = (
                            new.occurred_at is not None
                            and new.id
                            in {m.id for m in as_of(list(after.values()), new.occurred_at)}
                            and old.id
                            not in {m.id for m in as_of(list(after.values()), new.occurred_at)}
                        )
                        ok = closed and historical and current
                    else:
                        ok = state(old) == state(before[old.id]) and state(new) == state(
                            before[new.id]
                        )
                        historical = current = None
                    gold.append(
                        {
                            "question_id": qid,
                            "pair": pair,
                            "correct": ok,
                            "historical_probe": historical,
                            "new_time_probe": current,
                        }
                    )
        finally:
            target.close()
            anchors.close()
            baseline.close()
    return {
        "split": split,
        "class": "development",
        "prompt_version": "transition-pair-v3",
        "users_completed": len(qids),
        "checkpoint": progress,
        "provider_calls_by_audit": 0,
        "changed_memories": changed,
        "closures": closures,
        "gold_pairs_draft": gold,
        "isolation_violations": isolation,
        "store_invariant_violations": invariant,
        "order_replay": order,
        "timeline_rebuild": rebuild,
        "forward_replay_differences_from_d1": replay_match,
        "limitations": [
            "gold annotations and close reviews require maintainer confirmation",
            "no full-population recall without labelled negative and positive pairs",
            "order replay is batch traversal, not arbitrary ingestion order",
        ],
    }


def render(report, review_path):
    reviews = report.get("close_review", {}).get("entries")
    if reviews is None:
        reviews = json.loads(review_path.read_text())["entries"] if review_path.exists() else []
    by_pair = {(r["split"], r["old"], r["new"]): r for r in reviews}
    counts = Counter()
    rows = []
    for close in report["closures"]:
        old, new = close["old"], close["new"]
        review = by_pair.get((report["split"], old["id"], new["id"] if new else None))
        label = review["label"] if review else "unreviewed"
        counts[label] += 1
        rows.append((close, label, review["reason"] if review else "待复查"))
    gold = report["gold_pairs_draft"]
    replaces = [p for p in gold if p["pair"]["relation"] == "replace"]
    undisputed = [p for p in replaces if p.get("annotation_status") != "disputed"]
    coexists = [p for p in gold if p["pair"]["relation"] == "coexist"]
    compared = report["order_replay"]["comparison"]
    rebuild = report.get("timeline_rebuild")
    verified = counts["correct"] + counts["incorrect"]
    lines = [
        f"# D 回填审计：{report['split']}",
        "",
        "> 由 tools/link_backfill_audit.py 生成；零模型调用，开发证据。",
        "> 关系与关闭复查均为助手草稿，待维护者确认；不是端到端答题成绩。",
        "",
        f"- 完成用户：{report['users_completed']}/100；仅审计完成用户。",
        f"- 累计判定：{sum(s['judgments'] for s in report['checkpoint']['done'].values())}；"
        f"replace 应用次数：{sum(s['linked'] for s in report['checkpoint']['done'].values())}。",
        f"- 相对原库实际改变的关闭关系：{len(rows)}；它与 replace 应用次数不同。",
        f"- 关闭复查：合理 {counts['correct']}、错误 {counts['incorrect']}、"
        f"不确定 {counts['uncertain']}、未复查 {counts['unreviewed']}。",
        f"- 已明确标注的关闭精确率：{counts['correct']}/{verified}；"
        "排除了不确定／未复查，不能当作全库真实精确率。",
        f"- 已有草稿替换对（含旧时点／新时点探针）通过："
        f"{sum(p['correct'] for p in replaces)}/{len(replaces)}；"
        f"并存对状态保持：{sum(p['correct'] for p in coexists)}/{len(coexists)}。",
        f"- 旧草稿替换对与本次复查冲突：{len(replaces) - len(undisputed)}；"
        f"排除争议后的草稿对子集通过 {sum(p['correct'] for p in undisputed)}/{len(undisputed)}。",
        f"- 跨用户后继错误：{len(report['isolation_violations'])}；"
        f"非预期存储字段变化：{len(report['store_invariant_violations'])}。",
        f"- 缓存正反遍历比较：{compared['memories_equal']}/{compared['memories_compared']} "
        f"条状态一致（{compared['users_compared']} 个用户）。",
        f"- 正向重放与实际 d1 结果不一致用户："
        f"{len(report['forward_replay_differences_from_d1'])}。",
        "",
        "批内遍历顺序检查不等于任意乱序到达实验。全库召回率与误替换率需要完整正负例标注；",
        "当前仅能给出已有草稿对子集的结果和实际关闭复查。TERMINATE 判定尚未应用，不能宣称",
        "全操作类型的时间状态正确。缓存缺失的用户不进入顺序对比。",
        "",
        "## 实际关闭逐对复查",
        "",
        "| 用户 | 旧事实 | 后继事实 | 草稿结论 | 原因 |",
        "|---|---|---|---|---|",
    ]
    for close, label, reason in rows:
        cells = [
            close["question_id"],
            close["old"]["content"],
            close["new"]["content"] if close["new"] else "后继缺失",
            label,
            reason,
        ]
        lines.append(
            "| " + " | ".join(c.replace("|", "\\|").replace("\n", " ") for c in cells) + " |"
        )
    if rebuild is not None:
        lines += [
            "",
            "## 时间线重建（仅临时副本）",
            "",
            f"- 对实际 d1 副本完整重建一次，改变 {len(rebuild['first_pass'])} 条状态。",
            f"- 再重建一次，改变 {len(rebuild['second_pass'])} 条状态；"
            "第二次为零才说明该重建结果幂等。",
        ]
    control = report.get("rebuild_control")
    if control:
        lines += [
            "",
            f"原库对照重建改变 {control['baseline_changed']} 条；"
            f"与 d1 共有 {control['shared_changed_memories']} 条，"
            f"仅 d1 出现的改变为 {len(control['d1_only_changed_memories'])} 条。",
            "共有变化不归因于本次关系判定回填。",
        ]
    disputed = report.get("disputed_draft_pairs", [])
    if disputed:
        lines += ["", "## 标注争议（保留旧标注）", ""]
        for pair in disputed:
            lines.append(
                f"- `{pair['question_id']}`：旧草稿 `{pair['pair']['relation']}`；"
                f"本次关闭复查 `{pair['review']['label']}`：{pair['review']['reason']}"
            )
    differences = compared["differences"]
    if differences:
        lines += ["", "## 正反遍历差异（保留失败）", ""]
        for difference in differences:
            lines.append(
                f"- `{difference['question_id']}` / `{difference['memory']}`："
                f"正向 `{difference['forward']}`；反向 `{difference['reverse']}`。"
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("dev100", "heldout100"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rebuild-control", type=Path)
    args = parser.parse_args()
    path = args.out.with_suffix(".json")
    md = args.out.with_suffix(".md")
    if path.exists() or md.exists():
        parser.error(f"preserve existing report: {path}")
    report = audit(args.split)
    review_path = REPO / "research/link-backfill-d1-close-review-v1.json"
    if review_path.exists():
        entries = json.loads(review_path.read_text())["entries"]
        pairs = {(c["old"]["id"], c["new"]["id"]) for c in report["closures"] if c["new"]}
        reviews = [r for r in entries if r["split"] == args.split and (r["old"], r["new"]) in pairs]
        report["close_review"] = {
            "source": str(review_path.relative_to(REPO)),
            "sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
            "status": "assistant draft, not maintainer-confirmed",
            "entries": reviews,
        }
        by_pair = {(r["old"], r["new"]): r for r in reviews}
        disputed = []
        for pair in report["gold_pairs_draft"]:
            review = by_pair.get((pair["pair"]["old"], pair["pair"]["new"]))
            if review and pair["pair"]["relation"] == "replace" and review["label"] == "incorrect":
                pair["annotation_status"] = "disputed"
                pair["review"] = review
                disputed.append(pair)
        report["disputed_draft_pairs"] = disputed
        for close in report["closures"]:
            review = by_pair.get((close["old"]["id"], close["new"]["id"])) if close["new"] else None
            if review:
                close["review"] = review["label"]
                close["review_reason"] = review["reason"]
    if args.rebuild_control:
        control = json.loads(args.rebuild_control.read_text())
        if control["baseline_source"] != f"stores/{args.split}.db":
            parser.error("rebuild control belongs to another split")
        report["rebuild_control"] = control
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")
    md.write_text(render(report, review_path))
    print(
        json.dumps(
            {
                "split": args.split,
                "users_completed": report["users_completed"],
                "closures": len(report["closures"]),
                "output": str(path),
            }
        )
    )


if __name__ == "__main__":
    main()
