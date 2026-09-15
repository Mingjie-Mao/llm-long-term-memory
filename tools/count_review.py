"""Build a source-grounded review packet; human decisions are never inferred.

The HTML is a local review interface. Exported decisions bind the packet and its inputs.
Only reviewed, source-anchored entity questions can be exported for later experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

from entity_count_probes import _fingerprint

from llm_long_term_memory.store.session_keys import external_session_id

REPO = Path(__file__).resolve().parent.parent
POLICY = REPO / "configs/count-policy-v1.json"
PACKET = REPO / "results/review/count-v1/packet.json"
GOLD = REPO / "results/review/count-v1/gold.json"
MANIFEST = REPO / "results/review/count-v1/gold-manifest.json"


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def development_boundary(connection, repo: Path = REPO) -> tuple[set[str], set[str]]:
    """Protect namespaces AND the external sessions they share with readable data."""
    path = repo / "results/manifests/v3-reasoning-dev60.json"
    blocked = set(read(path)["question_ids"])
    if not blocked:
        raise ValueError("empty protected manifest")
    sessions = {
        external_session_id(row["id"])
        for row in connection.execute("SELECT id, user_id FROM sessions")
        if row["user_id"] in blocked
    }
    return blocked, sessions


def cluster_items(items: list[dict]) -> None:
    """Connected components of shared source sessions, including transitive overlap."""
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    for i, item in enumerate(items):
        keys = {s["session_key"] for s in item["sources"]}
        keys.add("namespace:" + item["namespace"])
        for key in keys:
            if key in owners:
                parent[find(i)] = find(owners[key])
            owners[key] = i
    groups = {}
    for i, item in enumerate(items):
        groups.setdefault(find(i), []).append(item["id"])
    for i, item in enumerate(items):
        item["cluster_id"] = "cluster_" + digest(sorted(groups[find(i)]))[:16]


def shared_evidence_audit(items: list[dict]) -> dict:
    """Report which questions rest on the same recorded turn, and refuse if one escaped.

    Two mechanisms make count questions non-independent, and only one of them is real.

    *Shared evidence* is real: the source corpus reuses conversations across question
    ids, so one recorded turn can be the evidence for probes in two namespaces — the
    stored meal that two probes both counted. Questions like that are one observation,
    not two, and a paired test that treats them as two overstates its own n.

    *A shared object name* is not. Two users who both read `sapiens: a brief history of
    humankind` produce identical object text over completely separate evidence, and
    merging them would delete a real observation. Eight object strings repeat across the
    probe set and only one pair of questions shares an actual turn; merging on text would
    therefore have been wrong seven times out of eight. So the criterion is the recorded
    turn, never the label.

    Clustering already unions on the external session, which is coarser than the turn, so
    this is an invariant rather than a new grouping: anything sharing a turn must already
    share a cluster. It is checked rather than assumed so that a later, finer clustering
    rule cannot silently split a duplicate pair back into two observations.
    """
    owners: dict[str, list[str]] = {}
    for item in items:
        for source in item["sources"]:
            owners.setdefault(source["id"], []).append(item["id"])
    cluster = {item["id"]: item["cluster_id"] for item in items}
    pairs, escaped = set(), []
    for sharers in owners.values():
        for i, a in enumerate(sorted(set(sharers))):
            for b in sorted(set(sharers))[i + 1 :]:
                pairs.add((a, b))
                if cluster[a] != cluster[b]:
                    escaped.append((a, b))
    if escaped:
        raise ValueError(f"questions share a recorded turn but not a cluster: {escaped[:3]}")
    return {
        "criterion": "shared recorded turn, not shared object label",
        "question_pairs_sharing_a_turn": len(pairs),
        "all_in_one_cluster": True,
    }


def build_packet(repo: Path = REPO) -> dict:
    paths = {
        "policy": repo / "configs/count-policy-v1.json",
        "entity_probes": repo / "results/analysis/entity-count-probes.json",
        "legacy_overlay": repo / "results/analysis/count-probe-gold-corrections.json",
        "legacy_probes": repo / "results/analysis/synthesis-probes.json",
        "legacy_split": repo / "results/manifests/v4-probe-split.json",
        "protected_manifest": repo / "results/manifests/v3-reasoning-dev60.json",
    }
    blobs = {name: path.read_bytes() for name, path in paths.items()}
    payloads = {name: json.loads(blob) for name, blob in blobs.items()}
    policy = payloads["policy"]
    entity = payloads["entity_probes"]["probes"]
    old = {p["probe_id"]: p for p in payloads["legacy_probes"]["probes"]}
    overlay = payloads["legacy_overlay"]["members"]
    development = set(payloads["legacy_split"]["development"])
    items, quarantine = [], []
    connection = sqlite3.connect((repo / "stores/train150.db").as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("BEGIN")
        blocked, protected_sessions = development_boundary(connection, repo)
        source_hash = _fingerprint(connection)
        specs = [("entity", p, None) for p in entity]
        specs += [("legacy", old[m["probe_id"]], m) for m in overlay]
        for kind, probe, correction in specs:
            item_id = probe["probe_id"]
            if correction:
                item_id += ":" + correction["memory_id"]
            mids = probe["evidence_memory_ids"]
            # Read ids/anchors first. Do not put protected question text or raw turns in
            # the packet, including when only a shared source crosses the boundary.
            anchors = [
                connection.execute(
                    "SELECT id,user_id,source_session_id,source_turn_index "
                    "FROM memories WHERE id=?",
                    (mid,),
                ).fetchone()
                for mid in mids
            ]
            reason = None
            if kind == "legacy" and probe["probe_id"] not in development:
                reason = "legacy_heldout"
            elif probe["namespace"] in blocked:
                reason = "protected_namespace"
            elif any(row is None for row in anchors):
                reason = "missing_memory"
            elif any(
                external_session_id(row["source_session_id"]) in protected_sessions
                for row in anchors
            ):
                reason = "protected_shared_session"
            elif any(row["user_id"] != probe["namespace"] for row in anchors):
                reason = "namespace_mismatch"
            if reason:
                quarantine.append({"id": item_id, "kind": kind, "reason": reason})
                continue
            if correction:
                anchors = [a for a in anchors if a["id"] == correction["memory_id"]]
            memories, sources = [], {}
            for anchor in anchors:
                memory = dict(
                    connection.execute(
                        "SELECT * FROM memories WHERE id=?", (anchor["id"],)
                    ).fetchone()
                )
                memories.append(
                    {
                        k: memory[k]
                        for k in (
                            "id",
                            "content",
                            "predicate",
                            "object",
                            "scope",
                            "status",
                            "event_time",
                        )
                    }
                )
                # Include adjacent dialogue so a quoted reply has its antecedent. This
                # is a bounded evidence-pool instrument, not end-to-end retrieval.
                index = anchor["source_turn_index"]
                if index is None or not anchor["source_session_id"]:
                    continue
                for turn in connection.execute(
                    "SELECT * FROM turns WHERE session_id=? "
                    "AND turn_index BETWEEN ? AND ? ORDER BY turn_index",
                    (anchor["source_session_id"], max(0, index - 1), index + 1),
                ):
                    key = digest([turn["session_id"], turn["turn_index"]])[:20]
                    sources[key] = {
                        "id": key,
                        "session_key": external_session_id(turn["session_id"]),
                        "turn_index": turn["turn_index"],
                        "role": turn["role"],
                        "text": turn["content"],
                        "recorded_at": turn["ts"],
                    }
            relation = probe.get("relation")
            allowed = set(policy["relations"].get(relation, {}).get("predicates", []))
            unsafe = (
                sorted({m["predicate"] for m in memories} - allowed) if kind == "entity" else []
            )
            proposed_entities = probe.get("entities", []) if kind == "entity" else []
            items.append(
                {
                    "id": item_id,
                    "kind": kind,
                    "namespace": probe["namespace"],
                    "question": probe["question"],
                    "relation": relation,
                    "memories": memories,
                    "sources": sorted(
                        sources.values(), key=lambda s: (s["session_key"], s["turn_index"])
                    ),
                    "flags": (["unapproved_predicates: " + ", ".join(unsafe)] if unsafe else [])
                    + (["missing_raw_source"] if not sources else []),
                    "proposed_entities": proposed_entities,
                    # Only a legacy item carries one, and only because re-reviewing that
                    # model decision *is* the item. An entity question never has one: its
                    # old "suggestion" was `unsafe` restated, which `flags` already says,
                    # and stating it as a verdict anchored the reviewer for no new
                    # information. Gold members come from entity questions alone, so
                    # nothing a model decided can reach the frozen set through here.
                    "prior_model_verdict": correction["verdict"] if correction else None,
                }
            )
        cluster_items(items)
        packet = {
            "schema_version": 1,
            "status": "pending_human_review",
            "provider_calls": 0,
            "scope": "development_only_bounded_source_pool",
            "policy": policy,
            "inputs": {
                name: {
                    "path": path.relative_to(repo).as_posix(),
                    "sha256": hashlib.sha256(blobs[name]).hexdigest(),
                }
                for name, path in paths.items()
            },
            "store_fingerprint_kind": "sqlite-evidence-content-v1",
            "store_fingerprint": source_hash,
            "items": items,
            "quarantine": quarantine,
            "summary": {
                "reviewable": dict(Counter(i["kind"] for i in items)),
                "quarantined": dict(Counter(i["kind"] for i in quarantine)),
                "quarantine_reasons": dict(Counter(i["reason"] for i in quarantine)),
                "clusters": len({i["cluster_id"] for i in items}),
                "shared_evidence": shared_evidence_audit(items),
            },
        }
        packet["packet_sha256"] = digest(packet)
        return packet
    finally:
        connection.close()


def verify_packet(packet: dict) -> None:
    unsigned = {k: v for k, v in packet.items() if k != "packet_sha256"}
    if digest(unsigned) != packet.get("packet_sha256"):
        raise ValueError("review packet content changed")


def validate_decisions(packet: dict, decisions: dict) -> list[dict]:
    verify_packet(packet)
    if decisions.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("decisions belong to a different packet")
    if (
        decisions.get("reviewer_kind") != "human"
        or not str(decisions.get("reviewer") or "").strip()
    ):
        raise ValueError("human reviewer identity is required")
    if decisions.get("policy_approved") is not True or not decisions.get("reviewed_at"):
        raise ValueError("policy and review time must be confirmed")
    rows = decisions.get("decisions", [])
    expected = {i["id"]: i for i in packet["items"]}
    if len(rows) != len(expected) or {r.get("id") for r in rows} != set(expected):
        raise ValueError("decisions must cover every reviewable item exactly once")
    accepted = []
    for row in rows:
        item = expected[row["id"]]
        if (
            row.get("verdict") not in {"accept", "exclude", "rewrite"}
            or not row.get("reason", "").strip()
        ):
            raise ValueError("each decision needs a verdict and reason")
        if row["verdict"] != "accept" or item["kind"] != "entity":
            continue
        if item["flags"]:
            raise ValueError("flagged question must be excluded or rewritten in a new packet")
        if row.get("scope_complete") is not True:
            raise ValueError("accepted question requires source-scope completeness review")
        members = row.get("members")
        if not isinstance(members, list):
            raise ValueError("accepted entity question needs an explicit member list")
        sources = {s["id"]: s for s in item["sources"]}
        names = set()
        for member in members:
            name = str(member.get("entity") or "").strip()
            quote = str(member.get("quote") or "").strip()
            source = sources.get(member.get("source_id"))
            key = name.casefold()
            if (
                not name
                or key in names
                or not quote
                or source is None
                or quote not in source["text"]
            ):
                raise ValueError("members must be distinct and cite a verbatim supplied source")
            names.add(key)
        if not members:
            zero = row.get("zero_evidence") or {}
            source = sources.get(zero.get("source_id"))
            quote = zero.get("quote")
            if not quote or source is None or quote not in source["text"]:
                raise ValueError("zero needs explicit source evidence, not an empty list")
        accepted.append(
            {
                "id": item["id"],
                "question": item["question"],
                "namespace": item["namespace"],
                "relation": item["relation"],
                "sources": item["sources"],
                "memories": item["memories"],
                "cluster_id": item["cluster_id"],
                # Kept so a later comparison can score the unreviewed generator output
                # as its control arm without reopening the packet.
                "proposed_entities": item["proposed_entities"],
                "members": members,
                "answer": len(members),
            }
        )
    return accepted


def prior_verdict_agreement(packet: dict, decisions: dict) -> dict:
    """How often the human landed on the prior model verdict, as a diagnostic only.

    Recorded, never enforced. A high rate is not proof of contamination — the model may
    simply have been right about obvious items — and a low one is not proof of
    independence. It exists so that a later reader can see the number at all: the review
    page now puts every conclusion-shaped field after the evidence, and this is the only
    way to notice if that stopped being true. Entity questions carry no prior verdict, so
    this covers the legacy re-reviews alone and nothing here touches a gold member.
    """
    # The overlay says keep/exclude; the review page says accept/exclude/rewrite.
    same = {"keep": "accept", "exclude": "exclude"}
    verdicts = {row["id"]: row.get("verdict") for row in decisions.get("decisions", [])}
    compared = [
        (same.get(item["prior_model_verdict"]), verdicts.get(item["id"]))
        for item in packet["items"]
        if item.get("prior_model_verdict")
    ]
    agreed = sum(1 for prior, human in compared if prior is not None and prior == human)
    return {
        "scope": "legacy_re_reviews_only",
        "compared": len(compared),
        "agreed": agreed,
        "rate": round(agreed / len(compared), 4) if compared else None,
        "enforced": False,
    }


def current_store_fingerprint(repo: Path = REPO) -> str:
    """Re-read the evidence store so a frozen set cannot outlive the data it quotes."""
    store = repo / "stores/train150.db"
    if not store.exists():
        raise ValueError("evidence store is missing; cannot bind a frozen set")
    connection = sqlite3.connect(store.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("BEGIN")
        return _fingerprint(connection)
    finally:
        connection.close()


def freeze_gold(packet: dict, decisions: dict, repo: Path = REPO) -> dict:
    """Bind accepted human decisions to the packet, its inputs and the live store.

    Freezing records what a reviewer decided over a bounded source pool. It does not
    make those decisions correct, and it is not an independent test set.
    """
    accepted = validate_decisions(packet, decisions)
    if not accepted:
        raise ValueError("no accepted entity question; nothing to freeze")
    live = current_store_fingerprint(repo)
    if live != packet["store_fingerprint"]:
        raise ValueError("evidence store changed since the packet was built; rebuild first")
    gold = {
        "schema_version": 1,
        "name": "count-gold-v1",
        "scope": packet["scope"],
        "provider_calls": 0,
        "bound_to": {
            "packet_sha256": packet["packet_sha256"],
            "decisions_sha256": digest(decisions),
            "inputs": packet["inputs"],
            "store_fingerprint_kind": packet["store_fingerprint_kind"],
            "store_fingerprint": packet["store_fingerprint"],
        },
        "reviewer": decisions["reviewer"],
        "reviewer_kind": decisions["reviewer_kind"],
        "reviewed_at": decisions["reviewed_at"],
        "questions": accepted,
        "clusters": sorted({q["cluster_id"] for q in accepted}),
        "prior_verdict_agreement": prior_verdict_agreement(packet, decisions),
        "summary": {
            "questions": len(accepted),
            "clusters": len({q["cluster_id"] for q in accepted}),
            "members": sum(q["answer"] for q in accepted),
            "zero_answers": sum(1 for q in accepted if q["answer"] == 0),
        },
    }
    gold["gold_sha256"] = digest(gold)
    return gold


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "validate", "freeze"])
    parser.add_argument("--packet", type=Path, default=PACKET)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--out", type=Path, default=GOLD)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    if args.command == "build":
        packet = build_packet()
        args.packet.parent.mkdir(parents=True, exist_ok=True)
        args.packet.write_text(
            json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        from count_review_ui import render

        args.packet.with_suffix(".html").write_text(render(packet), encoding="utf-8")
        print(json.dumps(packet["summary"], ensure_ascii=False))
        return 0
    if args.decisions is None:
        parser.error(f"{args.command} requires --decisions")
    packet, decisions = read(args.packet), read(args.decisions)
    try:
        if args.command == "validate":
            accepted = validate_decisions(packet, decisions)
            print(f"human review valid: {len(accepted)} accepted entity questions")
            return 0
        gold = freeze_gold(packet, decisions)
    except (ValueError, KeyError, TypeError) as exc:
        print(f"STOP: {exc}")
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(gold, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # The gold quotes raw dataset turns, so it stays out of version control like every
    # other raw record here. The manifest carries the hashes and the counts, which is what
    # a later claim has to be checkable against.
    manifest = {k: v for k, v in gold.items() if k != "questions"}
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(gold["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
