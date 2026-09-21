"""Can reshaping the context buy anything? Answered from the v2c rows, with no API calls.

Two mechanisms are proposed. v5.0 stops treating raw conversation as a fallback and
retrieves it alongside structured memory; v5.1 replaces the flat top-k with whole
sessions in event order. Both cost an ingest-free but answer-paying run, so the
question to settle first is free: **is the answer anywhere the reshaped context would
reach but the current context does not?**

The session-level numbers already on record cannot settle it. v2c recalled at least one
gold session for 46 of 48 questions and every gold session for 45 — and still got 11
wrong, 9 of them with complete session recall. That says retrieval found the right
conversations. It says nothing about whether the context carried the right *sentence*,
because a session counts as recalled when one of its thirty facts is in the top-20.

So this measures where the gold answer literal actually is, over four nested haystacks:

1. `selected`  — the memories v2c's answerer saw. The current context.
2. `store`     — every memory in the namespace. Extraction kept it; ranking dropped it.
                 The headroom for repacking alone (v5.1).
3. `local_raw` — raw turns of the sessions the selected memories came from. Extraction
                 dropped it, but the conversation v2c already located still holds it.
                 The headroom for a source-local parallel window (v5.0).
4. `all_raw`   — every turn in the namespace. The lossless ceiling; reaching it needs
                 archive-wide retrieval, which v2b measured as noisy.

A question whose gold is already in `selected` cannot be fixed by either mechanism: the
evidence was in front of the reader. That number is the honest cap on both.

`answer_present` is reused rather than reimplemented so that a memory and a raw turn are
judged by the same matcher; turns are wrapped as content-only memories to get there.
Multi-session and preference golds are not string-matchable and are reported apart
instead of being scored as misses.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import read_jsonl, write_report  # noqa: E402

# Hand audit of every wrong answer, recorded because at this size it is the reliable
# instrument and the automated ladder is not. Eleven failures is small enough to read,
# and reading them overturned the automated reading twice: the generous numeric branch
# of `answer_present` claimed the store already held "1 week", and the abstention golds
# matched the raw turns by construction. Each entry names where the evidence actually
# was, checked against the store and its raw turns.
#
#   reader        — the evidence was in the selected context and the answer still failed
#   local_raw     — a user or assistant sentence the extractor dropped, in the raw turns
#                   of a session that was already selected (what v5.0 would supply)
#   repack        — the fact was in the store and not selected (what v5.1 would supply)
#   retrieval     — the gold session was never found
AUDIT = {
    "0ddfec37_abs": (
        "reader",
        "answered with a count instead of declining a false premise",
    ),
    "gpt4_70e84552_abs": ("reader", "answered instead of declining a false premise"),
    "a9f6b44c": ("reader", "listed the qualifying events and never produced the count"),
    "gpt4_2f8be40d": ("reader", "counted four weddings where the gold names three"),
    "d905b33f": (
        "reader",
        "$24 and the original price were both in context; read it as 30 -> 30",
    ),
    "dd2973ad": (
        "reader",
        "the 2 AM memory was selected; the answer still said it did not know",
    ),
    "gpt4_7a0daae1": (
        "local_raw",
        "'I just received my new tennis racket today' is a raw turn of a session "
        "that was selected; only assistant recommendations were extracted from it",
    ),
    "a4996e51": (
        "local_raw",
        "'up to 50 hours per week' is an assistant turn; the surviving memory "
        "says 40-45 and was not selected",
    ),
    "58ef2f1c": (
        "local_raw",
        "structured kept 'February 2023'; the raw turn says the dinner was "
        '"back on Valentine\'s Day", which the gold states as February 14th. The '
        "evidence is a raw turn of a selected session, but reaching the gold from "
        "it also needs the reader to know the date of Valentine's Day",
    ),
    "gpt4_74aed68e": (
        "repack",
        "the dated 'February 14' memory was in the store and unselected; the "
        "undated restatement carried the session date instead",
    ),
    "e48988bc": ("retrieval", "the gold session was never found"),
}

ROWS = REPO / "results/raw/two_stage_v2c.reasoning48-v2c8.jsonl"
STORE = REPO / "stores/v2c-reasoning48.db"


def match_kind(gold: str, memories: list) -> str | None:
    """Which of `answer_present`'s three branches fired, or None.

    The branch matters more than the boolean. `answer_present` is a coverage
    diagnostic, and two of its branches are deliberately generous: a numeric gold is
    satisfied when its digits appear *anywhere* in the haystack, and a multi-word gold
    when 80% of its tokens do. Over 2,949 memories that is close to free, so a gate
    that counted those as "the answer is reachable" would manufacture headroom out of
    the matcher. Reported as `exact`, `numeric` or `tokens` so a weak match can be read
    as weak.
    """
    from llm_long_term_memory.evaluation.extraction_coverage import _NUM, _norm, _searchable

    gold_norm = _norm(gold)
    if not gold_norm:
        return None
    haystack = _norm(" || ".join(_searchable(m) for m in memories))
    if gold_norm in haystack:
        return "exact"
    numbers = _NUM.findall(gold)
    if numbers and all(n in set(_NUM.findall(haystack)) for n in numbers):
        return "numeric"
    tokens = gold_norm.split()
    if len(tokens) > 1:
        words = set(haystack.split())
        if sum(1 for t in tokens if t in words) / len(tokens) >= 0.8:
            return "tokens"
    return None


def _as_memory(turn, index: int):
    """One raw turn as a content-only memory, so the same matcher judges both."""
    from llm_long_term_memory.store import Memory

    return Memory(
        id=f"turn_{index}",
        user_id="raw",
        type="episodic",
        content=turn.content,
        token_count=len(turn.content.split()),
    )


def analyse(rows_path: Path, store_path: Path) -> dict:
    from llm_long_term_memory.evaluation.datasets.longmemeval import load
    from llm_long_term_memory.evaluation.extraction_coverage import (
        UNMEASURABLE_TYPES,
        answer_present,
    )
    from llm_long_term_memory.store import SQLiteMemoryStore

    rows = read_jsonl(rows_path)
    instances = {item.question_id: item for item in load("s", REPO / "data")}
    store = SQLiteMemoryStore(store_path, read_only=True)
    store.initialize()

    per_question = []
    try:
        for row in rows:
            qid = row["question_id"]
            instance = instances[qid]
            namespace = instance.store_namespace
            gold = str(row.get("gold") or "")
            # An abstention gold is a sentence declining to answer ("The information
            # provided is not enough. You mentioned collecting autographed baseball but
            # not football."). Its words are drawn from the conversation, so it matches
            # the raw turns by construction while adding raw text cannot fix it — the
            # failure is that the reader answered instead of declining. Excluded, not
            # scored as reachable.
            measurable = (
                bool(gold.strip())
                and not row.get("is_abstention")
                and row["question_type"] not in UNMEASURABLE_TYPES
            )

            selected = [
                memory
                for item in row["notes"]["retrieval"]
                if (memory := store.get(item["memory_id"])) is not None
            ]
            everything = list(store.iter_all(namespace))
            selected_sessions = {m.source_session_id for m in selected if m.source_session_id}
            all_sessions = store.session_ids_for_user(namespace)

            local_turns = [
                _as_memory(turn, index)
                for session_id in sorted(selected_sessions)
                for index, turn in enumerate(store.turns_for_session(session_id))
            ]
            all_turns = [
                _as_memory(turn, index)
                for session_id in sorted(all_sessions)
                for index, turn in enumerate(store.turns_for_session(session_id))
            ]

            # How much of a located conversation actually reached the reader. A session
            # represented by one fact out of thirty is "recalled" and nearly absent.
            in_context = Counter(m.source_session_id for m in selected if m.source_session_id)
            in_store = Counter(m.source_session_id for m in everything if m.source_session_id)
            coverage = [
                {
                    "session_id": session_id,
                    "memories_in_store": in_store.get(session_id, 0),
                    "memories_in_context": in_context.get(session_id, 0),
                    "turns_in_store": len(store.turns_for_session(session_id)),
                }
                for session_id in sorted(selected_sessions)
            ]

            per_question.append(
                {
                    "question_id": qid,
                    "question_type": row["question_type"],
                    "correct": bool(row["correct"]),
                    "measurable": measurable,
                    "fallback_level": row["notes"].get("fallback_level"),
                    "raw_reached_the_reader": row["notes"].get("fallback_level")
                    not in (
                        None,
                        "none",
                    ),
                    "all_source_sessions_recalled": bool(
                        row["notes"].get("all_source_sessions_recalled")
                    ),
                    "gold_in_selected": answer_present(gold, selected) if measurable else None,
                    "gold_in_selected_kind": match_kind(gold, selected) if measurable else None,
                    "gold_in_store": answer_present(gold, everything) if measurable else None,
                    "gold_in_store_kind": match_kind(gold, everything) if measurable else None,
                    "gold_in_local_raw": answer_present(gold, local_turns) if measurable else None,
                    "gold_in_local_raw_kind": match_kind(gold, local_turns) if measurable else None,
                    "gold_in_all_raw": answer_present(gold, all_turns) if measurable else None,
                    "gold_in_all_raw_kind": match_kind(gold, all_turns) if measurable else None,
                    "sessions_in_context": len(selected_sessions),
                    "session_coverage": coverage,
                    "context_tokens": row["context_tokens"],
                }
            )
    finally:
        store.close()

    return {
        "rows": str(rows_path.relative_to(REPO)),
        "store": str(store_path.relative_to(REPO)),
        "questions": len(per_question),
        "correct": sum(e["correct"] for e in per_question),
        "raw_never_reached_the_reader": sum(not e["raw_reached_the_reader"] for e in per_question),
        "headroom": _headroom(per_question),
        "audit": _audit(per_question),
        "coverage": _coverage(per_question),
        "per_question": per_question,
    }


def _audit(entries: list[dict]) -> dict:
    """The hand attribution, aligned against the run so a stale entry cannot pass."""
    wrong = {e["question_id"] for e in entries if not e["correct"]}
    missing = sorted(wrong - set(AUDIT))
    extra = sorted(set(AUDIT) - wrong)
    if missing or extra:
        raise SystemExit(
            f"the audit does not match this run: unattributed {missing}, stale {extra}"
        )
    counts = Counter(cause for cause, _ in AUDIT.values())
    return {
        "by_cause": dict(sorted(counts.items())),
        "reachable_by_v5_0": counts["local_raw"],
        "reachable_by_v5_1": counts["repack"],
        "not_reachable_by_either": counts["reader"] + counts["retrieval"],
        "per_question": {q: {"cause": c, "note": n} for q, (c, n) in sorted(AUDIT.items())},
    }


def _headroom(entries: list[dict]) -> dict:
    """Where the gold is, among the wrong answers whose gold is a literal.

    The ladder reads **exact matches only**. The generous branches are what make
    `answer_present` a good coverage diagnostic and a bad locator: because a numeric
    gold is satisfied by its digits occurring anywhere, "in the store" fired for
    `gpt4_7a0daae1` — whose gold is "1 week" — and pre-empted the rung that was
    actually true. The user's sentence "I just received my new tennis racket today",
    in a session dated a week after the purchase, is in the raw turns of a session
    that *was* selected, and the extractor had dropped it. Scoring the weak match as
    "already covered" hid a real source-local hit, which is the opposite of what a
    gate is for.
    """
    wrong = [e for e in entries if not e["correct"] and e["measurable"]]

    def at(entry: dict, field: str) -> bool:
        return entry[f"{field}_kind"] == "exact"

    already = [e for e in wrong if at(e, "gold_in_selected")]
    rest = [e for e in wrong if not at(e, "gold_in_selected")]
    repack = [e for e in rest if at(e, "gold_in_store")]
    rest = [e for e in rest if not at(e, "gold_in_store")]
    local = [e for e in rest if at(e, "gold_in_local_raw")]
    rest = [e for e in rest if not at(e, "gold_in_local_raw")]
    archive = [e for e in rest if at(e, "gold_in_all_raw")]
    nowhere = [e for e in rest if not at(e, "gold_in_all_raw")]
    return {
        "match_strength": dict(
            sorted(Counter(e["gold_in_store_kind"] or "none" for e in wrong).items())
        ),
        "wrong_total": sum(not e["correct"] for e in entries),
        "wrong_measurable": len(wrong),
        "wrong_unmeasurable": sum(not e["correct"] and not e["measurable"] for e in entries),
        "already_in_context": [e["question_id"] for e in already],
        "repack_only": [e["question_id"] for e in repack],
        "source_local_raw": [e["question_id"] for e in local],
        "archive_wide_raw": [e["question_id"] for e in archive],
        "nowhere": [e["question_id"] for e in nowhere],
    }


def _coverage(entries: list[dict]) -> dict:
    """How thin a located conversation is by the time the reader sees it."""
    pairs = [
        (row["memories_in_context"], row["memories_in_store"])
        for entry in entries
        for row in entry["session_coverage"]
    ]
    thin = [(c, s) for c, s in pairs if s and c / s <= 0.25]
    return {
        "sessions_in_context_total": len(pairs),
        "memories_in_context": sum(c for c, _ in pairs),
        "memories_in_those_sessions": sum(s for _, s in pairs),
        "mean_fraction_of_session_shown": (
            round(sum(c / s for c, s in pairs if s) / max(1, len([1 for _, s in pairs if s])), 3)
        ),
        "sessions_shown_at_most_a_quarter": len(thin),
    }


def _row(label: str, ids: list[str], reachable: str) -> str:
    listed = ", ".join(f"`{qid}`" for qid in ids) or "—"
    return f"| {label} | {len(ids)} | {reachable} | {listed} |"


def render(result: dict) -> str:
    head = result["headroom"]
    audit = result["audit"]
    cov = result["coverage"]
    lines = [
        "# v5.0 / v5.1 offline gate (zero model calls)",
        "",
        f"> Rows: `{result['rows']}` · store: `{result['store']}`. Derived from the",
        "> committed v2c run; no question is re-answered and no gold is read beyond the",
        "> string matching already used by the extraction-coverage diagnostic.",
        "",
        f"- questions: **{result['questions']}**, correct **{result['correct']}**",
        f"- questions where no raw turn ever reached the reader: "
        f"**{result['raw_never_reached_the_reader']}/{result['questions']}**",
        f"- wrong answers: **{head['wrong_total']}** "
        f"({head['wrong_measurable']} string-matchable, "
        f"{head['wrong_unmeasurable']} not: abstention, multi-session or preference golds)",
        f"- how the matchable ones match against the whole store: "
        f"{head['match_strength']} — `exact` is a real phrase hit, `numeric` only means "
        f"the digits occur somewhere, `tokens` only that 80% of the words do",
        "",
        "## Attribution of every wrong answer (hand audit)",
        "",
        "At eleven failures the automated ladder below is supporting evidence, not the",
        "instrument. Reading the rows overturned it twice, so each failure was checked",
        "against the store and its raw turns by hand.",
        "",
        "| cause | n | what it means |",
        "|---|---:|---|",
        f"| `reader` | {audit['by_cause'].get('reader', 0)} | "
        "the evidence was in the selected context and the answer still failed |",
        f"| `local_raw` | {audit['by_cause'].get('local_raw', 0)} | "
        "a sentence the extractor dropped, in the raw turns of an already-selected "
        "session — **v5.0** |",
        f"| `repack` | {audit['by_cause'].get('repack', 0)} | "
        "the fact was in the store and not selected — **v5.1** |",
        f"| `retrieval` | {audit['by_cause'].get('retrieval', 0)} | "
        "the gold session was never found |",
        "",
        "| question | cause | note |",
        "|---|---|---|",
        *(
            f"| `{qid}` | `{entry['cause']}` | {entry['note']} |"
            for qid, entry in audit["per_question"].items()
        ),
        "",
        "## Where a string-matchable gold is, by exact match only",
        "",
        "Supporting only. Seven of the eleven golds cannot be located this way: two are",
        "abstention sentences whose words come from the conversation, two are derived",
        "values that no turn ever spells out, and the rest are multi-session or",
        "preference answers the coverage diagnostic already treats as unmeasurable.",
        "",
        "| location | n | reachable by | questions |",
        "|---|---:|---|---|",
        _row(
            "already in the selected memories",
            head["already_in_context"],
            "neither mechanism — the reader had it",
        ),
        _row("in the store, not selected", head["repack_only"], "v5.1 repacking"),
        _row(
            "only in raw turns of located sessions",
            head["source_local_raw"],
            "v5.0 source-local window",
        ),
        _row("only in raw turns elsewhere", head["archive_wide_raw"], "archive-wide retrieval"),
        _row("nowhere", head["nowhere"], "neither — extraction or gold matching"),
        "",
        "## How much of a located conversation the reader sees",
        "",
        f"- sessions represented in a context, summed over questions: "
        f"**{cov['sessions_in_context_total']}**",
        f"- memories from them shown: **{cov['memories_in_context']}** of "
        f"**{cov['memories_in_those_sessions']}** stored",
        f"- mean fraction of a located session shown: "
        f"**{cov['mean_fraction_of_session_shown']:.1%}**",
        f"- sessions shown at most a quarter of: **{cov['sessions_shown_at_most_a_quarter']}**",
        "",
        "## Decision",
        "",
        f"**v5.0 is worth running: {audit['reachable_by_v5_0']} of 11 failures are a "
        "sentence the extractor dropped from a session retrieval already found.** All "
        "three have the same shape, which is the shape the mechanism addresses, and in "
        f"{result['raw_never_reached_the_reader']} of {result['questions']} questions "
        "no raw turn reached the reader at all.",
        "",
        f"**v5.1 repacking alone reaches {audit['reachable_by_v5_1']}**, and that one is "
        "compounded by a cheaper defect: the dated memory was unselected while an "
        "undated restatement carried the session date as its event time. Splitting event "
        "time from observation time may fix it without touching the context shape, so "
        "v5.1 is not registered on this evidence.",
        "",
        f"**{audit['not_reachable_by_either']} of 11 are out of reach of either "
        "mechanism** — six where the evidence was in front of the reader, one genuine "
        "retrieval miss. That is the dominant bucket and it is a reader question, not a "
        "context-shape question.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=ROWS)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument(
        "--json-out", type=Path, default=REPO / "results/analysis/v5-offline-gate.json"
    )
    parser.add_argument("--md-out", type=Path, default=REPO / "results/analysis/v5-offline-gate.md")
    args = parser.parse_args()
    if not args.store.is_file():
        raise SystemExit(f"{args.store} is missing; this gate reads the committed v2c store")
    result = analyse(args.rows, args.store)
    print(write_report(result, render, json_out=args.json_out, md_out=args.md_out), end="")


if __name__ == "__main__":
    main()
