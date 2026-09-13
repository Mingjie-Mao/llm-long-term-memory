"""Answer the synthesis probes with a frozen variant and grade without a judge.

Every probe's ground truth is derived, so grading is arithmetic and string
comparison rather than a model verdict. That removes the judge from the loop
entirely: one answerer call per probe, a second only when the answerer asks for
source, and no `gemma` quota at all.

**This is an instrument reading, not a benchmark score.** Two limits are measured
rather than assumed, and both are written into every row so a number cannot be
quoted without them:

  `context_complete`   whether the top-k context actually contained every fact the
                       ground truth counted. Where it did not, a perfect counter
                       still scores wrong, and the row measures retrieval.
  `anchor_conflict`    whether an evidence memory's `event_time` contradicts a date
                       written in its own text. Where it does, a model reading the
                       text correctly is graded against a different date.

Aggregates are reported split on both flags. A single headline number over all 229
probes would average a synthesis measurement together with a retrieval ceiling.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from contextlib import ExitStack, closing
from dataclasses import dataclass
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

PROBES = REPO / "results/analysis/synthesis-probes.json"
STORE = REPO / "stores/train150.db"

# ---------------------------------------------------------------- grading

_NUMBER_WORDS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
]
_WORDS = {w: i for i, w in enumerate(_NUMBER_WORDS)}
_ABSTAIN = re.compile(
    r"\b(?:i (?:do not|don't) know|no (?:information|memor|record|evidence)|"
    r"not (?:enough|sure|specified|mentioned|available)|cannot (?:determine|tell|answer)|"
    r"unable to determine|isn't (?:any|enough)|there is no)\b",
    re.IGNORECASE,
)


def _all_ints(text: str) -> list[tuple[int, int]]:
    """Every integer in the reply with its position, digits and words alike."""
    out = []
    for m in re.finditer(r"(?<![\d.])(\d[\d,]*)(?![\d.])", text):
        out.append((m.start(), int(m.group(1).replace(",", ""))))
    for m in re.finditer(rf"\b({'|'.join(_WORDS)})\b", text, re.IGNORECASE):
        out.append((m.start(), _WORDS[m.group(1).lower()]))
    return sorted(out)


# A stated result, e.g. "Count: 3", "there are 3", "has 3 distinct", "= 3".
_STATED = re.compile(
    r"(?:count|total|answer|result)\s*(?:is|:|=)?\s*(\d[\d,]*)"
    r"|(?:there (?:are|were)|has|have|had)\s+(\d[\d,]*)"
    r"|=\s*(\d[\d,]*)",
    re.IGNORECASE,
)


def _stated_int(text: str) -> int | None:
    hits = [g for m in _STATED.finditer(text) for g in m.groups() if g]
    return int(hits[-1].replace(",", "")) if hits else None


def _int_for(kind: str, text: str) -> int | None:
    """The number the reply is actually asserting.

    Reading the *first* integer is wrong for both kinds: a count reply narrates its
    evidence first ("$200 spent last month", "February 26-27") and states the total
    last, and a duration reply prints the two dates before the subtraction. So a
    stated result wins, then a unit-attached number, then the last number; only a
    reply with no number at all is unparseable.
    """
    if kind == "duration":
        days = re.findall(r"(\d[\d,]*)\s*days?\b", text, re.IGNORECASE)
        if days:
            return int(days[-1].replace(",", ""))
    # "9 distinct possessions" is the assertion; a trailing "(November 4, 2023)"
    # is not. Counting nouns bind tighter than any other pattern here.
    counted = re.findall(
        r"(\d[\d,]*)\s+(?:distinct|different|unique|separate|total)\b", text, re.IGNORECASE
    )
    if counted:
        return int(counted[0].replace(",", ""))
    stated = _stated_int(text)
    if stated is not None:
        return stated
    ints = [n for _, n in _all_ints(text)]
    # A bare year is a date the reply happens to cite, not a count.
    plausible = [n for n in ints if not 1900 <= n <= 2100]
    return (plausible or ints or [None])[-1]


def _tokens(s: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split())


def _halves(question: str) -> tuple[str, str]:
    ai, bi = question.find("A: "), question.find("B: ")
    return question[ai + 3 : bi].strip(), question[bi + 3 :].strip()


_FIRST_MARKER = re.compile(
    r"\bhappened (?:first|before)\b|\bcame first\b|\bwas first\b|\bprecede[ds]?\b"
    r"|\boccurred (?:first|before)\b|\bearlier\b|\bbefore\b",
    re.IGNORECASE,
)


def _choice(question: str, text: str) -> str | None:
    """Which of the two events the reply puts first.

    The answerer replies in prose ("X ... happened before Y"), not with a letter, so
    grading on a stray `A`/`B` token scored 60% of these unparseable. An explicit
    label still wins when the reply opens with one; otherwise the text before the
    ordering marker is matched against each described event, scoring only on words
    distinctive to one of them so shared boilerplate cannot decide it.
    """
    # "B: ...", "B happened first", "Event B happened first" all name a choice.
    lead = re.match(r"\s*(?:Event\s+)?([AB])\s*(?:[:.)\-]|\b)", text)
    if lead and (
        lead.group(0).rstrip().endswith((":", ".", ")", "-"))
        or _FIRST_MARKER.match(text[lead.end() :].lstrip())
    ):
        return lead.group(1)
    a, b = _halves(question)
    m = _FIRST_MARKER.search(text)
    head = text[: m.start()] if m else text
    at, bt, ht = _tokens(a), _tokens(b), _tokens(head)
    sa = len((at - bt) & ht)
    sb = len((bt - at) & ht)
    if sa == sb:
        return None
    return "A" if sa > sb else "B"


@dataclass(slots=True)
class Grade:
    verdict: str  # correct | wrong | abstained | unparseable
    parsed: str | None
    detail: str = ""


def grade(probe: dict, text: str, status: str | None, stale: list[str]) -> Grade:
    kind, gold = probe["kind"], probe["answer"]
    declined = status == "no_evidence" or bool(_ABSTAIN.search(text))

    if kind in ("count", "duration"):
        got = _int_for(kind, text)
        if got is None:
            return Grade("abstained" if declined else "unparseable", None)
        # "0 distinct items" from a reply that also reported no evidence is a
        # refusal wearing a number, not a count of zero.
        if got == 0 and declined and gold != 0:
            return Grade("abstained", "0")
        # No second reading is accepted for a duration any more: the generator now
        # subtracts dates, which is exactly what the answerer is shown, so a gold it
        # cannot reach would be a generator bug and must not be graded away here.
        return Grade("correct" if got == gold else "wrong", str(got))

    if kind == "comparison":
        got = _choice(probe["question"], text)
        if got is None:
            return Grade("abstained" if declined else "unparseable", None)
        return Grade("correct" if got == gold else "wrong", got)

    # current_state: three-way, because answering with the superseded value is the
    # specific failure this probe exists to catch.
    gt, at = _tokens(gold), _tokens(text)
    if gold.lower() in text.lower() or (gt and len(gt & at) / len(gt) >= 0.6):
        return Grade("correct", "active")
    for old in stale:
        ot = _tokens(old)
        if old.lower() in text.lower() or (ot and len(ot & at) / len(ot) >= 0.6):
            return Grade("wrong", "superseded", detail=old)
    if declined:
        return Grade("abstained", None)
    return Grade("wrong", "other")


# ---------------------------------------------------------------- flags

_MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
_MONTHS = {m.lower(): i for i, m in enumerate(_MONTH_NAMES, 1)}
_MONTH_RE = "|".join(_MONTHS)


def _stated_months(text: str) -> set[tuple[int, int]]:
    out = set()
    for m in re.finditer(
        rf"\b({_MONTH_RE})\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", text, re.I
    ):
        out.add((int(m.group(2)), _MONTHS[m.group(1).lower()]))
    for m in re.finditer(rf"\b({_MONTH_RE})\s+(\d{{4}})\b", text, re.I):
        out.add((int(m.group(2)), _MONTHS[m.group(1).lower()]))
    for m in re.finditer(r"\b(\d{4})-(\d{2})-\d{2}\b", text):
        out.add((int(m.group(1)), int(m.group(2))))
    return out


def build_flags(probes: list[dict], store_path: Path | None = None) -> dict[str, dict]:
    """Per-probe validity flags, computed from the store with no model involved."""
    with closing(
        sqlite3.connect((store_path or STORE).resolve().as_uri() + "?mode=ro", uri=True)
    ) as con:
        rows = {
            i: (c or "", e or "", s or "", o or "")
            for i, c, e, s, o in con.execute(
                "select id, content, event_time, status, object from memories"
            )
        }
    flags = {}
    for p in probes:
        conflict = False
        stale = []
        for mid in p["evidence_memory_ids"]:
            content, et, status, obj = rows.get(mid, ("", "", "", ""))
            said = _stated_months(content) if et else set()
            if said and (int(et[:4]), int(et[5:7])) not in said:
                conflict = True
            if p["kind"] == "current_state" and status != "active" and obj:
                stale.append(obj)
        # A current_state probe only needs the *active* member of the chain; the
        # superseded ones are what the answerer must not use. Demanding the whole
        # chain scored every one of these `context_incomplete`, which is the flag
        # measuring the wrong thing rather than the context being short.
        required = [
            m
            for m in p["evidence_memory_ids"]
            if p["kind"] != "current_state" or rows.get(m, ("", "", "", ""))[2] == "active"
        ]
        flags[p["probe_id"]] = {
            "anchor_conflict": conflict,
            "stale_values": stale,
            "required_ids": required,
        }
    return flags


# ---------------------------------------------------------------- run


def _replay_retrieval(probes: list[dict]) -> dict[str, list[str]]:
    """Reproduce what the frozen variant retrieved, with no provider call.

    Retrieval is a pure function of the store and the question, so a run that did
    not record its retrieved ids can still have them recovered exactly.
    """
    from llm_long_term_memory.config import ExperimentConfig
    from llm_long_term_memory.embed import Encoder
    from llm_long_term_memory.retrieve.hybrid import HybridRetriever
    from llm_long_term_memory.store import NumpyFlatIndex, SQLiteMemoryStore

    cfg = ExperimentConfig.from_yaml("configs/v3-phase5-compact.yaml")
    store = SQLiteMemoryStore(STORE)
    store.initialize()
    index = NumpyFlatIndex("stores/train150-index", dim=cfg.models.embedding_dim)
    retriever = HybridRetriever(
        store,
        index,
        weights=cfg.retrieval.weights.model_dump(),
        candidate_limit=cfg.retrieval.candidate_limit,
        recency_halflife_days=cfg.retrieval.recency_halflife_days,
    )
    vectors = Encoder().encode([p["question"] for p in probes])
    return {
        p["probe_id"]: [
            h.memory.id
            for h in retriever.retrieve(
                v, p["question"], p["namespace"], temporal=True, limit=cfg.retrieval.top_k
            )
        ]
        for p, v in zip(probes, vectors, strict=True)
    }


def regrade(rows_path: Path, spec: dict, flags: dict) -> list[dict]:
    """Re-score saved replies without paying for them again.

    Grading rules are the part of this instrument most likely to be wrong, and the
    first version was: it read a count reply's first integer and looked for a bare
    `A`/`B` in prose. Model output is the expensive artifact, so verdicts are a
    derived column that can be rebuilt from it offline.
    """
    by_id = {p["probe_id"]: p for p in spec["probes"]}
    out = []
    missing_retrieval = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and "retrieved_ids" not in json.loads(line)
    ]
    replay = (
        _replay_retrieval([by_id[row["probe_id"]] for row in missing_retrieval])
        if missing_retrieval
        else {}
    )
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        probe = by_id[row["probe_id"]]
        g = grade(probe, row["text"], row["answer_status"], flags[row["probe_id"]]["stale_values"])
        row["verdict"], row["parsed"], row["detail"] = g.verdict, g.parsed, g.detail
        # Retrieval is deterministic given the store, so completeness can be
        # recomputed offline rather than re-run against the provider.
        got = set(row["retrieved_ids"] if "retrieved_ids" in row else replay[row["probe_id"]])
        need = set(flags[row["probe_id"]]["required_ids"])
        row["evidence_found"], row["evidence_needed"] = len(need & got), len(need)
        row["context_complete"] = need <= got
        out.append(row)
    return out


def summarise(rows: list[dict], spec: dict) -> dict:
    """Aggregates split on the two validity flags, never pooled into one number."""
    kinds = ("count", "duration", "comparison", "current_state")
    planned = Counter(p["kind"] for p in spec["probes"])

    def block(subset):
        res = {}
        for k in kinds:
            rs = [r for r in subset if r["kind"] == k]
            if not rs:
                continue
            c = Counter(r["verdict"] for r in rs)
            res[k] = {
                "n": len(rs),
                "correct": c["correct"],
                "wrong": c["wrong"],
                "abstained": c["abstained"],
                "unparseable": c["unparseable"],
                "accuracy": round(c["correct"] / len(rs), 4),
                "abstention_rate": round(c["abstained"] / len(rs), 4),
            }
        return res

    clean = [r for r in rows if r["context_complete"] and not r["anchor_conflict"]]
    return {
        "schema_version": 1,
        "not_a_benchmark": spec["not_a_benchmark"],
        "store_fingerprint": spec["store_fingerprint"],
        "variant": rows[0]["variant"] if rows else None,
        "coverage": {
            k: {"run": sum(1 for r in rows if r["kind"] == k), "planned": planned[k]} for k in kinds
        },
        "median_context_tokens": median(r["context_tokens"] for r in rows) if rows else None,
        "strata": {
            "interpretable": block(clean),
            "context_incomplete": block([r for r in rows if not r["context_complete"]]),
            "anchor_conflict": block([r for r in rows if r["anchor_conflict"]]),
            "all_rows_pooled": block(rows),
        },
        "current_state_wrong_kind": dict(
            Counter(
                r["parsed"]
                for r in rows
                if r["kind"] == "current_state" and r["verdict"] == "wrong"
            )
        ),
        "fallback_level": dict(Counter(r["fallback_level"] for r in rows)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="two_stage_reasoned_evidence")
    ap.add_argument("--config", default="configs/v3-phase5-compact.yaml")
    ap.add_argument("--store-name", default="train150")
    ap.add_argument("--label", default="v3.3")
    ap.add_argument("--out", default="results/raw")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="no API calls; canned replies")
    ap.add_argument("--regrade", action="store_true", help="re-score saved replies; no API calls")
    ap.add_argument(
        "--half",
        choices=("development", "held_out", "all"),
        default="development",
        help=(
            "which half of results/manifests/v4-probe-split.json to answer. Defaults to "
            "development: the held-out half is answered once, after every v4 choice is "
            "frozen, and 'all' would spend it on whatever run happened to be next."
        ),
    )
    args = ap.parse_args()

    from llm_long_term_memory.locking import AlreadyRunning, exclusive

    out = Path(args.out) / f"probes.{args.label}.jsonl"
    try:
        with ExitStack() as stack:
            stack.enter_context(exclusive(out, what="synthesis probe run"))
            return _run(args, stack)
    except (OSError, ValueError, KeyError, sqlite3.Error, AlreadyRunning) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2


def _run(args, stack: ExitStack) -> int:
    from probe_safety import (
        claim_holdout,
        complete_holdout,
        resume_ids,
        run_identity,
        select_probes,
    )

    from llm_long_term_memory.config import Settings

    probe_bytes = PROBES.read_bytes()
    spec = json.loads(probe_bytes)
    split_path = REPO / "results/manifests/v4-probe-split.json"
    chosen = select_probes(spec, probe_bytes, split_path, args.half)
    if args.limit < 0:
        raise ValueError("--limit must be nonnegative")
    if args.half == "all" and not (args.dry_run or args.regrade):
        raise ValueError("--half all is rehearsal-only; run development and held_out separately")
    probes = chosen[: args.limit or None]
    print(f"{len(probes)} probes selected ({args.half})", file=sys.stderr)
    identity = run_identity(REPO, Path(args.config), args.variant, probe_bytes, args.half)
    out = Path(args.out) / (
        f"probes.{args.label}.dry.jsonl" if args.dry_run else f"probes.{args.label}.jsonl"
    )
    # Validate row identity before client construction, including in a checkout with no key/store.
    if not (args.dry_run or args.regrade):
        resume_ids(out, chosen, identity)

    store_path = (Settings().store_dir / f"{args.store_name}.db").resolve()
    from synthesis_probes import _fingerprint

    with closing(sqlite3.connect(store_path.as_uri() + "?mode=ro", uri=True)) as con:
        actual = _fingerprint(con)
    if actual != spec["store_fingerprint"]:
        raise ValueError("store fingerprint mismatch: selected store differs from the probe store")
    flags = build_flags(chosen, store_path)

    if args.regrade:
        rows_path = Path(args.out) / f"probes.{args.label}.jsonl"
        rows = regrade(rows_path, spec, flags)
        rows_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        summary = summarise(rows, {**spec, "probes": chosen})
        dest = REPO / "results/analysis" / f"synthesis-probes.{args.label}.json"
        dest.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(summary["strata"]["interpretable"], indent=1))
        print(f"\nwrote {dest}")
        return 0

    from llm_long_term_memory.cli import _build
    from llm_long_term_memory.evaluation.datasets.longmemeval import Instance

    pending: list[str] = []

    from types import SimpleNamespace

    _cfg, _settings, runner, _judge, usage = _build(
        args.variant,
        args.config,
        store_name=args.store_name,
        client_override=SimpleNamespace() if args.dry_run else None,
    )
    fingerprint = runner.store_fingerprint
    stack.callback(runner.store.close)
    if not args.dry_run:
        stack.callback(usage.save, out.with_suffix(".usage.json"), merge=True)
    if args.dry_run:
        from llm_long_term_memory.llm.client import Completion

        # A canned reply in the *base* verdict shape rehearses nothing that v4 changed:
        # `compute` never runs, the derivation is never recorded, and the grader never
        # sees a computed answer. The rehearsal would pass while the paid run failed on
        # its first probe. So the reply is shaped for the policy under test.
        policy = getattr(runner, "answer_policy", "")
        v4 = policy.startswith("synthesis_v4")
        # v4.2 counts from cited labels, so a canned count without them rehearses the
        # free-text fallback instead of the mechanism under test — the same class of
        # miss this block already exists to prevent, one policy later. The labels have
        # to be ones the context really carries, which is why they are read off the
        # runner rather than hard-coded.
        cites = policy == "synthesis_v4_enumerate"
        canned_by_kind = {
            "count": {
                "operation": "count",
                "items": ["2 concert tickets", "a book", "a scarf"],
            },
            "duration": {
                "operation": "duration",
                "start_date": "2022-01-01",
                "end_date": "2022-01-19",
            },
            "comparison": {
                "operation": "comparison",
                "start_date": "2023-05-20",
                "end_date": "2023-03-10",
                "answer": "B happened first.",
            },
            "current_state": {"operation": "current_state", "answer": "the current value"},
        }

        def canned(**kw):
            payload = {"status": "answer", "answer": "3 days"}
            if v4:
                kind = pending[-1] if pending else ""
                payload |= canned_by_kind.get(kind, {"operation": "lookup"})
                if cites and kind == "count":
                    labels = sorted(getattr(runner, "_context_labels", set()))[:3]
                    payload["member_labels"] = labels
                    payload["items"] = payload["items"][: len(labels)]
            return Completion(
                text=json.dumps(payload),
                model="dry-run",
                input_tokens=0,
                output_tokens=0,
                thinking_tokens=0,
                attempts=1,
            )

        runner.client.generate = canned  # type: ignore[method-assign]

    out.parent.mkdir(parents=True, exist_ok=True)
    done = set() if args.dry_run else resume_ids(out, chosen, identity)
    if args.dry_run:
        out.write_text("", encoding="utf-8")
    ledger_path = REPO / "results/manifests/v4-probe-heldout-run.json"
    ledger = None
    if args.half == "held_out" and not args.dry_run:
        ledger = claim_holdout(ledger_path, out, identity)

    todo = [p for p in probes if p["probe_id"] not in done]
    print(f"{len(done)} already done, {len(todo)} to run against {fingerprint}")

    with out.open("a", encoding="utf-8") as fh:
        for i, p in enumerate(todo, 1):
            # Tells the dry-run which shape of verdict to fake for this probe; a real
            # run never reads it.
            pending.append(p["kind"])
            inst = Instance(
                question_id=p["namespace"],
                question_type=p["kind"],
                question=p["question"],
                answer=str(p["answer"]),
                question_date=spec.get("question_date", "2026-09-06"),
                sessions=[],
                answer_session_ids=[],
            )
            ans = runner.answer(inst)
            f = flags[p["probe_id"]]
            got = set(ans.retrieved_ids)
            need = set(f["required_ids"])
            g = grade(p, ans.text, ans.notes.get("answer_status"), f["stale_values"])
            row = {
                "run_identity": identity,
                "probe_id": p["probe_id"],
                "kind": p["kind"],
                "namespace": p["namespace"],
                "question": p["question"],
                "gold": p["answer"],
                "text": ans.text,
                "verdict": g.verdict,
                "parsed": g.parsed,
                "detail": g.detail,
                "answer_status": ans.notes.get("answer_status"),
                "reasoning_kind": ans.notes.get("reasoning_kind"),
                # v4 records what the code computed and from which operands. Without
                # these two the answer "3 distinct: ..." cannot be told apart from the
                # model having said three, which is the whole claim v4 is making.
                # Absent for v2/v3, which compute nothing.
                "synthesis_operation": ans.notes.get("synthesis_operation"),
                "synthesis_computation": ans.notes.get("synthesis_computation"),
                "synthesis_missing_field": ans.notes.get("synthesis_missing_field"),
                "answer_was_raw_structure": ans.notes.get("answer_was_raw_structure"),
                "missing_field_was_narration": ans.notes.get("missing_field_was_narration"),
                "scan_route": ans.notes.get("scan_route"),
                "fallback_level": ans.notes.get("fallback_level"),
                "context_tokens": ans.context_tokens,
                "evidence_found": len(need & got),
                "evidence_needed": len(need),
                "context_complete": need <= got,
                "anchor_conflict": f["anchor_conflict"],
                "retrieved_ids": list(ans.retrieved_ids),
                "store_fingerprint": fingerprint,
                "variant": args.variant,
            }
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            done.add(p["probe_id"])
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  {p['probe_id']} -> {g.verdict}", flush=True)

    if ledger is not None and {p["probe_id"] for p in chosen} <= done:
        complete_holdout(ledger_path, ledger)
    print(f"\nwrote {out}")
    if not args.dry_run:
        print(json.dumps(usage.snapshot() if hasattr(usage, "snapshot") else {}, indent=1)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
