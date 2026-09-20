"""The extraction-fidelity command.

Scores how much of the user's own specifics survive extraction, against the source
text rather than gold answers, and caches each extraction so re-scoring a changed
metric costs nothing.
"""

from __future__ import annotations

import typer
from rich.table import Table

from llm_long_term_memory.commands.common import console
from llm_long_term_memory.config import Settings
from llm_long_term_memory.evaluation.datasets import longmemeval as lme


def _grounded_memory(anchored, session_id: str):
    """One anchored fact as a `Memory`, so both arms are scored by the same ruler.

    `score_sessions` reads `content`, so nothing else has to be faithful — but the typed
    fields are carried anyway, because a comparison that quietly scored a different
    object than the one being proposed would not be a comparison.
    """
    from llm_long_term_memory.store import Memory

    fact = anchored.fact
    return Memory(
        id=f"g_{abs(hash((session_id, fact.verbatim_span))) & 0xFFFFFFFFFF:010x}",
        user_id="fidelity",
        type="semantic",
        content=fact.content,
        token_count=len(fact.content.split()),
        subject=fact.subject,
        predicate=fact.attribute or None,
        object=(f"{fact.value} {fact.unit}".strip() or None) if fact.value else None,
        source_session_id=session_id,
        source_turn_index=anchored.turn_index,
        source_char_start=anchored.char_start,
        source_char_end=anchored.char_end,
    )


def ingest_fidelity(
    config: str = typer.Option("configs/baselines.yaml", "--config", "-c"),
    sessions: int = typer.Option(30, help="How many sessions to extract and score"),
    show_misses: int = typer.Option(5, help="Missed values to print per facet"),
    holdout: bool = typer.Option(
        True, help="Score sessions from the held-out split, never the dev questions"
    ),
    batch: int | None = typer.Option(None, help="Override sessions per request"),
    two_stage: bool | None = typer.Option(None, help="Override the two-stage flag"),
    grounded: bool = typer.Option(
        False,
        "--grounded",
        help=(
            "Score the evidence-grounded extractor instead: atomic facts each quoting "
            "the turn they came from, typed value/unit, event time separate from "
            "observation time. Refused facts are counted, not silently dropped. It "
            "extracts one session per request, so compare it against `--batch 1` or the "
            "batch size is a second variable."
        ),
    ),
    min_score: float | None = typer.Option(
        None,
        help="Exit non-zero if overall fidelity falls below this (0-1). Turns the "
        "report into a gate for unattended runs.",
    ),
) -> None:
    """What fraction of the user's own specifics survive extraction?

    Needs no gold answers, so it cannot be fitted to the evaluation set — the
    reference is the source text. Run it before spending an ingest on a prompt
    change.
    """
    from llm_long_term_memory.config import ExperimentConfig
    from llm_long_term_memory.ingest import Extractor, TwoStageExtractor
    from llm_long_term_memory.ingest.fidelity import score_sessions
    from llm_long_term_memory.llm import Limits, QuotaManager, UsageTracker
    from llm_long_term_memory.llm.client import GeminiClient
    from llm_long_term_memory.store import Memory

    settings = Settings()
    cfg = ExperimentConfig.from_yaml(config)
    quota = QuotaManager(
        state_dir=settings.store_dir / "quota",
        default=Limits(rpm=cfg.quota.rpm, tpm=cfg.quota.tpm, rpd=cfg.quota.rpd),
    )
    quota.load_learned()
    usage = UsageTracker()
    client = GeminiClient(settings.require_api_key(), quota=quota, usage=usage)
    use_two_stage = cfg.ingest.two_stage if two_stage is None else two_stage
    if grounded:
        from llm_long_term_memory.ingest.grounded import GroundedExtractor

        extractor = GroundedExtractor(client, cfg.models.extractor)
    else:
        extractor = (
            TwoStageExtractor(client, cfg.models.extractor)
            if use_two_stage
            else Extractor(client, cfg.models.extractor)
        )

    with console.status("Loading corpus…"):
        every = lme.load(cfg.dataset_variant, settings.data_dir)
        dev, test = lme.split_dev_test(every)
        pool = test if holdout else dev
        picked: list = []
        for inst in pool:
            for sess in inst.sessions:
                picked.append(sess)
                if len(picked) >= sessions:
                    break
            if len(picked) >= sessions:
                break

    # One session per request when grounded: `turn_index` has to name a turn the model
    # can see, and a fifteen-session prompt makes that ambiguous. Forced rather than
    # defaulted, and stated in the header, because batch size alone moved yield about
    # five-fold (D5) — comparing it against the shipped batch of 15 would be two
    # variables wearing one number.
    batch = 1 if grounded else (batch or cfg.ingest.sessions_per_request)
    console.print(
        f"[bold]fidelity[/bold] · {len(picked)} sessions from the "
        f"{'held-out' if holdout else 'dev'} split · {batch}/request · "
        f"{'two-stage' if use_two_stage else 'single-stage'} · {cfg.models.extractor}\n"
    )

    # Cache the extraction so a change to the *metric* can be re-scored for free.
    # The same sessions were re-extracted four times while the denominator was being
    # corrected; each pass cost requests and returned identical memories.
    import hashlib
    import json as _json

    from llm_long_term_memory.ingest.extract import _PROMPT, EXTRACT_SYSTEM
    from llm_long_term_memory.ingest.extract_facts import _PROMPT as FACTS_PROMPT
    from llm_long_term_memory.ingest.extract_facts import FACTS_SYSTEM
    from llm_long_term_memory.ingest.keying import _PROMPT as KEYING_PROMPT
    from llm_long_term_memory.ingest.keying import KEYING_SYSTEM

    if grounded:
        active_prompt = "".join(extractor.prompt_texts())
    elif use_two_stage:
        active_prompt = FACTS_SYSTEM + FACTS_PROMPT + KEYING_SYSTEM + KEYING_PROMPT
    else:
        active_prompt = EXTRACT_SYSTEM + _PROMPT
    fingerprint = hashlib.sha1(
        (
            active_prompt + cfg.models.extractor + str(batch) + str(use_two_stage) + str(grounded)
        ).encode()
    ).hexdigest()[:12]
    cache_path = settings.store_dir / "fidelity-cache" / f"{fingerprint}.json"
    cached = _json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    if cached:
        console.print(
            f"[dim]reusing cached extraction {fingerprint} ({len(cached)} sessions)[/dim]"
        )

    from types import SimpleNamespace

    refusals = attempted = 0
    pairs = []
    for i in range(0, len(picked), batch):
        chunk = picked[i : i + batch]
        if all(s.session_id in cached for s in chunk):
            for s in chunk:
                mems = [
                    Memory(**{**d, "event_time": None, "valid_from": None, "valid_to": None})
                    for d in cached[s.session_id]
                ]
                pairs.append((s, mems))
            continue
        by_session: dict[str, list] = {s.session_id: [] for s in chunk}
        if grounded:
            written, refused = 0, 0
            for session in chunk:
                report = extractor.extract(
                    [(t.role, t.content) for t in session.turns], session.date
                )
                by_session[session.session_id] = [
                    _grounded_memory(anchored, session.session_id) for anchored in report.anchored
                ]
                written += len(report.anchored)
                refused += len(report.rejected)
            outcome = SimpleNamespace(memories=[m for v in by_session.values() for m in v])
            refusals += refused
            attempted += written + refused
        else:
            outcome = extractor.extract(chunk)
            for m in outcome.memories:
                if m.source_session_id in by_session:
                    by_session[m.source_session_id].append(m)
        pairs.extend((s, by_session[s.session_id]) for s in chunk)
        for s in chunk:
            cached[s.session_id] = [
                {
                    "id": m.id,
                    "user_id": m.user_id,
                    "type": m.type,
                    "content": m.content,
                    "token_count": m.token_count,
                    "subject": m.subject,
                    "predicate": m.predicate,
                    "object": m.object,
                }
                for m in by_session[s.session_id]
            ]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(_json.dumps(cached), encoding="utf-8")
        console.print(f"  batch {i // batch + 1}: +{len(outcome.memories)} memories")

    report = score_sessions(pairs)

    t = Table(title="extraction fidelity", show_header=True)
    t.add_column("facet", style="cyan")
    t.add_column("stated", justify="right")
    t.add_column("retained", justify="right")
    t.add_column("recall", justify="right")
    for facet, score in sorted(report.per_facet.items(), key=lambda kv: kv[1].recall):
        style = "red" if score.recall < 0.4 else ("yellow" if score.recall < 0.7 else "green")
        t.add_row(
            facet, str(score.stated), str(score.retained), f"[{style}]{score.recall:.1%}[/{style}]"
        )
    console.print(t)
    console.print(
        f"[bold]overall {report.overall:.1%}[/bold] · "
        f"{report.memories_per_session:.1f} memories/session · {report.memories} memories"
    )
    if grounded and attempted:
        console.print(
            f"[bold]grounding[/bold] · {attempted - refusals}/{attempted} facts cited a span "
            f"that was in the turn they named ({1 - refusals / attempted:.1%}); "
            f"{refusals} refused rather than stored"
        )

    for facet, misses in report.missed_examples.items():
        console.print(
            f"\n[red]dropped {facet}[/red]: " + ", ".join(repr(m) for m in misses[:show_misses])
        )

    usage.save(settings.results_dir / "raw" / "fidelity.usage.json")
    artifact = settings.results_dir / "raw" / "fidelity.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        _json.dumps(
            {
                "overall": report.overall,
                "memories": report.memories,
                "memories_per_session": report.memories_per_session,
                "sessions": sessions,
                "holdout": holdout,
                "per_facet": {f: s.recall for f, s in report.per_facet.items()},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    console.print(f"[dim]→ {artifact}[/dim]")

    # Without --min-score this stays a report, which is what it is when a human is
    # reading it. Automation needs a verdict it cannot misread as success: printing
    # a bad number and exiting 0 is how a closed gate gets walked through.
    if min_score is not None and report.overall < min_score:
        console.print(
            f"\n[red]Fidelity gate closed.[/red] "
            f"overall {report.overall:.1%} < required {min_score:.1%}"
        )
        raise typer.Exit(1)
    if min_score is not None:
        console.print(
            f"\n[green]Fidelity gate open.[/green] overall {report.overall:.1%} >= {min_score:.1%}"
        )
