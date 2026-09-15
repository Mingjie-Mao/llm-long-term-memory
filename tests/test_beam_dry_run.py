"""The rehearsal itself, which is only worth running if its canned replies are shaped
for the policy under test.

A dry run that answers in the wrong shape passes and lets the paid run fail on its first
request — the defect the report's 花配额之前 section step 3 exists for, and one this project
has already committed once (a base-shaped verdict rehearsing a v4 policy).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from llm_long_term_memory.evaluation.beam_judge import (
    InvalidJudgement,
    RubricVerdict,
    checked_grades,
    render_prompt,
)
from llm_long_term_memory.ingest.extract_facts import FactsResult, render_batch
from llm_long_term_memory.ingest.keying import KeyingResult
from llm_long_term_memory.ingest.schemas import DedupDecision

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

dry = pytest.importorskip("beam_dry_run")

from llm_long_term_memory.evaluation.datasets.longmemeval import (  # noqa: E402
    HaystackSession,
    HaystackTurn,
)


def sessions():
    return [
        HaystackSession(
            session_id="s00c000",
            date="2024/03/15 (Fri) 09:00",
            turns=[
                HaystackTurn(
                    role="user",
                    content="I finally counted my postcard collection last weekend. "
                    "There are twenty-five of them now, up from twelve in December.",
                ),
                HaystackTurn(role="assistant", content="That is a big jump."),
            ],
        )
    ]


def test_stage_a_returns_facts_the_extractor_can_parse_under_the_right_session():
    fake = dry.FakeProvider()
    _, text = fake._reply("extractor", render_batch(sessions()), FactsResult)
    result = FactsResult.model_validate_json(text)
    assert [group.session_index for group in result.sessions] == [0]
    assert result.sessions[0].facts
    assert "postcard" in " ".join(fact.content for fact in result.sessions[0].facts)


def test_stage_b_keys_every_fact_it_was_given_by_index():
    fake = dry.FakeProvider()
    prompt = "## Facts\n\n0. The user owns 25 postcards.\n1. The user restarted in December."
    _, text = fake._reply("extractor", prompt, KeyingResult)
    result = KeyingResult.model_validate_json(text)
    assert sorted(item.index for item in result.facts) == [0, 1]


def test_deduplication_is_rehearsed_in_all_three_verdicts():
    fake = dry.FakeProvider()

    def verdict(old, new):
        prompt = f"Existing memory: {old}\nNew candidate:   {new}\n"
        _, text = fake._reply("extractor", prompt, DedupDecision)
        return DedupDecision.model_validate_json(text).verdict

    same = "The user owns 25 postcards."
    assert verdict(same, same) == "DUPLICATE"
    assert verdict(f"{same[:-1]} now.", f"{same[:-1]} today.") == "UPDATE"
    assert verdict(same, "The user's sister lives in Osaka.") == "DISTINCT"


def test_the_answerer_asks_for_raw_source_sometimes_and_only_in_the_structured_pass():
    """A canned reply that always answers never enters the fallback, and the fallback's
    second call is the whole difference between one quota day and two."""
    fake = dry.FakeProvider(need_source_percent=100)
    _, structured = fake._reply("answerer", "context\n\nQuestion: what did I buy?", object)
    assert json.loads(structured)["status"] == "need_source"
    kind, prose = fake._reply("answerer", "evidence\n\nQuestion: what did I buy?", None)
    assert kind == "answer_from_raw_source"
    with pytest.raises(json.JSONDecodeError):
        json.loads(prose)


def test_the_judge_grades_the_rubric_exactly_once_and_stays_on_the_scale():
    fake = dry.FakeProvider()
    rubric = ("mentions twenty-five postcards", "mentions December", "mentions Kyoto")
    prompt = render_prompt("how many postcards?", rubric, "twenty-five postcards", ordering=False)
    _, text = fake._reply("judge", prompt, RubricVerdict)
    grades = checked_grades(RubricVerdict.model_validate_json(text), len(rubric))
    assert [grade["item"] for grade in grades] == [1, 2, 3]
    assert grades[0]["score"] > grades[2]["score"]


def test_event_ordering_gets_positions_and_nothing_else_does():
    fake = dry.FakeProvider()
    rubric = ("skate purchase and budget", "workshop planning with local artists")
    answer = "skate purchase and budget, then workshop planning with local artists"
    ordered = RubricVerdict.model_validate_json(
        fake._reply("judge", render_prompt("order?", rubric, answer, ordering=True), RubricVerdict)[
            1
        ]
    )
    assert [grade.position for grade in ordered.grades] == [1, 2]
    plain = RubricVerdict.model_validate_json(
        fake._reply(
            "judge", render_prompt("order?", rubric, answer, ordering=False), RubricVerdict
        )[1]
    )
    assert all(grade.position is None for grade in plain.grades)


def test_an_unshaped_call_is_refused_rather_than_answered_with_something():
    """A reply invented for a call the rehearsal does not understand rehearses nothing
    and hides the fact that it did not."""
    with pytest.raises(NotImplementedError):
        dry.FakeProvider()._reply("summariser", "anything", None)


def test_the_off_rubric_grade_is_refused_whatever_the_rubric_length():
    """The drill used to send one grade too few, which a one-item rubric accepts — the
    median rubric length for five of the ten abilities."""
    broken = dry.Interrupted(mode="judge", fail_at=0)
    completion = broken.generate(role="judge", model="m", prompt="x", schema=RubricVerdict)
    verdict = RubricVerdict.model_validate_json(completion.text)
    for length in (1, 2, 5):
        with pytest.raises(InvalidJudgement):
            checked_grades(verdict, length)


def test_an_interruption_fires_once_so_the_resume_can_finish():
    broken = dry.Interrupted(mode="quota", fail_at=1)
    broken.generate(role="answerer", model="m", prompt="a\n\nQuestion: q", schema=object)
    from llm_long_term_memory.llm.client import DailyQuotaExhausted

    with pytest.raises(DailyQuotaExhausted):
        broken.generate(role="answerer", model="m", prompt="a\n\nQuestion: q", schema=object)
    assert broken.fired
    broken.generate(role="answerer", model="m", prompt="a\n\nQuestion: q", schema=object)


def test_the_rehearsal_never_writes_where_a_real_run_resumes_from():
    source = (REPO / "tools/beam_dry_run.py").read_text(encoding="utf-8")
    assert 'os.environ["LLTM_RESULTS_DIR"] = str(work / "results")' in source
    assert 'os.environ["LLTM_STORE_DIR"] = str(work / "stores")' in source
    assert 'parser.add_argument("--work", default="stores/beam-dry-run")' in source


def test_the_rehearsal_reads_only_the_development_half():
    """Exporting the final half needs --final-run; rehearsing on it would spend the last
    unseen set this project has."""
    source = (REPO / "tools/beam_dry_run.py").read_text(encoding="utf-8")
    assert 'HALF = "dev"' in source
    assert '"test"' not in source.split("def main(")[-1]
