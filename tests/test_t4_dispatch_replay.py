import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from t4_dispatch_replay import GAP_GOLD, decide


def rows():
    dev = [
        {"question_id": f"q{i % 100}", "rep": i // 100 + 1, "matched": True, "context_tokens": 6300}
        for i in range(300)
    ]
    train = [
        {
            "question_id": f"supported{i}",
            "matched": True,
            "historical_correct": True,
            "context_tokens": 6300,
        }
        for i in range(18)
    ]
    train += [
        {
            "question_id": q,
            "matched": True,
            "historical_correct": False,
            "historical_faithful_gap": True,
            "context_tokens": 6300,
        }
        for q in GAP_GOLD
    ]
    return dev, train


def test_replay_requires_all_rows_and_known_cases_but_retains_false_original_gap_grades():
    dev, train = rows()
    assert decide(dev, train)["pass"] is True
    assert decide(dev, train)["faithful_gaps"]["correct"] == 3
    assert decide(dev[:-1], train)["pass"] is False
    dev[0]["matched"] = False
    assert decide(dev, train)["pass"] is False


def test_changed_original_supported_grade_or_gap_grade_blocks_acceptance():
    dev, train = rows()
    train[0]["historical_correct"] = False
    assert decide(dev, train)["pass"] is False
    dev, train = rows()
    train[-1]["historical_faithful_gap"] = False
    assert decide(dev, train)["pass"] is False


def test_duplicate_questions_cannot_fill_required_replay_counts():
    dev, train = rows()
    dev[-1] = dev[-2].copy()
    assert decide(dev, train)["pass"] is False
    dev, train = rows()
    train[1] = train[0].copy()
    assert decide(dev, train)["pass"] is False
