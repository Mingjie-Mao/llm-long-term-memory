from chronomem.evaluation.budget import (
    BudgetPoint,
    pareto_frontier,
    render_pareto_svg,
    sweep_artifact_stem,
)


def point(budget: int, accuracy: float, tokens: float, completed: bool = True) -> BudgetPoint:
    return BudgetPoint(budget, accuracy, tokens, 100.0, 50, completed)


def test_pareto_frontier_drops_points_worse_on_both_axes():
    points = [point(1_000, 0.40, 900), point(2_000, 0.50, 1_800), point(4_000, 0.45, 2_400)]
    assert pareto_frontier(points) == points[:2]


def test_pareto_frontier_excludes_incomplete_runs():
    assert pareto_frontier([point(1_000, 0.50, 900, completed=False)]) == []


def test_svg_labels_budgets_and_marks_the_frontier():
    svg = render_pareto_svg([point(1_000, 0.40, 900), point(2_000, 0.50, 1_800)], "P6")
    assert "<svg" in svg
    assert 'class="frontier"' in svg
    assert ">1000<" in svg


def test_sweep_artifact_stems_are_distinct_by_selector():
    assert sweep_artifact_stem("chronomem", "relevance") == "pack-sweep-chronomem-relevance"
    assert sweep_artifact_stem("chronomem", "utility") == "pack-sweep-chronomem-utility"
