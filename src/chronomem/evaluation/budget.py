"""Summaries and a dependency-free Pareto chart for P6 budget sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape


@dataclass(frozen=True, slots=True)
class BudgetPoint:
    budget: int
    accuracy: float
    median_context_tokens: float
    p95_latency_ms: float
    n: int
    completed: bool


def sweep_artifact_stem(variant: str, selector: str) -> str:
    """Keep relevance and utility sweeps from overwriting each other."""
    return f"pack-sweep-{variant}-{selector}"


def pareto_frontier(points: list[BudgetPoint]) -> list[BudgetPoint]:
    """Keep points not dominated on accuracy (higher) and context (lower)."""
    frontier = []
    for point in points:
        dominated = any(
            other is not point
            and other.accuracy >= point.accuracy
            and other.median_context_tokens <= point.median_context_tokens
            and (
                other.accuracy > point.accuracy
                or other.median_context_tokens < point.median_context_tokens
            )
            for other in points
            if other.completed
        )
        if point.completed and not dominated:
            frontier.append(point)
    return sorted(frontier, key=lambda point: point.median_context_tokens)


def render_pareto_svg(points: list[BudgetPoint], title: str) -> str:
    """Render accuracy against actual median context tokens without plotting deps."""
    width, height = 760, 440
    left, right, top, bottom = 70, 30, 50, 60
    completed = [point for point in points if point.completed]
    if not completed:
        completed = points
    if not completed:
        return ""

    x_values = [point.median_context_tokens for point in completed]
    y_values = [point.accuracy for point in completed]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(0.0, min(y_values)), max(1.0, max(y_values))
    if x_min == x_max:
        x_min = max(0.0, x_min - 1.0)
        x_max += 1.0

    def x(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (width - left - right)

    def y(value: float) -> float:
        return height - bottom - (value - y_min) / (y_max - y_min) * (height - top - bottom)

    frontier = pareto_frontier(points)
    frontier_points = " ".join(
        f"{x(point.median_context_tokens):.1f},{y(point.accuracy):.1f}" for point in frontier
    )
    bottom_y = height - bottom
    svg_open = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    style = (
        "<style>text{font:13px sans-serif;fill:#1f2937}.axis{stroke:#6b7280}"
        ".frontier{fill:none;stroke:#0f766e;stroke-width:2}.point{fill:#2563eb}"
        ".partial{fill:#9ca3af}</style>"
    )
    parts = [
        svg_open,
        style,
        f'<text x="{left}" y="25" font-size="18">{escape(title)}</text>',
        f'<line class="axis" x1="{left}" y1="{bottom_y}" x2="{width - right}" y2="{bottom_y}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}"/>',
        f'<text x="{width / 2:.1f}" y="{height - 16}" text-anchor="middle">'
        "Median context tokens</text>",
        f'<text x="18" y="{height / 2:.1f}" '
        f'transform="rotate(-90 18 {height / 2:.1f})" text-anchor="middle">Accuracy</text>',
        f'<text x="{left}" y="{height - bottom + 22}">{x_min:.0f}</text>',
        f'<text x="{width - right}" y="{height - bottom + 22}" '
        f'text-anchor="end">{x_max:.0f}</text>',
        f'<text x="{left - 10}" y="{y(y_min) + 4:.1f}" text-anchor="end">{y_min:.0%}</text>',
        f'<text x="{left - 10}" y="{y(y_max) + 4:.1f}" text-anchor="end">{y_max:.0%}</text>',
    ]
    if frontier_points:
        parts.append(f'<polyline class="frontier" points="{frontier_points}"/>')
    for point in points:
        parts.append(
            f'<circle class="{"point" if point.completed else "partial"}" '
            f'cx="{x(point.median_context_tokens):.1f}" cy="{y(point.accuracy):.1f}" r="5"/>'
        )
        parts.append(
            f'<text x="{x(point.median_context_tokens) + 8:.1f}" '
            f'y="{y(point.accuracy) - 8:.1f}">{point.budget}</text>'
        )
    return "".join(parts) + "</svg>\n"
