"""Choosing what goes in the context window.

The packing decision is a knapsack: each memory has a token cost and a value, and
the budget is fixed. Two things make it more than sorting.

**Value per token, not value.** A memory worth 0.9 that costs 60 tokens is a worse
buy than three worth 0.4 costing 12 each. Ranking by score alone systematically
prefers long memories, which is the opposite of what a token budget wants.

**Redundancy is only visible between items.** Two memories carrying the same fact
each look valuable in isolation; taking both spends the budget twice for one fact.
So selection is greedy with a penalty applied against what has *already* been
chosen, rather than a single sort.

Type floors exist because ranking is not the only consideration: without a reserved
slice, a flood of high-scoring episodic memories crowds out the profile facts that
almost every question needs a little of.

**Which field the floors key on is a choice, and the default one barely discriminates.**
`Memory.type` is assigned by a handful of verb regexes, and on a real store 89.8% of
rows fall through to the `semantic` default — so a floor on `semantic` reserves a slice
for almost everything and a floor on `profile` reserves it for 1.4%. `Memory.scope` is
assigned by the extractor from the sentence and spreads across seven values, so floors
keyed on it reserve what they read as reserving. `floor_field` selects; the default
stays `type` so that no committed configuration changes meaning underneath itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_long_term_memory.store import Memory


def _tokens(text: str) -> set[str]:
    return {
        w for w in "".join(c if c.isalnum() else " " for c in text.lower()).split() if len(w) > 2
    }


@dataclass(slots=True)
class PackResult:
    selected: list[Memory] = field(default_factory=list)
    tokens_used: int = 0
    budget: int = 0
    considered: int = 0
    dropped_for_budget: int = 0
    dropped_for_redundancy: int = 0
    dropped_negative: int = 0
    """Memories predicted to make the answer worse. Excluded even with budget to
    spare — the harmful case is why utility is signed rather than a ranking."""

    @property
    def utilisation(self) -> float:
        return self.tokens_used / self.budget if self.budget else 0.0


def pack(
    memories: list[Memory],
    utilities: list[float],
    budget: int,
    *,
    type_floors: dict[str, float] | None = None,
    floor_field: str = "type",
    redundancy_penalty: float = 0.7,
    redundancy_threshold: float = 0.6,
) -> PackResult:
    """Greedy selection by utility density under a token budget.

    Greedy rather than exact: the optimal 0/1 knapsack is available at this size,
    but the values are *predictions* whose error dwarfs the gap between greedy and
    optimal. Solving the wrong objective more precisely would buy nothing.
    """
    # Validated before the empty-input shortcut, or a misspelled field is accepted
    # whenever the candidate list happens to be empty and refused only later, under load.
    if floor_field not in ("type", "scope"):
        raise ValueError(f"floors key on 'type' or 'scope', not {floor_field!r}")
    result = PackResult(budget=budget, considered=len(memories))
    if not memories or budget <= 0:
        return result

    # Floors are *reserved*, not merely permitted. An earlier version widened the
    # allowance for an under-spent type instead of holding room back, which does
    # nothing: the general pool is exhausted by higher-density items first and the
    # loop ends before the floored type is ever reached.
    floors = type_floors or {}
    reserve = {t: int(budget * share) for t, share in floors.items()}
    general = budget - sum(reserve.values())

    candidates = [
        (m, u) for m, u in zip(memories, utilities, strict=True) if m.token_count <= budget
    ]
    result.dropped_negative = sum(1 for _, u in candidates if u <= 0)
    remaining = [(m, u) for m, u in candidates if u > 0]

    chosen: list[Memory] = []
    chosen_tokens: list[set[str]] = []
    used = 0

    def affordable(memory: Memory) -> bool:
        bucket = getattr(memory, floor_field, None) or ""
        from_reserve = min(reserve.get(bucket, 0), memory.token_count)
        return memory.token_count - from_reserve <= general

    while remaining:
        best, best_density, best_value = None, float("-inf"), 0.0
        for memory, utility in remaining:
            value = utility
            if chosen_tokens:
                words = _tokens(memory.content)
                overlap = (
                    max(len(words & seen) / len(words) for seen in chosen_tokens) if words else 0.0
                )
                if overlap >= redundancy_threshold:
                    value *= 1.0 - redundancy_penalty
            density = value / max(1, memory.token_count)
            if density > best_density:
                best, best_density, best_value = (memory, utility), density, value

        memory, _ = best
        remaining.remove(best)

        if best_value <= 0:
            result.dropped_for_redundancy += 1
            continue
        if not affordable(memory):
            result.dropped_for_budget += 1
            continue

        bucket = getattr(memory, floor_field, None) or ""
        from_reserve = min(reserve.get(bucket, 0), memory.token_count)
        if from_reserve:
            reserve[bucket] -= from_reserve
        general -= memory.token_count - from_reserve

        chosen.append(memory)
        chosen_tokens.append(_tokens(memory.content))
        used += memory.token_count

    result.selected = chosen
    result.tokens_used = used
    return result
