# Superseded probe set v1, and the v3.3 reading taken on it

Kept because the reading is real and its defects are the reason the set was
regenerated. It must not be compared with any later probe set: the probe ids,
the questions and the ground truth all changed.

What was wrong with it, all measured without a provider call:

- **`count` did not name its scope.** Ground truth was a `COUNT` over one mapped
  `relation_type`; the question was natural language, and the store splits one
  concept across several predicates. The answerer counted neighbouring predicates
  and was marked wrong for it.
- **43% of `count` probes were unanswerable from the retrieved context** — the
  top-20 context did not hold every item the truth counted.
- **22% of `duration`/`comparison` probes were anchored on an `event_time` that
  contradicts a date written in the same memory's text.**

The v3.3 rows here were graded by the final grader in
`tools/run_synthesis_probes.py`. See
[`../../analysis/synthesis-probes-v3.3.md`](../../analysis/synthesis-probes-v3.3.md).
