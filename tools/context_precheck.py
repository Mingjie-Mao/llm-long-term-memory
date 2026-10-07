"""Pre-answer context size per arm, measured with a stub client: no provider calls.

    python tools/context_precheck.py train150 train150.json
    python tools/context_precheck.py dev100 dev100.json \
        two_stage_raw_primary two_stage_raw_primary_v2

Builds each arm exactly as `lltm eval run` does (`cli._build`, `configs/fallback.yaml`),
but hands it a client that answers every call with a fixed verdict of "answer". So the
recorded `context_tokens` is the context the real answerer would receive on its first
call; the conditional fallback, which the stub never asks for, adds text on the questions
where the real one does.

Validated on heldout100 before use: estimated 1,458 / 5,587 median tokens for
`two_stage_hydrated` / `two_stage_raw_primary`, against measured 1,502 / 5,598.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

ARMS = ("two_stage_hydrated", "two_stage_raw_primary")


class StubClient:
    """Answers every call with a fixed verdict; counts calls, sends nothing."""

    calls = 0

    def generate(self, *, role, model, prompt, **kw):
        StubClient.calls += 1

        class Completion:
            text = json.dumps({"status": "answer", "answer": "x", "reason": ""})
            input_tokens = output_tokens = thinking_tokens = 0
            api_latency_ms = 0.0

        return Completion()


def main() -> int:
    from llm_long_term_memory import cli
    from llm_long_term_memory.evaluation.datasets import longmemeval as lme

    store, manifest = sys.argv[1], sys.argv[2]
    arms = tuple(sys.argv[3:]) or ARMS
    ids = json.loads((REPO / "results" / "manifests" / manifest).read_text(encoding="utf-8"))[
        "question_ids"
    ]
    instances = {i.question_id: i for i in lme.load("s", REPO / "data")}
    for variant in arms:
        _, _, runner, _, _ = cli._build(
            variant,
            "configs/fallback.yaml",
            store,
            client_override=StubClient(),
            read_only_store=True,
        )
        sizes = sorted(runner.answer(instances[q]).context_tokens for q in ids)
        print(
            f"{variant:24s} median {statistics.median(sizes):,.0f}  "
            f"p90 {sizes[int(0.9 * len(sizes))]:,}  max {sizes[-1]:,}"
        )
    print(f"stub calls: {StubClient.calls} (no provider calls)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
