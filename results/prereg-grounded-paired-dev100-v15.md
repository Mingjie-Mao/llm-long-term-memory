# Paired dev100 v15 — retain reader outputs before original grading

2026-10-03 before any paired outputs/calls. Conditional on COMPLETE v15 selection
acceptance (original17 source-supported correct +2 predefined faithful uncertainty,
all19 original grades retained, context/source/tenant/final replay checks passed).
Intermediate selection audit with incomplete grades is NOT a passed acceptance.

Same previously registered larger gate, no threshold/model/question changes:
LongMemEval-S dev100 manifest,100 exposed development/regression questions, baseline
raw-primary memory-aware-v2 vs grounded-v15, three runs per arm, temperature0,
provider seed unsupported. Both configs/fallback.yaml: Gemini3.5flash-lite reader,
Gemma4-31b-it original judge, existing extractor/embedder, no new ingestion.
No per-question dev100 gold/failure inspection for later tuning. Aggregate/type only.
Reuse v15-r2 passing train146/146 and dev100 context-only preflight. Source archive,
public dataset/store proof, source/config/store/index/manifest hashes and git/Python
identity before external calls. No commits or default promotion.

New failure-safe driver tools/run_grounded_paired.py and unique namespace
results/raw/grounded-paired-dev100-v15-{baseline,candidate}-rep{1,2,3}.
Each arm/repeat saves100 readers before judging their exact saved outputs. Judge
errors/daily quota checkpoint all requests and reader rows; resume never reruns a
saved reader to obtain an ungraded hypothesis. Record original judge row SHA and
outer execution identity, combine only complete bound reader/grade pairs. Preserve
incomplete/negative results. No score-dependent retry/selection; repeat order is
baseline/candidate for rep1, then same rep2, then rep3. Models/prompts stay frozen.

Budget <=1200 reader calls (at most2 each) +600 successful judge calls before
internal retries; estimated~700reader and600judge,~4million reader input/output tokens,
not a money quote or guarantee.500reader daily request cap means multiple quota days;
judge high demand503 errors can increase elapsed time. Check current quota/cost;
never silently change model/schema. No scheduled continuation unless user requests.

Gate requires all600 exact rows; mean paired net>=4 correct/100, every type mean
net>=-2, candidate median context<=6000. The +4 adoption threshold reuses dev100's
registered3-repeat noise rule; it is not a significance claim. Independent units
are100questions, not300 independent questions. Failure/incomplete does not permit
recursive-summary tests or a default promotion. A pass permits the separately
registered summary navigation/cost/no-loss experiment, not new unseen-final figures.

Verify new persistence/identity join with targeted tests and original gate tests,
Ruff/format; freeze the actual driver in paired inventory before execution.
