# v4 gate calculation correction — 2026-09-12

This qualifies the existing gate protocol, diagnosis and v4.2 preregistration; historical results and preregistration thresholds are preserved. No provider calls were made.

1. The flat/flat2 pair has **38 categorical verdict changes but 27 binary correctness changes**: 10 wins and 17 losses. Eleven switches among wrong, abstained and unparseable do not move accuracy. The paired-sign variance uses the 27, not the 38. `check_gate_is_resolvable.py` now reads wins + losses; `diagnose_v4_probes.py` emits both measures.
2. For n=30, the historical pair implies a conditional two-sigma null spread of **4.8 probes for one run** and **2.7 for three repeats under the homogeneous-probability model**. The old 5.7/3.9 values used categorical changes. These are null-spread approximations, not minimum detectable effects at a specified statistical power.
3. Heterogeneity does **not** guarantee more benefit from repeats. Counterexample: half the probes are deterministic and half have correctness probability 0.5. Pair disagreement is 0.25 and remains 0.25 after any odd-majority vote. The homogeneous model predicts a reduction. Tests now preserve this counterexample, and the tool explicitly labels its output conditional.
4. The two historical runs lack a captured source/configuration identity. Equal variant labels and equal retrieval do not establish identical prompts, fallback behavior or provider settings. “Pure sampling” cannot be concluded from these files. Historical categorical changes are descriptive evidence, not a verified noise floor.
5. The v4.2 candidate and preregistration exist, but the quoted resolution check needs a dated prospective addendum before a paid run. Keep the original document intact. Specify the estimand, repeated-run aggregation, paired inference and uncertainty assumptions; bind source/configuration/database/index hashes. The current harness also needs an explicit 30-count-only repeat schedule to implement the registered 404-row budget without rerunning all 142 probes for every repeat.

These corrections neither promote v4.1 nor prove that v4.2 cannot work. They remove unsupported certainty from the decision procedure. The two old runs are archived as exploratory, not promoted:
[v4.0-flat2](../archive/v4.0-flat2/README.md), [v4.1-scan](../archive/v4.1-scan/README.md).
