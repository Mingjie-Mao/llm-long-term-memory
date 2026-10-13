# Uniform finite judge transport continuation

2026-10-10 Australia/Sydney, before resumed calls. Parent source/config/decision
freeze and all previous operational amendments remain unchanged.

The continuation completed control-rep1 and100 candidate-rep1 answers, then stopped
after12 candidate grades with two ConnectError attempts reporting DNS resolution
failure. Those exceptions were outside the registered RecoveringJudge's transient
classification and escaped, leaving progress RUNNING. This is execution evidence;
partial correctness outcomes are not consulted.

- Both arms and every repeat receive the same policy, based only on a saved
  grading block and its last failed judge usage entry.
- A transport-interrupted grading identity gets at most TWO further whole-process
  resumptions, durably charged before spawning. Eligible types are ConnectError,
  ConnectTimeout, ReadTimeout, ReadError, WriteTimeout and RemoteProtocolError.
- Requires all100 saved answers and an unsaved grade. The first missing grade in
  manifest order is selected. Reader and last failed-attempt SHA256 are recorded.
- Original server-recovery state, prior amendment state and usage are retained.
  Existing completed grades are verified and skipped; no answer is regenerated.
- Restart does not refresh this allowance. Existing client limits remain API2 /
  transport2; at most six extra client attempts per grading identity are enabled.
- HTTP400/401/429, schema errors, a second server-recovery exhaustion, or other
  permanent errors stop. Daily quota is never cleared or bypassed.
- No model, question set, grade prompt, outcome, gate or reader behavior changes.
  Completed reporting must disclose this amendment and all added usage.

The separate `t4_transport_resume.py` freezes its source and this rule, verifies the
parent resume's159 source hashes plus saved row prefixes, and delegates to the
unchanged parent continuation. An incomplete run remains incomplete.
