# Four prerequisite repairs, 2026-10-03

Development and regression evidence only. Models and databases unchanged; no commits.
Old dev100 v15 paired run stopped by user priority change, never resumed under new code.

1. Source gaps: original unsupported cases stay uncertain, expose missing period/event
   identity with raw citations. Synthetic same-tenant clarification resolves a supported
   computation, another tenant's clarification cannot. No guessed source dates.
2. Extraction: preserve verbatim personal user assertions as an opt-in additive source
   repair, never classify plans as preferences. Exact quote anchors; conservative numeric
   assistant-echo reanchoring only with one matching user source. Existing original facts
   retained; fingerprint changes; real pipeline and tenant isolation checks required.
   Train150's 9 preference cases are exposed development. No dev100 per-item labels.
   Measure quotation retention and provenance, not QA accuracy. Zero model requests.
3. History: replay the actual v15 reader context using the existing fact-key vector index
   at 1/2/4x history on scratch copies of train150. Every gold source body must be intact,
   context <=6000. First loss classified before any new retrieval fix. Target: no drop from
   1x to 4x; if not met, preserve failure and repair in a separately named run. Synthetic
   donor history does not test contradictions, unbounded growth or answer accuracy.
4. Judge recovery: bounded outer recovery after inner client retries; only HTTP500/502/503/
   504 or transport failures, no permanent/schema/quota/identity retry. Persist usage on
   every attempt, cooldown and retry budget survive resume, strict saved-reader hash and
   no duplicate grade or reader rerun. Fault injection, zero provider calls.

Do not claim new overall accuracy from any of these gates. Recursive LLM summary cost/
correctness comparison follows repairs in its own preregistration; summaries are navigation
only and raw evidence stays authoritative. No paid calls in this prerequisite repair.
