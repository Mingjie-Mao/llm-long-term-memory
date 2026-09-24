# Delayed paraphrase retention v1 — preregistration

Date: 2026-09-24. Class: regression over the inspected `v2b-gate16` question set.
It is not an unseen QA benchmark. All 16 item IDs and their one hand-written
paraphrase are fixed in `results/manifests/retention-paraphrase-v1.json` before
retrieval. Gold source-session IDs come from the LongMemEval-S corpus, not from
the candidate store. Every item uses its original tenant namespace.

Store: `stores/v2b-gate16-repair.db` and its existing NumPy index. Model for local
embeddings: `sentence-transformers/all-MiniLM-L6-v2`. Retrieval: config
`configs/v2b-batch8-repair.yaml`, semantic-only weights, top 20, temporal active
filter. No provider calls, tokens or quota; no new ingestion or answer generation.
Historical prompt/extractor versions remain the ones recorded in this store. Git
HEAD at registration: `4da58681b4e7ae83ed31529f8efc4c738175f6e9` plus the
existing working tree. Output namespace:
`results/analysis/retention-paraphrase-v1.{json,md}`.

For each item, run the original and paraphrased question at its recorded question
date plus 1, 7 and 30 days. Reopen the database and index from disk. Report:

1. whether every gold source session still has any durable memory;
2. candidate-set, top-20 any-source and all-source coverage;
3. original-versus-paraphrase changes at each delay;
4. cross-tenant violations (must be zero).

Do not score answer correctness: no answerer is called. `as_of` changes the ranking
reference time, not actual wall-clock persistence. Thus this tests delayed-query
semantics and restart durability as a regression, not a real 30-day operational soak.
No threshold is used to promote a default; failures and negative results stay in
the generated report. Historical and frozen artifacts remain read-only.
