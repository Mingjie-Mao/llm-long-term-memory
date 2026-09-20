# v2d retrieval gate (zero model calls)

Decision: **KEEP_RETRIEVAL_FIXED**

- candidate / ranked / selected source recall: 46/48, 46/48, 46/48
- losses introduced by ranking: 0
- losses introduced by selection: 0
- wrong answers with source evidence already selected: 10/11
- wrong answers without selected source evidence: 1/11 (e48988bc)
- optimistic ceiling from fixing every observed retrieval miss: +2.1%

Most wrong answers already had a gold source session selected; no evidence was lost between candidate generation, ranking, and selection.

Therefore v2d does not add broad top-k, reranking, relation-scan, or raw-context expansion. The existing conditional archive fallback remains available. A new retrieval mechanism should be reconsidered only on a separately frozen set if candidate or ranked recall becomes a repeated failure mode.
