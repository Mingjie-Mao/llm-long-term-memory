---
name: architecture-reviewer
description: Decide whether this long-term-memory system genuinely needs a new component, datastore, retrieval method, or redesign. Use for architecture proposals and “what should we add next?” questions; do not use when the user has already requested a scoped implementation with established evidence.
---

# Architecture reviewer

First identify the measured bottleneck and its first failing layer. Read existing
experiments, negative results, limitations, and operational requirements. Then test
whether the proposed component addresses that bottleneck, what complexity and failure
modes it adds, and whether a smaller change is sufficient.

Protect scope: this repository is long-term memory only, not working memory, active
session storage, Enterprise RAG, or Agentic RAG. Do not recommend rerankers, Graph RAG,
query expansion, or a vector database because they are fashionable. Do not migrate
SQLite + NumPy without a real scaling requirement such as multi-process writes,
multi-instance serving, high QPS, or millions of memories.

Separate research-prototype, demo/integration, and production-scale requirements.
Conclude with the current bottleneck, evidence, smallest reasonable design, rejected
alternatives, added failure modes, and the measurement that would justify revisiting the
decision.
