# Experimental area

This directory is reserved for work that is **not** part of the default
production path.

The RAG/agent prototype currently lives in [`src/agents/`](../src/agents/) and is
kept there because the existing API/orchestrator imports it. It is explicitly
marked as a prototype, excluded from `make pipeline`, and is **not** production
RAG:

- no pgvector / embedding model,
- no hybrid search / reranker / citations,
- no retrieval evaluation set.

A future real RAG implementation should move here (for example
`experimental/agents/rag/`) together with an evaluation harness. Until then,
do not describe the project as having production RAG.
