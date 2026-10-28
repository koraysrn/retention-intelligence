"""RAG component: product catalog, campaign history and brand guideline retrieval.

======================================================================
WARNING: EXPERIMENTAL PROTOTYPE ONLY. NOT PRODUCTION RAG.
This module is a deterministic keyword-overlap retriever (Jaccard) over four
in-memory documents. It does NOT use pgvector, embeddings, hybrid search,
a reranker, citations or a retrieval evaluation set. It is excluded from the
default `make pipeline`; see `experimental/README.md`.
======================================================================

pgvector is the design target; when it is not installed (prototype/test), the
deterministic fallback below is used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class RetrievalResult:
    chunks: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)


DEFAULT_DOCUMENTS: list[dict] = [
    {
        "id": "camp-1",
        "text": "End-of-season discount campaign up to 15% on sports and outdoor products.",
        "source": "campaign_history",
    },
    {
        "id": "camp-2",
        "text": "Loyalty discount in the home decoration and kitchen products category.",
        "source": "campaign_history",
    },
    {
        "id": "cat-1",
        "text": "New season products for electronic accessories and gadgets.",
        "source": "product_catalog",
    },
    {
        "id": "tone-1",
        "text": "Brand tone: friendly, short sentences, address the customer by name, KVKK compliant.",
        "source": "brand_guidelines",
    },
]


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


class RAGRetriever:
    """Context retriever. Uses the in-memory document set by default."""

    def __init__(self, documents: list[dict] | None = None) -> None:
        self.documents = documents if documents is not None else DEFAULT_DOCUMENTS

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Return the most relevant document chunks for the customer context."""
        query_tokens = _tokenize(query)
        scored: list[tuple[float, dict]] = []
        for doc in self.documents:
            doc_tokens = _tokenize(doc["text"])
            if not doc_tokens:
                score = 0.0
            else:
                score = len(query_tokens & doc_tokens) / len(query_tokens | doc_tokens)
            scored.append((score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[: max(1, top_k)]
        return RetrievalResult(
            chunks=[doc["text"] for _, doc in top],
            sources=[doc["source"] for _, doc in top],
            scores=[score for score, _ in top],
        )
