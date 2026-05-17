"""
Hybrid search combining BM25 (keyword) and dense (semantic) retrieval,
fused with Reciprocal Rank Fusion (RRF).

Equivalent to LangChain's EnsembleRetriever but fully self-contained
and runnable without an OpenAI key (uses a hash-based default embedding).
"""

import math
from typing import Optional
import numpy as np
from rank_bm25 import BM25Okapi


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

class BM25Index:
    """Keyword-based BM25 search."""

    def __init__(self, documents: list[str]):
        self.documents = documents
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """Returns list of (doc_index, score) sorted by score desc."""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def get_top_docs(self, query: str, top_k: int = 5) -> list[str]:
        results = self.search(query, top_k)
        return [self.documents[i] for i, _ in results if _ > 0]


# ---------------------------------------------------------------------------
# Dense index
# ---------------------------------------------------------------------------

class DenseIndex:
    """Semantic search using cosine similarity on numpy vectors."""

    def __init__(self, documents: list[str], embed_fn=None):
        self.documents = documents
        self.embed_fn = embed_fn or self._default_embed
        self.embeddings = np.array([self.embed_fn(d) for d in documents])

    def _default_embed(self, text: str) -> list[float]:
        """Deterministic hash-based embedding (no model downloads)."""
        import hashlib
        h = hashlib.md5(text.encode()).hexdigest()
        vec = [int(h[i:i+2], 16) / 255.0 for i in range(0, 32, 2)]
        norm = math.sqrt(sum(x**2 for x in vec))
        return [x / norm for x in vec] if norm > 0 else vec

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        q_vec = np.array(self.embed_fn(query))
        # Cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0 or np.any(norms == 0):
            return []
        sims = self.embeddings @ q_vec / (norms * q_norm)
        ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def get_top_docs(self, query: str, top_k: int = 5) -> list[str]:
        results = self.search(query, top_k)
        return [self.documents[i] for i, _ in results]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = 60
) -> list[tuple[int, float]]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    Formula: RRF(d) = Σ 1 / (k + rank_i(d))

    k=60 is the standard constant (Cormack et al., 2009).
    Higher k → less emphasis on top ranks.
    """
    scores: dict[int, float] = {}
    for ranked_list in ranked_lists:
        for rank, (doc_idx, _) in enumerate(ranked_list):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Hybrid searcher
# ---------------------------------------------------------------------------

class HybridSearcher:
    """
    Combines BM25 (keyword) + Dense (semantic) search with RRF fusion.
    Equivalent to LangChain's EnsembleRetriever.
    """

    def __init__(
        self,
        documents: list[str],
        embed_fn=None,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
    ):
        self.documents = documents
        self.bm25  = BM25Index(documents)
        self.dense = DenseIndex(documents, embed_fn)
        self.bm25_weight  = bm25_weight
        self.dense_weight = dense_weight

    def search(self, query: str, top_k: int = 5) -> list[str]:
        bm25_results  = self.bm25.search(query, top_k=top_k * 2)
        dense_results = self.dense.search(query, top_k=top_k * 2)
        fused = reciprocal_rank_fusion([bm25_results, dense_results])
        return [self.documents[i] for i, _ in fused[:top_k]]

    def explain(self, query: str, top_k: int = 3) -> list[dict]:
        """Return search results with scores from each retriever for inspection."""
        bm25_results  = dict(self.bm25.search(query, top_k * 2))
        dense_results = dict(self.dense.search(query, top_k * 2))
        fused = reciprocal_rank_fusion([
            list(bm25_results.items()),
            list(dense_results.items())
        ])
        return [
            {
                "doc": self.documents[i][:80],
                "rrf_score":   round(s, 4),
                "bm25_score":  round(bm25_results.get(i, 0.0), 4),
                "dense_score": round(dense_results.get(i, 0.0), 4),
            }
            for i, s in fused[:top_k]
        ]


# ---------------------------------------------------------------------------
# Cohere rerank (mock)
# ---------------------------------------------------------------------------

def rerank_mock(query: str, docs: list[str]) -> list[str]:
    """
    Mock Cohere rerank. In production:
        import cohere
        co = cohere.Client(api_key)
        results = co.rerank(query=query, documents=docs, model="rerank-english-v3.0")

    Mock: sort by query word overlap (Jaccard-like).
    """
    query_words = set(query.lower().split())
    def overlap(doc):
        doc_words = set(doc.lower().split())
        return len(query_words & doc_words) / max(len(query_words | doc_words), 1)
    return sorted(docs, key=overlap, reverse=True)


# ---------------------------------------------------------------------------
# Query rewriting
# ---------------------------------------------------------------------------

def query_rewrite(query: str, strategy: str = "expand") -> str:
    """
    Rewrite query to improve retrieval.

    Strategies:
    - expand: add clarifying words
    - hypothetical: generate a hypothetical answer document
    - decompose: break into sub-questions (returns first sub-question)
    """
    if strategy == "expand":
        return query + " detailed explanation overview"
    elif strategy == "hypothetical":
        return f"A document that answers '{query}' would say:"
    elif strategy == "decompose":
        # In production: use LLM to decompose
        parts = query.split(" and ")
        return parts[0].strip() if len(parts) > 1 else query
    return query
