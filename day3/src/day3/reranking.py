"""
reranking.py — Cross-Encoder Reranking, MMR, Metadata Filtering
================================================================
Covers: cross-encoder reranking, Maximal Marginal Relevance,
        metadata-based filtering, top-k retrieval strategies.

Why reranking?
  First-stage retrieval (BM25, dense, hybrid) is fast but imprecise.
  It returns top-50 candidates. A slower but more accurate reranker
  then re-scores all 50 and picks the best 3-5.

  Two-stage retrieval pattern (industry standard):
    Stage 1: fast retrieval (BM25/dense) → top-50 candidates in <10ms
    Stage 2: reranker (cross-encoder) → top-5 final results in 50-200ms

  Cross-encoder vs Bi-encoder:
    Bi-encoder: query and document embedded SEPARATELY → fast, approximate.
    Cross-encoder: query and document processed TOGETHER → slow, accurate.
    Rerankers use cross-encoders because they can model query-document
    interaction at the token level.

Run: python -m day3.reranking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


# ══════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════

@dataclass
class RankedResult:
    """Result with both initial (retrieval) and final (reranker) rank.

    Fields:
        text:          Document text.
        initial_rank:  Rank assigned by first-stage retrieval.
        final_rank:    Rank assigned after reranking.
        initial_score: Score from first-stage retrieval (BM25/cosine).
        final_score:   Score from reranker (cross-encoder logit).
        metadata:      Attached document metadata.
    """
    text:          str
    initial_rank:  int
    final_rank:    int
    initial_score: float
    final_score:   float
    metadata:      dict = field(default_factory=dict)

    @property
    def rank_changed(self) -> bool:
        """True if the reranker changed the document's position."""
        return self.initial_rank != self.final_rank


# ══════════════════════════════════════════════════════
# CROSS-ENCODER RERANKER
# ══════════════════════════════════════════════════════

class CrossEncoderReranker:
    """Cross-encoder reranking using sentence-transformers CrossEncoder.

    Cross-encoders process (query, document) pairs jointly — the query
    tokens attend to document tokens through full transformer attention.
    This models query-document interaction much more accurately than
    bi-encoders, at the cost of being O(n) in the number of candidates.

    Production models:
      - cross-encoder/ms-marco-MiniLM-L-6-v2  (fast, good quality)
      - cross-encoder/ms-marco-MiniLM-L-12-v2 (slower, better quality)
      - cross-encoder/ms-marco-electra-base    (best quality, slowest)

    MS-MARCO: Microsoft's large-scale passage retrieval dataset.
    Training on MS-MARCO gives strong generalisation to product queries,
    support tickets, and domain-specific documents.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Args:
            model_name: HuggingFace cross-encoder model identifier.
                        Lazy-loaded on first call to rerank().
        """
        self._model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy-load the CrossEncoder model (avoids import at module load time)."""
        if self._model is None:
            from sentence_transformers.cross_encoder import CrossEncoder
            self._model = CrossEncoder(self._model_name)

    def rerank(
        self,
        query:      str,
        candidates: list[str],
        top_k:      int = 5,
    ) -> list[RankedResult]:
        """Rerank candidates using the cross-encoder model.

        Args:
            query:      The user's query string.
            candidates: List of candidate documents from first-stage retrieval.
            top_k:      Number of top documents to return after reranking.

        Returns:
            List of RankedResult showing how ranks changed after reranking.
        """
        self._load_model()

        pairs = [(query, doc) for doc in candidates]
        scores = self._model.predict(pairs)

        # Sort by cross-encoder score descending
        reranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for final_rank, (initial_idx, score) in enumerate(reranked[:top_k], start=1):
            results.append(RankedResult(
                text=candidates[initial_idx],
                initial_rank=initial_idx + 1,
                final_rank=final_rank,
                initial_score=float(initial_idx + 1) * -1.0,  # proxy: retrieval order
                final_score=float(score),
            ))
        return results

    def rerank_mock(
        self,
        query:      str,
        candidates: list[str],
        top_k:      int = 5,
    ) -> list[RankedResult]:
        """Mock reranker for tests — scores by word overlap with query.

        No model download required. Useful for unit tests and CI environments.
        Word overlap is a reasonable proxy for relevance in test scenarios
        because our test corpus is designed with query-matching words.

        Args:
            query:      The user's query string.
            candidates: List of candidate documents.
            top_k:      Number of top documents to return.

        Returns:
            List of RankedResult with overlap-based scores.
        """
        query_words = set(query.lower().split())

        def overlap_score(doc: str) -> float:
            doc_words = set(doc.lower().split())
            if not doc_words:
                return 0.0
            return len(query_words & doc_words) / len(query_words | doc_words)

        scored = [(i, candidates[i], overlap_score(candidates[i]))
                  for i in range(len(candidates))]
        ranked = sorted(scored, key=lambda x: x[2], reverse=True)

        results = []
        for final_rank, (initial_idx, text, score) in enumerate(ranked[:top_k], start=1):
            results.append(RankedResult(
                text=text,
                initial_rank=initial_idx + 1,
                final_rank=final_rank,
                initial_score=0.0,
                final_score=round(score, 4),
            ))
        return results


# ══════════════════════════════════════════════════════
# MAXIMAL MARGINAL RELEVANCE
# ══════════════════════════════════════════════════════

def mmr_select(
    query_vec:    np.ndarray,
    doc_vecs:     np.ndarray,
    docs:         list[str],
    top_k:        int   = 5,
    lambda_param: float = 0.5,
) -> list[tuple[int, float]]:
    """Select documents using Maximal Marginal Relevance (MMR).

    MMR balances relevance to the query against diversity among selected docs.
    This prevents returning 5 near-duplicate documents that all say the same thing.

    Algorithm (greedy):
      1. Start with the document most similar to the query.
      2. At each step, pick the document that maximises:
           lambda * sim(query, doc_i) - (1 - lambda) * max(sim(selected_j, doc_i))
      3. Repeat until top_k documents are selected.

    The second term penalises documents similar to ALREADY SELECTED documents,
    ensuring diversity in the final set.

    Lambda values:
      lambda=1.0 → pure relevance (same as top-k cosine, no diversity)
      lambda=0.0 → pure diversity (maximise distance between selected docs)
      lambda=0.5 → balanced (recommended default for most RAG pipelines)

    When to use MMR:
      - Multi-document summarisation: avoid redundant source documents.
      - Diverse product recommendations: show different categories.
      - Knowledge base Q&A: cover multiple aspects of a question.

    Args:
        query_vec:    L2-normalised query embedding vector. Shape (d,).
        doc_vecs:     L2-normalised document embedding matrix. Shape (n, d).
        docs:         Raw text of each document (parallel to doc_vecs).
        top_k:        Number of documents to select.
        lambda_param: Trade-off between relevance (1.0) and diversity (0.0).

    Returns:
        List of (doc_index, mmr_score) tuples for selected documents in order.
    """
    if len(docs) == 0:
        return []

    top_k = min(top_k, len(docs))

    # Similarity of each document to the query
    query_sims = doc_vecs @ query_vec  # shape: (n,)

    selected_indices: list[int] = []
    selected_scores: list[float] = []
    remaining = list(range(len(docs)))

    for _ in range(top_k):
        if not remaining:
            break

        best_idx = -1
        best_score = -np.inf

        for cand_idx in remaining:
            relevance = float(query_sims[cand_idx])

            if not selected_indices:
                # No documents selected yet — pure relevance
                diversity_penalty = 0.0
            else:
                # Penalty = maximum similarity to any already-selected doc
                selected_vecs = doc_vecs[selected_indices]  # (k, d)
                sims_to_selected = selected_vecs @ doc_vecs[cand_idx]  # (k,)
                diversity_penalty = float(np.max(sims_to_selected))

            mmr_score = (lambda_param * relevance
                         - (1.0 - lambda_param) * diversity_penalty)

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = cand_idx

        selected_indices.append(best_idx)
        selected_scores.append(best_score)
        remaining.remove(best_idx)

    return list(zip(selected_indices, selected_scores))


# ══════════════════════════════════════════════════════
# METADATA FILTERING
# ══════════════════════════════════════════════════════

def metadata_filter(
    documents: list[dict],
    filters:   dict,
) -> list[dict]:
    """Filter documents by metadata before retrieval (pre-filtering).

    Pre-filtering dramatically reduces the candidate set for vector search,
    improving both speed and precision. Instead of searching all 10,000 docs,
    filter to 200 "laptop" docs first, then run vector search on those 200.

    Supported filter types:
      - Exact string match: {"category": "laptop"}
      - Numeric range (tuple): {"price_range": (50000, 200000)}
        Documents must have "price" field within [min, max].
      - List membership: {"tags": ["portable", "business"]}
        Documents must have a "tags" list containing at least one match.

    All filters are ANDed together (document must match ALL filters).

    Args:
        documents: List of {"text": str, "metadata": dict} dicts.
        filters:   Dict of filter conditions.
                   Special key "price_range": expects (min, max) tuple
                   and checks metadata["price"] field.

    Returns:
        Subset of documents matching ALL filter conditions.
    """
    if not filters:
        return documents

    results = []
    for doc in documents:
        meta = doc.get("metadata", {})
        match = True

        for key, value in filters.items():
            if key == "price_range":
                # Numeric range filter on metadata["price"]
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    price = meta.get("price", 0)
                    try:
                        price_val = float(str(price).replace("₹", "").replace(",", ""))
                        if not (value[0] <= price_val <= value[1]):
                            match = False
                            break
                    except (ValueError, TypeError):
                        # Can't parse price — skip this filter
                        pass
            elif isinstance(value, (list, tuple)):
                # List membership: doc's metadata value must overlap with filter list
                doc_val = meta.get(key, [])
                if isinstance(doc_val, str):
                    doc_val = [doc_val]
                if not set(doc_val) & set(value):
                    match = False
                    break
            else:
                # Exact string/value match
                if meta.get(key) != value:
                    match = False
                    break

        if match:
            results.append(doc)

    return results


# ══════════════════════════════════════════════════════
# FULL RETRIEVAL PIPELINE
# ══════════════════════════════════════════════════════

def build_retrieval_pipeline(
    documents:     list[dict],
    metadata_list: list[dict],
    mock_embed_fn: Optional[Callable] = None,
) -> Callable:
    """Build a full retrieval pipeline: filter → dense → rerank → return.

    Pipeline steps:
      1. Metadata pre-filter: narrow the candidate set.
      2. Dense retrieval: find semantically relevant candidates.
      3. Cross-encoder reranking: re-score candidates with higher accuracy.
      4. Return top_k final results.

    Args:
        documents:     List of {"text": str, "metadata": dict} dicts to index.
        metadata_list: Parallel list of metadata dicts (aligned with documents).
        mock_embed_fn: Optional mock embedding function for testing.

    Returns:
        A callable: retrieve(query: str, filters: dict = {}, top_k: int = 5)
        → list[RankedResult]
    """
    from day3.hybrid_search import DenseIndex

    # Build index at pipeline creation time
    all_texts = [d["text"] for d in documents]
    dense_index = DenseIndex(mock_embed_fn=mock_embed_fn)
    dense_index.index(all_texts)
    reranker = CrossEncoderReranker()

    def retrieve(
        query:   str,
        filters: dict = {},
        top_k:   int  = 5,
    ) -> list[RankedResult]:
        # Step 1: Metadata filter
        if filters:
            filtered = metadata_filter(documents, filters)
        else:
            filtered = documents

        if not filtered:
            return []

        # Step 2: Dense retrieval on filtered texts
        filtered_texts = [d["text"] for d in filtered]
        # Create a sub-index for the filtered set
        sub_index = DenseIndex(mock_embed_fn=mock_embed_fn)
        sub_index.index(filtered_texts)
        candidates_results = sub_index.search(query, n=min(top_k * 4, len(filtered_texts)))
        candidate_texts = [r.text for r in candidates_results]

        if not candidate_texts:
            return []

        # Step 3: Rerank (use mock to avoid model download in tests)
        reranked = reranker.rerank_mock(query, candidate_texts, top_k=top_k)
        return reranked

    return retrieve


# ══════════════════════════════════════════════════════
# STRATEGY COMPARISON
# ══════════════════════════════════════════════════════

def compare_retrieval_strategies(
    query:  str,
    corpus: list[str],
    mock_embed_fn: Optional[Callable] = None,
) -> dict:
    """Compare top-k, reranked, and MMR strategies on the same query.

    Shows the diversity vs relevance trade-off between:
      - Top-k: pure relevance, may return duplicates.
      - Top-k + rerank: higher precision, still may be redundant.
      - MMR lambda=0.5: balanced relevance and diversity.
      - MMR lambda=0.0: maximum diversity (often sacrifices top relevance).

    Args:
        query:         Query string to test.
        corpus:        List of document strings.
        mock_embed_fn: Mock embedding function for testing.

    Returns:
        Dict with results per strategy.
    """
    from day3.hybrid_search import DenseIndex

    dense = DenseIndex(mock_embed_fn=mock_embed_fn)
    dense.index(corpus)
    reranker = CrossEncoderReranker()

    # Top-k results
    topk_results = dense.search(query, n=5)
    topk_texts = [r.text for r in topk_results]

    # Reranked results
    reranked = reranker.rerank_mock(query, topk_texts, top_k=5)

    # MMR: need query vector and doc vectors
    query_vec = dense._encode([query])[0]
    doc_vecs = dense._embeddings
    if doc_vecs is None:
        mmr_balanced = []
        mmr_diverse = []
    else:
        mmr_balanced_idx = mmr_select(query_vec, doc_vecs, corpus, top_k=5, lambda_param=0.5)
        mmr_diverse_idx  = mmr_select(query_vec, doc_vecs, corpus, top_k=5, lambda_param=0.0)
        mmr_balanced = [corpus[i] for i, _ in mmr_balanced_idx]
        mmr_diverse  = [corpus[i] for i, _ in mmr_diverse_idx]

    return {
        "top_k":          topk_texts,
        "top_k_reranked": [r.text for r in reranked],
        "mmr_balanced":   mmr_balanced,
        "mmr_diverse":    mmr_diverse,
    }


# ══════════════════════════════════════════════════════
# MAIN DEMO
# ══════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("RERANKING, MMR & METADATA FILTERING DEMO")
    print("=" * 70)

    from day3.hybrid_search import generate_product_corpus, DenseIndex

    corpus = generate_product_corpus()

    # Demo mock embed function
    def _demo_embed(texts):
        vecs = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**31))
            v = rng.standard_normal(384).astype(np.float32)
            v = v / np.linalg.norm(v)
            vecs.append(v)
        return np.array(vecs)

    print("\n[1] Cross-Encoder Reranking (Mock)")
    reranker = CrossEncoderReranker()
    query = "noise cancelling wireless headphones"
    candidates = corpus[:8]  # simulate first-stage retrieval returning 8 docs
    reranked = reranker.rerank_mock(query, candidates, top_k=3)
    print(f"  Query: '{query}'")
    for r in reranked:
        change = "↑" if r.final_rank < r.initial_rank else ("↓" if r.final_rank > r.initial_rank else "=")
        print(f"  [{r.final_rank}] {change} (was #{r.initial_rank}) score={r.final_score:.4f} | {r.text[:70]}")

    print("\n[2] MMR — Relevance vs Diversity")
    dense = DenseIndex(mock_embed_fn=_demo_embed)
    dense.index(corpus)
    query_vec = dense._encode(["laptop for programming"])[0]
    doc_vecs = dense._embeddings

    print("  lambda=1.0 (pure relevance):")
    for idx, score in mmr_select(query_vec, doc_vecs, corpus, top_k=3, lambda_param=1.0):
        print(f"    [{score:.4f}] {corpus[idx][:80]}")

    print("  lambda=0.5 (balanced):")
    for idx, score in mmr_select(query_vec, doc_vecs, corpus, top_k=3, lambda_param=0.5):
        print(f"    [{score:.4f}] {corpus[idx][:80]}")

    print("  lambda=0.0 (pure diversity):")
    for idx, score in mmr_select(query_vec, doc_vecs, corpus, top_k=3, lambda_param=0.0):
        print(f"    [{score:.4f}] {corpus[idx][:80]}")

    print("\n[3] Metadata Filtering")
    docs_with_meta = [
        {"text": corpus[i], "metadata": {
            "category": "laptop" if "laptop" in corpus[i].lower() or "macbook" in corpus[i].lower() else "other",
            "price": 150000 if i < 5 else 50000,
        }}
        for i in range(len(corpus))
    ]

    laptop_docs = metadata_filter(docs_with_meta, {"category": "laptop"})
    print(f"  Total docs: {len(docs_with_meta)} | Laptop filter: {len(laptop_docs)} docs")

    budget_docs = metadata_filter(docs_with_meta, {"price_range": (0, 60000)})
    print(f"  Budget filter (₹0–60K): {len(budget_docs)} docs")

    print("\n[4] Full Pipeline (filter → dense → rerank)")
    retrieve = build_retrieval_pipeline(docs_with_meta, [], mock_embed_fn=_demo_embed)
    results = retrieve("best laptop for productivity", filters={"category": "laptop"}, top_k=3)
    print(f"  Query: 'best laptop for productivity' (filtered to laptops only)")
    for r in results:
        print(f"  [{r.final_rank}] score={r.final_score:.4f} | {r.text[:70]}")


if __name__ == "__main__":
    main()
