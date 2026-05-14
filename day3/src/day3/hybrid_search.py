"""
hybrid_search.py — BM25 + Dense Hybrid Search with RRF
=======================================================
Covers: BM25 keyword search, dense semantic search,
        Reciprocal Rank Fusion, hybrid retrieval.

Core insight:
  BM25 wins on exact terminology: "AMOLED 120Hz" matches exactly.
  Dense wins on semantic meaning: "lightweight device" ≈ "portable gadget".
  Hybrid wins overall: combines both signals for best recall.

Industry standard: Weaviate, Azure AI Search, Elasticsearch, and
Databricks Vector Search all ship hybrid search by default because
neither BM25 nor dense retrieval alone beats the combination.

Run: python -m day3.hybrid_search
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


# ══════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════

@dataclass
class SearchResult:
    """Single result returned by any search backend.

    Fields:
        text:     The retrieved document text.
        score:    Relevance score (higher is better, range depends on method).
        rank:     1-based rank within this result set.
        source:   Which retrieval method produced this result ("bm25", "dense", "hybrid").
        metadata: Arbitrary key-value data attached to this document.
    """
    text:     str
    score:    float
    rank:     int
    source:   str
    metadata: dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════
# BM25 INDEX
# ══════════════════════════════════════════════════════

class BM25Index:
    """BM25Okapi keyword search index.

    BM25 (Best Match 25) is the gold standard for keyword retrieval.
    It improves on TF-IDF by adding:
      - Term frequency saturation: repeated keywords give diminishing returns.
      - Document length normalisation: long docs are penalised proportionally.

    Formula: BM25(q,d) = sum_t IDF(t) * (tf(t,d) * (k1+1)) / (tf(t,d) + k1*(1 - b + b*|d|/avgdl))
      where k1=1.5 (saturation), b=0.75 (length penalty).

    When to use BM25:
      - Legal/medical documents with exact terminology.
      - Product catalogs (model numbers, SKUs, specs like "128GB", "120Hz").
      - Any query where the user knows the exact words in the target document.

    When BM25 fails:
      - Synonyms: "fast laptop" doesn't match "high-performance notebook".
      - Paraphrases: "what is the price" doesn't match "costs ₹24,900".
      → Use dense search for these cases.
    """

    def __init__(self):
        self._bm25 = None
        self._corpus: list[str] = []
        self._tokenized: list[list[str]] = []

    def index(self, documents: list[str]) -> None:
        """Tokenise and build BM25 index from document list.

        Args:
            documents: List of raw text strings to index.
        """
        from rank_bm25 import BM25Okapi

        self._corpus = documents
        self._tokenized = [self._tokenize(d) for d in documents]
        self._bm25 = BM25Okapi(self._tokenized)

    def search(self, query: str, n: int = 10) -> list[SearchResult]:
        """Search the BM25 index for a query.

        Args:
            query: Raw query string (lowercased and tokenised internally).
            n:     Number of top results to return.

        Returns:
            List of SearchResult sorted by BM25 score descending.
        """
        if self._bm25 is None or not self._corpus:
            return []

        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Sort by score descending, take top n
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n]
        results = []
        for rank, (idx, score) in enumerate(ranked, start=1):
            results.append(SearchResult(
                text=self._corpus[idx],
                score=float(score),
                rank=rank,
                source="bm25",
            ))
        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase and split on whitespace.

        Production: use NLTK or spaCy for stemming/lemmatisation.
        For teaching purposes, simple whitespace split is sufficient.
        """
        return text.lower().split()


# ══════════════════════════════════════════════════════
# DENSE INDEX
# ══════════════════════════════════════════════════════

class DenseIndex:
    """Dense semantic search index using sentence embeddings.

    Dense retrieval embeds all documents into a shared vector space.
    At query time, the query is also embedded and cosine similarity
    identifies the semantically closest documents.

    Embedding model: all-MiniLM-L6-v2 (384-dim, fast, good quality).
    Production alternatives: text-embedding-3-small (OpenAI),
    bge-large-en (BAAI), or Databricks embedding endpoints.

    When dense wins over BM25:
      - Synonym queries: "laptop for programming" matches "notebook for coding".
      - Conceptual queries: "best device for travel" finds lightweight gadgets.
      - Questions: "how long does the battery last?" finds "18-hour battery life".

    When dense fails:
      - Exact technical terms: "Snapdragon 8 Gen 3" may not rank above
        "best processor" unless the model saw this term in training.
      - Out-of-vocabulary tokens: model numbers, product IDs, rare terminology.
      → Use BM25 for these cases.
    """

    def __init__(
        self,
        model_name:     str = "all-MiniLM-L6-v2",
        mock_embed_fn:  Optional[Callable] = None,
    ):
        """
        Args:
            model_name:    SentenceTransformer model to load (lazy-loaded on first index()).
            mock_embed_fn: Callable(list[str]) -> np.ndarray for testing without model download.
        """
        self._model_name = model_name
        self._model = None
        self._mock_embed_fn = mock_embed_fn
        self._corpus: list[str] = []
        self._embeddings: Optional[np.ndarray] = None

    def index(self, documents: list[str]) -> None:
        """Encode documents and store embedding matrix.

        Args:
            documents: List of raw text strings to embed and index.
        """
        self._corpus = documents
        self._embeddings = self._encode(documents)

    def search(self, query: str, n: int = 10) -> list[SearchResult]:
        """Semantic search using cosine similarity.

        Args:
            query: Natural language query string.
            n:     Number of top results to return.

        Returns:
            List of SearchResult sorted by cosine similarity descending.
        """
        if self._embeddings is None or not self._corpus:
            return []

        query_vec = self._encode([query])[0]
        # Cosine similarity: dot product of L2-normalised vectors
        sims = self._embeddings @ query_vec  # (n_docs,)

        ranked_idx = np.argsort(sims)[::-1][:n]
        results = []
        for rank, idx in enumerate(ranked_idx, start=1):
            results.append(SearchResult(
                text=self._corpus[idx],
                score=float(sims[idx]),
                rank=rank,
                source="dense",
            ))
        return results

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to normalised embedding vectors.

        Uses mock_embed_fn if provided (for tests), otherwise lazy-loads
        the SentenceTransformer model on first call.
        """
        if self._mock_embed_fn is not None:
            vecs = self._mock_embed_fn(texts)
        else:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
            vecs = self._model.encode(texts, convert_to_numpy=True)

        vecs = np.array(vecs, dtype=np.float32)
        # L2 normalise each vector so dot product == cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-9)


# ══════════════════════════════════════════════════════
# RECIPROCAL RANK FUSION
# ══════════════════════════════════════════════════════

def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k:            int = 60,
) -> list[SearchResult]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion (RRF).

    RRF is rank-position based, so scores from different systems (BM25 scores
    vs cosine similarities) don't need to be on the same scale. This is the
    key advantage over score-based fusion methods.

    Formula:
        RRF_score(d) = sum_i [ 1 / (k + rank_i(d)) ]

    where rank_i(d) is the position of document d in the i-th result list,
    and k=60 is a smoothing constant (standard value from the original paper).

    Why k=60?
      Empirically shown to give the best trade-off between high-ranked and
      low-ranked documents across many retrieval tasks.
      Lower k: high-ranked documents get disproportionately high weight.
      Higher k: rank differences matter less, results are more evenly weighted.

    Documents appearing in MULTIPLE lists get higher fused scores —
    this is the cross-system agreement signal that makes RRF powerful.

    Args:
        result_lists: List of ranked result lists from different retrievers.
        k:            Smoothing constant (default 60, per Cormack et al. 2009).

    Returns:
        Deduplicated list of SearchResult sorted by fused RRF score descending.
    """
    if not result_lists:
        return []

    # Accumulate RRF scores per document (keyed by text)
    fused_scores: dict[str, float] = {}
    doc_map: dict[str, SearchResult] = {}

    for result_list in result_lists:
        for result in result_list:
            key = result.text
            rrf_contribution = 1.0 / (k + result.rank)
            fused_scores[key] = fused_scores.get(key, 0.0) + rrf_contribution
            if key not in doc_map:
                doc_map[key] = result

    # Sort by fused score descending
    sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    fused_results = []
    for rank, (text, score) in enumerate(sorted_items, start=1):
        original = doc_map[text]
        fused_results.append(SearchResult(
            text=original.text,
            score=round(score, 6),
            rank=rank,
            source="hybrid_rrf",
            metadata=original.metadata,
        ))

    return fused_results


# ══════════════════════════════════════════════════════
# HYBRID SEARCHER
# ══════════════════════════════════════════════════════

class HybridSearcher:
    """Combines BM25 and dense retrieval with Reciprocal Rank Fusion.

    Architecture:
        Query → BM25 top-n  ─┐
                              ├─→ RRF fusion → Final ranked results
        Query → Dense top-n ─┘

    Weights (bm25_weight, dense_weight) are informational only —
    RRF is rank-based, so these weights don't directly scale scores.
    They are used here for documentation and future score-blending variants.
    """

    def __init__(
        self,
        bm25_weight:   float = 0.3,
        dense_weight:  float = 0.7,
        mock_embed_fn: Optional[Callable] = None,
    ):
        """
        Args:
            bm25_weight:   Informational weight for BM25 component.
            dense_weight:  Informational weight for dense component.
            mock_embed_fn: Pass to DenseIndex for test environments.
        """
        self.bm25_weight  = bm25_weight
        self.dense_weight = dense_weight
        self._bm25  = BM25Index()
        self._dense = DenseIndex(mock_embed_fn=mock_embed_fn)
        self._corpus: list[str] = []

    def index(self, documents: list[str]) -> None:
        """Index documents in both BM25 and dense backends.

        Args:
            documents: List of raw text strings.
        """
        self._corpus = documents
        self._bm25.index(documents)
        self._dense.index(documents)

    def search(self, query: str, n: int = 5) -> list[SearchResult]:
        """Run hybrid search: BM25 + dense, fused with RRF.

        Args:
            query: Natural language or keyword query.
            n:     Number of top results to return.

        Returns:
            List of SearchResult from RRF fusion.
        """
        fetch_n = max(n * 3, 20)  # fetch more from each backend before fusion
        bm25_results  = self._bm25.search(query, n=fetch_n)
        dense_results = self._dense.search(query, n=fetch_n)

        fused = reciprocal_rank_fusion([bm25_results, dense_results])
        return fused[:n]

    def compare(self, query: str, n: int = 5) -> dict:
        """Run all three retrieval methods and return side-by-side results.

        Returns:
            Dict with keys: bm25_results, dense_results, hybrid_results.
            Each value is a list[SearchResult].
        """
        bm25_results  = self._bm25.search(query, n=n)
        dense_results = self._dense.search(query, n=n)
        hybrid_results = self.search(query, n=n)

        return {
            "bm25_results":   bm25_results,
            "dense_results":  dense_results,
            "hybrid_results": hybrid_results,
        }


# ══════════════════════════════════════════════════════
# PRODUCT CORPUS
# ══════════════════════════════════════════════════════

def generate_product_corpus() -> list[str]:
    """Return 20 realistic product descriptions for hybrid search demos.

    Designed to highlight BM25 vs dense differences:
      - Technical specs (BM25 territory): "AMOLED", "120Hz", "Snapdragon 8 Gen 3"
      - Semantic concepts (dense territory): "lightweight", "great for travel",
        "best for machine learning"
    """
    return [
        # Smartphones
        "Samsung Galaxy S24 Ultra with 200MP camera, Snapdragon 8 Gen 3 processor, "
        "6.8-inch Dynamic AMOLED 2X 120Hz display, and titanium frame. Price ₹1,29,999.",

        "Apple iPhone 15 Pro Max features A17 Pro chip, 48MP main camera with 5x optical zoom, "
        "titanium design, and USB-C. Battery life 29 hours video playback. Price ₹1,59,900.",

        "OnePlus 12 has 100W SUPERVOOC wired charging, 50W wireless charging, "
        "Hasselblad camera system, 6.7-inch AMOLED 120Hz display. Price ₹64,999.",

        "Google Pixel 8 Pro has Google Tensor G3 chip, 50MP main camera with Magic Eraser, "
        "Temperature sensor, and 7 years of OS updates. Price ₹1,06,999.",

        "Xiaomi 14 Ultra has Leica Summilux optics, four 50MP cameras including a 1-inch sensor, "
        "5000mAh battery with 90W charging, Snapdragon 8 Gen 3. Price ₹99,999.",

        # Laptops
        "MacBook Pro 14 M3 Pro chip with 18GB unified memory, 18-hour battery life, "
        "Liquid Retina XDR 120Hz display, ideal for machine learning and data science. Price ₹1,99,900.",

        "Dell XPS 15 with 15.6-inch OLED display, Intel Core i9-13900H, NVIDIA RTX 4070, "
        "32GB RAM, ideal for content creation and video editing. Price ₹1,89,990.",

        "Lenovo ThinkPad X1 Carbon Gen 12, ultra-lightweight at 1.12kg, MIL-SPEC durability, "
        "Intel Core Ultra 7, 32GB LPDDR5. Best laptop for travelling executives. Price ₹1,49,990.",

        "ASUS ROG Zephyrus G14 with AMD Ryzen 9 8945HS, Radeon RX 7700S, "
        "144Hz display, best gaming laptop under ₹1.5 lakh. Price ₹1,39,990.",

        "Microsoft Surface Pro 10, 2-in-1 tablet and laptop with Intel Core Ultra 5, "
        "13-inch PixelSense display, Surface Slim Pen 2 support, runs Windows 11 Pro. Price ₹1,59,999.",

        # Headphones
        "Sony WH-1000XM5 wireless headphones with industry-leading active noise cancellation, "
        "8 microphones, dual processors, 30-hour battery life, and multipoint Bluetooth. Price ₹29,990.",

        "Apple AirPods Pro 2nd generation with H2 chip, Adaptive Transparency, "
        "Personalized Spatial Audio, and USB-C charging case. Price ₹24,900.",

        "Bose QuietComfort 45 headphones with Quiet Mode ANC, 24-hour battery life, "
        "soft earcushion comfort, and high-fidelity audio. Price ₹24,900.",

        "Jabra Elite 10 true wireless earbuds with MultiSensor Voice technology, "
        "Adaptive ANC, Dolby Atmos, and 6-hour battery per charge. Price ₹22,999.",

        # Tablets
        "iPad Pro 12.9-inch M2 chip with Apple Pencil 2nd gen hover support, "
        "Liquid Retina XDR display, 16GB RAM, ideal for creative professionals. Price ₹1,12,900.",

        "Samsung Galaxy Tab S9 Ultra, 14.6-inch Dynamic AMOLED 2X display, "
        "Snapdragon 8 Gen 2, S Pen included, IP68 water resistance. Price ₹1,08,999.",

        "Amazon Kindle Paperwhite 11th gen with 6.8-inch glare-free display, "
        "300ppi resolution, adjustable warm light, and 10-week battery life. Price ₹13,999.",

        # Accessories
        "Logitech MX Master 3S mouse with MagSpeed electromagnetic scroll wheel, "
        "Flow multi-device feature, 8000 DPI. Best mouse for productivity. Price ₹9,995.",

        "Anker 737 PowerCore 24K power bank with 140W output, 24000mAh capacity, "
        "charges MacBook Pro from 0 to 50% in 45 minutes. Price ₹6,999.",

        "Samsung T7 Shield portable SSD with 2TB capacity, IP65 dust and water resistance, "
        "USB 3.2 Gen 2 with 1050 MB/s read speed. Price ₹12,999.",
    ]


# ══════════════════════════════════════════════════════
# DEMO: HYBRID vs SPARSE vs DENSE
# ══════════════════════════════════════════════════════

def demo_hybrid_vs_sparse_vs_dense(mock_embed_fn=None) -> dict:
    """Demonstrate when each retrieval method wins.

    Three curated queries:
      1. "best laptop for ML" — dense wins (semantic, no exact match needed)
      2. "AMOLED 120Hz display" — BM25 wins (exact technical terms)
      3. "lightweight device for travel" — hybrid wins (combines both signals)

    Args:
        mock_embed_fn: Optional embed function for testing (no model download).

    Returns:
        Dict with results per query per method.
    """
    corpus = generate_product_corpus()
    searcher = HybridSearcher(mock_embed_fn=mock_embed_fn)
    searcher.index(corpus)

    test_queries = [
        ("best laptop for ML",        "dense wins — semantic similarity to 'machine learning'"),
        ("AMOLED 120Hz display",       "BM25 wins — exact technical terms match"),
        ("lightweight device for travel", "hybrid wins — combines portability + travel signals"),
    ]

    results = {}
    for query, explanation in test_queries:
        comparison = searcher.compare(query, n=3)
        results[query] = {
            "explanation": explanation,
            "bm25_top1":   comparison["bm25_results"][0].text[:80] if comparison["bm25_results"] else "",
            "dense_top1":  comparison["dense_results"][0].text[:80] if comparison["dense_results"] else "",
            "hybrid_top1": comparison["hybrid_results"][0].text[:80] if comparison["hybrid_results"] else "",
        }
    return results


# ══════════════════════════════════════════════════════
# MAIN DEMO
# ══════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("HYBRID SEARCH: BM25 + DENSE + RRF DEMO")
    print("=" * 70)

    corpus = generate_product_corpus()
    print(f"\n[1] Building index on {len(corpus)} product descriptions...")

    # Use mock embeddings for standalone demo (no model download required at import)
    def _demo_embed(texts):
        vecs = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**31))
            v = rng.standard_normal(384).astype(np.float32)
            v = v / np.linalg.norm(v)
            vecs.append(v)
        return np.array(vecs)

    searcher = HybridSearcher(mock_embed_fn=_demo_embed)
    searcher.index(corpus)
    print("  Index built.")

    print("\n[2] BM25 Search — 'AMOLED 120Hz display' (exact terms)")
    bm25 = BM25Index()
    bm25.index(corpus)
    for r in bm25.search("AMOLED 120Hz display", n=3):
        print(f"  [{r.rank}] score={r.score:.3f} | {r.text[:80]}")

    print("\n[3] Dense Search — 'best laptop for machine learning' (semantic)")
    dense = DenseIndex(mock_embed_fn=_demo_embed)
    dense.index(corpus)
    for r in dense.search("best laptop for machine learning", n=3):
        print(f"  [{r.rank}] score={r.score:.4f} | {r.text[:80]}")

    print("\n[4] RRF Fusion — 'lightweight device for travel'")
    comparison = searcher.compare("lightweight device for travel", n=5)
    print("  BM25 top-3:")
    for r in comparison["bm25_results"][:3]:
        print(f"    [{r.rank}] {r.text[:70]}")
    print("  Dense top-3:")
    for r in comparison["dense_results"][:3]:
        print(f"    [{r.rank}] {r.text[:70]}")
    print("  Hybrid RRF top-3:")
    for r in comparison["hybrid_results"][:3]:
        print(f"    [{r.rank}] score={r.score:.5f} | {r.text[:70]}")

    print("\n[5] RRF Formula Demo (k=60)")
    # Show what RRF does with a tiny example
    list_a = [
        SearchResult("Doc A", 10.0, 1, "bm25"),
        SearchResult("Doc B", 8.0,  2, "bm25"),
        SearchResult("Doc C", 5.0,  3, "bm25"),
    ]
    list_b = [
        SearchResult("Doc C", 0.95, 1, "dense"),
        SearchResult("Doc A", 0.90, 2, "dense"),
        SearchResult("Doc D", 0.85, 3, "dense"),
    ]
    fused = reciprocal_rank_fusion([list_a, list_b])
    print("  BM25 list:   Doc A(#1) > Doc B(#2) > Doc C(#3)")
    print("  Dense list:  Doc C(#1) > Doc A(#2) > Doc D(#3)")
    print("  Fused RRF:")
    for r in fused:
        print(f"    [{r.rank}] {r.text} | RRF_score={r.score:.6f}")


if __name__ == "__main__":
    main()
