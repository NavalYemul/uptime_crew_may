"""
embeddings.py — Embedding Model Selection & Generation
========================================================
Covers: dense vs sparse vectors, embedding dimensions (384/768/1024/1536),
        sentence-transformers, model selection criteria, batch encoding.

Run:  python -m day2.embeddings
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Lazy import — only loads when first used (model download happens here)
_model_cache: dict[str, "SentenceTransformer"] = {}


# ══════════════════════════════════════════════════════
# MODEL CATALOGUE
# ══════════════════════════════════════════════════════

@dataclass
class EmbeddingModelSpec:
    """
    Metadata about an embedding model.
    Use this to pick the right model for your use case.
    """
    name:         str
    dims:         int       # output vector dimensionality
    max_tokens:   int       # max input tokens (longer = truncated)
    speed:        str       # "fast" | "medium" | "slow" | "API"
    quality:      str       # subjective quality tier
    size_mb:      int | str # download size ("API" for hosted models)
    multilingual: bool = False
    notes:        str  = ""


MODEL_CATALOGUE: list[EmbeddingModelSpec] = [
    EmbeddingModelSpec(
        name="all-MiniLM-L6-v2",
        dims=384, max_tokens=256, speed="fast",
        quality="good", size_mb=90, multilingual=False,
        notes="Best default for English. Small, fast, good quality. Start here.",
    ),
    EmbeddingModelSpec(
        name="all-mpnet-base-v2",
        dims=768, max_tokens=384, speed="medium",
        quality="better", size_mb=420, multilingual=False,
        notes="Higher quality than MiniLM. Use when accuracy > speed.",
    ),
    EmbeddingModelSpec(
        name="paraphrase-multilingual-mpnet-base-v2",
        dims=768, max_tokens=128, speed="medium",
        quality="multilingual", size_mb=420, multilingual=True,
        notes="50+ languages. Use for non-English or mixed-language corpora.",
    ),
    EmbeddingModelSpec(
        name="BAAI/bge-large-en-v1.5",
        dims=1024, max_tokens=512, speed="slow",
        quality="excellent", size_mb=1340, multilingual=False,
        notes="Top MTEB benchmark English model. Use for production RAG.",
    ),
    EmbeddingModelSpec(
        name="text-embedding-3-small",
        dims=1536, max_tokens=8191, speed="API",
        quality="excellent", size_mb="API", multilingual=True,
        notes="OpenAI API. Very long context. Requires OPENAI_API_KEY.",
    ),
    EmbeddingModelSpec(
        name="text-embedding-3-large",
        dims=3072, max_tokens=8191, speed="API",
        quality="best", size_mb="API", multilingual=True,
        notes="OpenAI API. Best quality. Higher cost. For highest-stakes RAG.",
    ),
]


def model_selection_guide(
    multilingual: bool = False,
    speed_priority: bool = True,
    long_context: bool = False,
    budget: str = "free",  # "free" | "paid"
) -> EmbeddingModelSpec:
    """
    Decision tree for picking the right embedding model.

    Criteria (in order of importance):
    1. Language requirements → multilingual model if needed
    2. Context length → max_tokens limit matters for long documents
    3. Budget → free (local) vs paid (API)
    4. Speed vs quality tradeoff

    Real-world: for a Databricks RAG pipeline, start with all-MiniLM-L6-v2,
    benchmark on your dataset, then upgrade if quality is insufficient.
    """
    candidates = [m for m in MODEL_CATALOGUE if m.size_mb != "API" or budget == "paid"]

    if multilingual:
        candidates = [m for m in candidates if m.multilingual] or candidates

    if long_context:
        candidates = [m for m in candidates if m.max_tokens >= 512] or candidates

    if budget == "paid":
        api_models = [m for m in candidates if m.size_mb == "API"]
        if api_models:
            return api_models[0] if speed_priority else api_models[-1]

    if speed_priority:
        return min(candidates, key=lambda m: m.dims)
    else:
        return max(candidates, key=lambda m: (m.dims if isinstance(m.dims, int) else 0))


# ══════════════════════════════════════════════════════
# ENCODER CLASS
# ══════════════════════════════════════════════════════

class EmbeddingEncoder:
    """
    Wrapper around sentence-transformers with:
    - Lazy model loading (download on first use)
    - Batch encoding with timing
    - L2 normalisation option
    - Similarity helpers
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model     = None    # lazy load

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[EmbeddingEncoder] Loading '{self.model_name}' (first run downloads model)...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dims(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: list[str],
        normalize: bool = True,
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode texts to embedding vectors.

        normalize=True → L2-normalised unit vectors (cosine sim = dot product).
        This is standard practice for semantic search.
        """
        t0 = time.perf_counter()
        vecs = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar=show_progress,
        )
        elapsed = time.perf_counter() - t0
        rate    = len(texts) / elapsed if elapsed > 0 else float("inf")
        print(f"[encode] {len(texts)} texts → {vecs.shape} in {elapsed:.2f}s "
              f"({rate:.0f} texts/sec)")
        return vecs

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two texts."""
        a, b = self.encode([text_a, text_b])
        return float(np.dot(a, b))  # already normalised → dot = cosine

    def top_k_similar(
        self, query: str, corpus: list[str], k: int = 5
    ) -> list[tuple[str, float]]:
        """Find top-k most similar texts to query."""
        query_vec   = self.encode([query])[0]
        corpus_vecs = self.encode(corpus)
        scores      = corpus_vecs @ query_vec   # dot product for each row
        top_indices = scores.argsort()[::-1][:k]
        return [(corpus[i], float(scores[i])) for i in top_indices]


# ══════════════════════════════════════════════════════
# DENSE vs SPARSE VECTORS
# ══════════════════════════════════════════════════════

def dense_vs_sparse_comparison(texts: list[str]) -> dict:
    """
    Dense vectors  (Word2Vec, BERT, sentence-transformers):
      - Fixed low dimension (384–3072)
      - Every dimension has a value (mostly non-zero)
      - Captures SEMANTIC meaning
      - 'running shoes' ≈ 'jogging trainers'  ← dense can match this

    Sparse vectors (TF-IDF, BM25):
      - Very high dimension (equal to vocabulary size, often 50k–500k)
      - Mostly zeros — only vocab words present in the text are non-zero
      - Captures EXACT keyword presence
      - Fast, interpretable, great for keyword search

    Hybrid search (modern best practice): use BOTH.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf_vec = TfidfVectorizer(max_features=100)
    sparse = tfidf_vec.fit_transform(texts).toarray()

    # Sparsity: % of zero elements
    sparse_sparsity = (sparse == 0).sum() / sparse.size

    return {
        "n_texts":          len(texts),
        "sparse_shape":     sparse.shape,           # (n, vocab_size)
        "sparse_sparsity":  round(float(sparse_sparsity), 4),
        "sparse_nonzero_per_doc": round(float((sparse != 0).sum(axis=1).mean()), 1),
        "dense_shape":      f"({len(texts)}, 384)",  # after encoding
        "dense_sparsity":   "~0 (all dims have values)",
        "semantic_gap_example": {
            "sparse_match": "'dog' and 'canine' → ZERO overlap in TF-IDF",
            "dense_match":  "'dog' and 'canine' → ~0.85 cosine similarity",
        },
    }


# ══════════════════════════════════════════════════════
# BENCHMARK HELPER
# ══════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    model_name:     str
    dims:           int
    encode_time_s:  float
    texts_per_sec:  float
    sample_similarities: list[tuple[str, str, float]] = field(default_factory=list)


def benchmark_model(
    model_name: str = "all-MiniLM-L6-v2",
    texts: list[str] | None = None,
) -> BenchmarkResult:
    """
    Benchmark an embedding model on a sample corpus.
    Use this to decide whether to upgrade from MiniLM to a heavier model.
    """
    if texts is None:
        texts = [
            "machine learning enables predictive analytics",
            "python is the dominant language for data science",
            "neural networks learn hierarchical representations",
            "data engineering builds scalable data pipelines",
            "vector databases store and query embeddings efficiently",
        ] * 20   # 100 texts

    enc   = EmbeddingEncoder(model_name)
    t0    = time.perf_counter()
    vecs  = enc.encode(texts)
    elapsed = time.perf_counter() - t0

    # Sample similarity pairs (first 4 texts)
    pairs = [
        (texts[0], texts[1]),
        (texts[0], texts[2]),
        (texts[1], texts[3]),
    ]
    sims = [
        (a[:40], b[:40], round(float(vecs[texts.index(a)] @ vecs[texts.index(b)]), 4))
        for a, b in pairs
    ]

    return BenchmarkResult(
        model_name    = model_name,
        dims          = vecs.shape[1],
        encode_time_s = round(elapsed, 3),
        texts_per_sec = round(len(texts) / elapsed, 1),
        sample_similarities = sims,
    )


# ══════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    import pprint

    print("=" * 60)
    print("EMBEDDING MODELS DEMO")
    print("=" * 60)

    # Model catalogue
    print("\n[1] Embedding Model Catalogue")
    print(f"{'Model':<50} {'Dims':>5} {'Tokens':>7} {'Speed':>8} {'MB':>6}")
    print("─" * 80)
    for m in MODEL_CATALOGUE:
        mb = str(m.size_mb)
        print(f"{m.name:<50} {m.dims:>5} {m.max_tokens:>7} {m.speed:>8} {mb:>6}")

    # Model selection
    print("\n[2] Model Selection Guide")
    fast_model = model_selection_guide(speed_priority=True)
    qual_model = model_selection_guide(speed_priority=False)
    multi_model = model_selection_guide(multilingual=True)
    print(f"  Speed priority:       {fast_model.name}")
    print(f"  Quality priority:     {qual_model.name}")
    print(f"  Multilingual:         {multi_model.name}")

    # Dense vs sparse
    sample_texts = [
        "The dog ran across the field.",
        "A canine sprinted through the meadow.",
        "Machine learning models process data.",
        "Python pandas data wrangling essentials.",
    ]
    print("\n[3] Dense vs Sparse Vectors")
    pprint.pprint(dense_vs_sparse_comparison(sample_texts))

    # Encode and benchmark
    print("\n[4] Encoding + Benchmark (all-MiniLM-L6-v2)")
    enc = EmbeddingEncoder("all-MiniLM-L6-v2")
    vecs = enc.encode(sample_texts)
    print(f"  Shape: {vecs.shape}")
    print(f"  Norms: {np.linalg.norm(vecs, axis=1).round(6).tolist()} (should all be ~1.0)")

    # Semantic similarity
    print("\n[5] Semantic Similarity")
    pairs = [
        ("The dog ran across the field.", "A canine sprinted through the meadow."),
        ("The dog ran across the field.", "Machine learning models process data."),
    ]
    for a, b in pairs:
        score = enc.similarity(a, b)
        print(f"  [{score:.4f}] '{a[:40]}' ↔ '{b[:40]}'")

    # Top-k similar
    print("\n[6] Top-K Semantic Search")
    query   = "deep learning neural networks"
    corpus  = [
        "Neural networks have multiple hidden layers.",
        "Python syntax is clean and readable.",
        "Deep learning models require GPUs.",
        "Embeddings represent semantic meaning.",
        "Convolutional networks excel at images.",
    ]
    results = enc.top_k_similar(query, corpus, k=3)
    for doc, score in results:
        print(f"  [{score:.4f}] {doc}")
