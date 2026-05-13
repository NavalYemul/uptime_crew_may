"""
vector_store.py — Vector Databases, HNSW, Distance Metrics & Semantic Search
==============================================================================
Covers: ChromaDB (local), HNSW algorithm, cosine/euclidean/dot product,
        semantic search vs keyword search, 100+ document indexing.

Run:  python -m day2.vector_store
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# ══════════════════════════════════════════════════════
# CONCEPT: VECTOR DATABASES
# ══════════════════════════════════════════════════════
"""
WHY vector databases?

Traditional databases store and query structured values (numbers, strings).
They cannot answer: "find the 5 most similar paragraphs to this question."

Vector databases:
  1. Store high-dimensional vectors (embeddings)
  2. Build an index for fast nearest-neighbour search
  3. Return similar vectors efficiently — even in a 1B-vector collection

Popular choices:
  Chroma    — local, in-memory or file-backed. Great for dev/prototyping.
  pgvector  — PostgreSQL extension. Use when you already run Postgres.
  Weaviate  — managed or self-hosted. Built-in BM25 hybrid search.
  Pinecone  — fully managed SaaS. Zero infra. Pay per vector.
  Databricks Vector Search — native Delta Lake integration for production RAG.
  Qdrant    — self-hosted, high-performance, excellent filtering.

TODAY: we use Chroma (in-memory) — zero config, zero API keys, runs anywhere.
"""


# ══════════════════════════════════════════════════════
# DISTANCE METRICS
# ══════════════════════════════════════════════════════

def distance_metrics_demo() -> dict:
    """
    Three distance metrics for vector similarity:

    Cosine similarity: measures ANGLE between vectors, ignores magnitude.
      → Best for semantic similarity (embeddings are L2-normalised).
      → Range [-1, 1]. Higher = more similar.
      → "cat" and "cats" are close in direction even if magnitudes differ.

    Euclidean distance: straight-line distance in vector space.
      → Sensitive to magnitude. Commonly used in clustering (k-means).
      → Range [0, ∞). Lower = more similar.
      → Problem: long documents have larger magnitude → inflate distance.

    Dot product: dot(A, B) = |A||B| cos(θ).
      → Combines angle AND magnitude.
      → Used when you want to favour "prominent" (high-magnitude) vectors.
      → For L2-normalised vectors: dot product == cosine similarity.

    Rule of thumb: ALWAYS L2-normalise your embeddings and use cosine/dot.
    """
    # Sample embedding pairs
    a = np.array([0.6, 0.8, 0.0])           # unit vector
    b = np.array([0.8, 0.6, 0.0])           # unit vector, close to a
    c = np.array([0.0, 0.0, 1.0])           # orthogonal to both
    d = np.array([1.2, 1.6, 0.0])           # same direction as a, 2x magnitude

    def cosine(x, y):
        return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))

    def euclidean(x, y):
        return float(np.linalg.norm(x - y))

    def dot(x, y):
        return float(np.dot(x, y))

    return {
        "pair_a_b": {
            "description":  "Similar direction",
            "cosine":       round(cosine(a, b),    4),
            "euclidean":    round(euclidean(a, b), 4),
            "dot_product":  round(dot(a, b),       4),
        },
        "pair_a_c": {
            "description":  "Orthogonal (unrelated)",
            "cosine":       round(cosine(a, c),    4),
            "euclidean":    round(euclidean(a, c), 4),
            "dot_product":  round(dot(a, c),       4),
        },
        "pair_a_d": {
            "description":  "Same direction, different magnitude",
            "cosine":       round(cosine(a, d),    4),     # should be 1.0
            "euclidean":    round(euclidean(a, d), 4),     # non-zero (magnitude matters)
            "dot_product":  round(dot(a, d),       4),     # larger than cosine
        },
    }


# ══════════════════════════════════════════════════════
# HNSW CONCEPT
# ══════════════════════════════════════════════════════

@dataclass
class HNSWConfig:
    """
    HNSW (Hierarchical Navigable Small World) — the algorithm powering
    fast ANN search in Chroma, Weaviate, Qdrant, FAISS, and most modern DBs.

    Exact search: compare query to ALL n vectors → O(n). Slow at scale.
    HNSW: graph-based approximate nearest neighbour → O(log n). Fast.

    How it works:
      Layer 0:  contains ALL vectors with short-range links.
      Layer 1:  a subset with medium-range "express" links.
      Layer 2+: smaller subsets with long-range links (skip list style).

      Search: enter at top layer, greedily navigate to closest node,
              descend to next layer, repeat until Layer 0.
              Result: approximately nearest neighbours in milliseconds.

    Key parameters:
      M:           number of links per node. Higher M → better recall,
                   more memory. Typical: 16–64.
      ef_construction: size of candidate list during index build.
                   Higher → better index quality, slower build. Typical: 100–400.
      ef_search:   size of candidate list during query.
                   Higher → better recall, slower query. Typical: 50–200.
    """
    M:               int   = 16
    ef_construction: int   = 200
    ef_search:       int   = 100
    space:           str   = "cosine"   # "cosine" | "l2" | "ip" (inner product)

    def recall_estimate(self) -> str:
        if self.ef_search >= 200:
            return "~99%"
        elif self.ef_search >= 100:
            return "~95–98%"
        elif self.ef_search >= 50:
            return "~90–95%"
        else:
            return "~80–90%"


# ══════════════════════════════════════════════════════
# VECTOR STORE WRAPPER
# ══════════════════════════════════════════════════════

class ChromaVectorStore:
    """
    In-memory ChromaDB vector store.
    No server required. Persists nothing — clears on process exit.
    Perfect for teaching and prototyping.

    Production: switch to PersistentClient("./chroma_db") or
                use Databricks Vector Search via LangChain.
    """

    def __init__(
        self,
        collection_name: str = "day2_docs",
        embedding_model:  str = "all-MiniLM-L6-v2",
    ):
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        self.client    = chromadb.Client()   # in-memory
        self.ef        = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        self.collection = self.client.get_or_create_collection(
            name               = collection_name,
            embedding_function = self.ef,
            metadata           = {"hnsw:space": "cosine"},
        )
        self._model_name = embedding_model

    def add_documents(
        self,
        documents: list[str],
        ids:       list[str] | None = None,
        metadatas: list[dict] | None = None,
    ) -> None:
        """
        Add documents to the collection.
        ChromaDB automatically:
          1. Embeds each document using the embedding function
          2. Stores the vector + original text + metadata
          3. Builds HNSW index
        """
        if ids is None:
            ids = [f"doc_{i:05d}" for i in range(len(documents))]

        # ChromaDB has a batch size limit — chunk if needed
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs  = documents[i:i + batch_size]
            batch_ids   = ids[i:i + batch_size]
            batch_meta  = metadatas[i:i + batch_size] if metadatas else None
            self.collection.add(
                documents = batch_docs,
                ids       = batch_ids,
                metadatas = batch_meta,
            )

    def search(
        self,
        query:      str,
        n_results:  int = 5,
        where:      dict | None = None,   # metadata filter
    ) -> list[dict]:
        """
        Semantic search: embed the query, find closest document vectors.
        Optional `where` filter: e.g. {"topic": "nlp"} only searches NLP docs.
        """
        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results":   min(n_results, self.collection.count()),
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        return [
            {
                "rank":       i + 1,
                "id":         results["ids"][0][i],
                "document":   results["documents"][0][i],
                "distance":   round(float(results["distances"][0][i]), 4),
                "score":      round(1 - float(results["distances"][0][i]), 4),
                "metadata":   results["metadatas"][0][i] if results["metadatas"] else {},
            }
            for i in range(len(results["ids"][0]))
        ]

    def count(self) -> int:
        return self.collection.count()

    def get_all_ids(self) -> list[str]:
        return self.collection.get()["ids"]


# ══════════════════════════════════════════════════════
# SEMANTIC SEARCH vs KEYWORD SEARCH
# ══════════════════════════════════════════════════════

class KeywordSearch:
    """BM25-style TF-IDF keyword search for comparison with semantic search."""

    def __init__(self):
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._corpus: list[str] = []
        self._matrix = None

    def index(self, documents: list[str]) -> None:
        self._corpus = documents
        self._matrix = self._vectorizer.fit_transform(documents)

    def search(self, query: str, n: int = 5) -> list[dict]:
        from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
        q_vec  = self._vectorizer.transform([query])
        scores = sk_cosine(q_vec, self._matrix)[0]
        top_i  = scores.argsort()[::-1][:n]
        return [
            {"rank": r + 1, "document": self._corpus[i], "score": round(float(scores[i]), 4)}
            for r, i in enumerate(top_i)
        ]


def semantic_vs_keyword_demo(
    corpus: list[str],
    queries: list[str],
    vector_store: ChromaVectorStore,
) -> list[dict]:
    """
    Side-by-side comparison of semantic search vs keyword (TF-IDF) search.

    KEY difference:
      Keyword: matches exact words. "canine" does NOT match query "dog".
      Semantic: matches meaning.   "canine" DOES match query "dog" (same concept).

    Hybrid search (best of both worlds): combine scores from both methods.
    Used in production by Weaviate, Azure AI Search, Databricks Vector Search.
    """
    kw = KeywordSearch()
    kw.index(corpus)

    results = []
    for q in queries:
        semantic_hits = vector_store.search(q, n_results=3)
        keyword_hits  = kw.search(q, n=3)
        results.append({
            "query":    q,
            "semantic": [h["document"][:70] for h in semantic_hits],
            "keyword":  [h["document"][:70] for h in keyword_hits],
        })
    return results


# ══════════════════════════════════════════════════════
# CORPUS GENERATOR (100+ documents)
# ══════════════════════════════════════════════════════

def generate_corpus(n: int = 120) -> list[dict]:
    """
    Generate a synthetic corpus of 120 AI/ML documents with topics and metadata.
    Used to demonstrate indexing at scale.
    """
    templates = [
        # Machine Learning
        ("ml",       "Supervised learning uses labelled data to train predictive models that generalise to unseen examples."),
        ("ml",       "Random forests build hundreds of decision trees and aggregate their predictions to reduce variance."),
        ("ml",       "Gradient boosting sequentially fits weak learners, each correcting the errors of the previous one."),
        ("ml",       "Cross-validation estimates the true generalisation error by rotating the held-out evaluation set."),
        ("ml",       "Regularisation techniques like L1 and L2 penalise large weights to prevent overfitting."),
        # Deep Learning
        ("dl",       "Convolutional neural networks apply learnable filters to detect local patterns in image data."),
        ("dl",       "Transformer architecture uses self-attention to model dependencies across the entire input sequence."),
        ("dl",       "Transfer learning fine-tunes a pre-trained model on a smaller domain-specific dataset."),
        ("dl",       "Backpropagation computes gradients layer by layer using the chain rule of calculus."),
        ("dl",       "Dropout randomly deactivates neurons during training to act as an ensemble of networks."),
        # NLP
        ("nlp",      "Tokenisation converts raw text into sub-word units that map to vocabulary indices."),
        ("nlp",      "Named entity recognition identifies people, organisations, and locations in text."),
        ("nlp",      "Sentiment analysis classifies text as positive, negative, or neutral in opinion."),
        ("nlp",      "BERT uses bidirectional attention to understand context from both left and right."),
        ("nlp",      "Large language models are pre-trained on trillions of tokens and fine-tuned for tasks."),
        # Embeddings & Vector Search
        ("vectors",  "Sentence embeddings encode entire sentences into dense fixed-size vectors."),
        ("vectors",  "Cosine similarity measures the angle between two embedding vectors."),
        ("vectors",  "HNSW builds a hierarchical graph for efficient approximate nearest-neighbour search."),
        ("vectors",  "Dense vectors capture semantic meaning; sparse vectors capture keyword frequency."),
        ("vectors",  "Hybrid search combines dense and sparse retrieval for best-of-both-worlds results."),
        # Data Engineering
        ("data_eng", "ETL extracts, transforms, then loads data into a target warehouse."),
        ("data_eng", "ELT loads raw data first, then transforms inside the warehouse using SQL."),
        ("data_eng", "Apache Spark processes data in parallel across a cluster of machines."),
        ("data_eng", "Delta Lake adds ACID transactions to Parquet files in object storage."),
        ("data_eng", "Data pipelines can be batch or streaming depending on latency requirements."),
        # RAG
        ("rag",      "Retrieval Augmented Generation grounds LLM answers in retrieved document context."),
        ("rag",      "Chunking splits long documents into overlapping segments for better retrieval."),
        ("rag",      "Faithfulness measures whether the answer is supported by the retrieved context."),
        ("rag",      "Answer relevancy measures how directly the response addresses the original question."),
        ("rag",      "Context precision measures the signal-to-noise ratio in the retrieved documents."),
    ]

    # Expand to n documents by varying them slightly
    corpus = []
    for idx in range(n):
        topic, base_text = templates[idx % len(templates)]
        variation = idx // len(templates)
        text = base_text if variation == 0 else (
            base_text.replace(".", f". (Extended note {variation}: "
                              f"This is a key concept in modern {topic} systems.)")
        )
        corpus.append({
            "text":     text,
            "topic":    topic,
            "doc_id":   f"doc_{idx:04d}",
            "priority": "high" if idx % 5 == 0 else "normal",
        })
    return corpus


# ══════════════════════════════════════════════════════
# INDEX VALIDATION
# ══════════════════════════════════════════════════════

def validate_index(store: ChromaVectorStore, expected_count: int) -> dict:
    """
    Validate that the index is correctly populated.
    Run these checks after every bulk ingestion.
    """
    actual = store.count()
    all_ids = store.get_all_ids()
    id_dupes = len(all_ids) - len(set(all_ids))

    # Spot-check: query should return results
    test_query  = "machine learning models"
    test_result = store.search(test_query, n_results=1)
    retrieval_ok = len(test_result) > 0 and test_result[0]["score"] > 0

    return {
        "expected":     expected_count,
        "actual":       actual,
        "count_ok":     actual == expected_count,
        "duplicate_ids": id_dupes,
        "retrieval_ok": retrieval_ok,
        "top_result_score": test_result[0]["score"] if test_result else 0,
    }


# ══════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    import pprint

    print("=" * 60)
    print("VECTOR SEARCH DEMO")
    print("=" * 60)

    print("\n[1] Distance Metrics")
    pprint.pprint(distance_metrics_demo())

    print("\n[2] HNSW Config")
    cfg = HNSWConfig(M=16, ef_construction=200, ef_search=100)
    print(f"  M={cfg.M}, ef_construction={cfg.ef_construction}, "
          f"ef_search={cfg.ef_search}, recall≈{cfg.recall_estimate()}")

    print("\n[3] Building ChromaDB index with 120 documents...")
    corpus_dicts = generate_corpus(120)
    documents    = [d["text"]  for d in corpus_dicts]
    ids          = [d["doc_id"] for d in corpus_dicts]
    metadatas    = [{"topic": d["topic"], "priority": d["priority"]} for d in corpus_dicts]

    t0    = time.perf_counter()
    store = ChromaVectorStore("demo_collection")
    store.add_documents(documents, ids=ids, metadatas=metadatas)
    elapsed = time.perf_counter() - t0
    print(f"  Indexed {store.count()} docs in {elapsed:.2f}s")

    print("\n[4] Index Validation")
    pprint.pprint(validate_index(store, expected_count=120))

    print("\n[5] Semantic Search")
    queries = [
        "how do transformer models work",
        "database transactions ACID",
        "measuring retrieval quality in RAG",
    ]
    for q in queries:
        hits = store.search(q, n_results=3)
        print(f"\n  Query: '{q}'")
        for h in hits:
            print(f"    [{h['score']:.4f}] [{h['metadata'].get('topic','')}] "
                  f"{h['document'][:70]}")

    print("\n[6] Filtered Search (only 'rag' topic)")
    hits = store.search("evaluation metrics", n_results=3, where={"topic": "rag"})
    for h in hits:
        print(f"  [{h['score']:.4f}] {h['document'][:70]}")

    print("\n[7] Semantic vs Keyword Search")
    comparison_queries = [
        "canine runs across field",        # semantic wins ("dog" synonym)
        "HNSW approximate nearest neighbour",  # keyword wins (exact tech term)
    ]
    small_corpus = [d["text"] for d in corpus_dicts[:30]]
    small_store  = ChromaVectorStore("small_demo")
    small_store.add_documents(small_corpus)

    results = semantic_vs_keyword_demo(small_corpus, comparison_queries, small_store)
    for r in results:
        print(f"\n  Query: '{r['query']}'")
        print(f"  Semantic: {r['semantic'][0][:70]}")
        print(f"  Keyword:  {r['keyword'][0][:70]}")
