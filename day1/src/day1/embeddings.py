"""
embeddings.py — Word Embeddings & Semantic Similarity
=======================================================
Covers: word embeddings, Word2Vec, GloVe (conceptual), cosine similarity,
        vector space model, semantic search with TF-IDF.

Run:  python -m day1.embeddings
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("[NOTE] gensim not installed → install with: uv add gensim")


# ══════════════════════════════════════════════════════
# CONCEPT: WHY EMBEDDINGS?
# ══════════════════════════════════════════════════════
"""
PROBLEM with Bag of Words (BoW):
  BoW is a sparse, high-dimensional, UNORDERED representation.
  Vocabulary of 50,000 words → each doc = 50,000-dim vector, mostly zeros.
  "king" and "queen" are at RANDOM columns — they know nothing about each other.

  Can BoW answer: "Paris is to France as Berlin is to ____?"  → NO.

SOLUTION: Dense Word Embeddings
  Every word → a dense float vector (e.g., 100 or 300 dims).
  Words with SIMILAR MEANING are CLOSE in vector space.

  king  ≈ [0.12, -0.45, 0.78, ...]
  queen ≈ [0.15, -0.42, 0.81, ...]   ← very similar direction

  Famous arithmetic: king - man + woman ≈ queen   (Word2Vec, 2013)

Word2Vec (Google, 2013):
  Trains a shallow neural network to predict context words.
  Two architectures:
    CBOW       → predict centre word from context
    Skip-gram  → predict context words from centre word (better for rare words)

GloVe (Stanford, 2014):
  "Global Vectors" — uses global word co-occurrence statistics.
  Often better for analogy tasks.

fastText (Facebook):
  Subword embeddings → handles out-of-vocabulary words.

Today we use TF-IDF for document similarity and
gensim Word2Vec on a toy corpus to show the concept.
"""


# ══════════════════════════════════════════════════════
# CORPUS
# ══════════════════════════════════════════════════════

TRAINING_SENTENCES: list[str] = [
    "machine learning is a type of artificial intelligence",
    "deep learning is a subset of machine learning",
    "neural networks are the foundation of deep learning",
    "natural language processing helps computers understand text",
    "word embeddings represent words as numerical vectors",
    "python is popular for machine learning and data science",
    "scikit-learn provides tools for classical machine learning",
    "tensorflow and pytorch are popular deep learning frameworks",
    "data preprocessing is important for machine learning models",
    "clustering and classification are machine learning tasks",
    "supervised learning uses labelled training data",
    "unsupervised learning discovers hidden patterns in data",
    "reinforcement learning trains agents through reward signals",
    "tokenization splits text into individual tokens",
    "semantic similarity measures closeness of meaning between texts",
    "cosine similarity computes the angle between two vectors",
    "word2vec learns embeddings from large text corpora",
    "glove embeddings are trained on global co-occurrence statistics",
    "embedding vectors capture syntactic and semantic relationships",
    "bert and gpt use transformer architecture for language modelling",
]

DOCUMENTS: list[str] = [
    "Python is excellent for machine learning and data science",
    "Java is a popular object-oriented programming language",
    "Deep learning uses neural networks for image recognition tasks",
    "Natural language processing enables chatbots and translation",
    "Data engineering involves building robust scalable data pipelines",
    "Machine learning models require careful feature engineering",
    "Neural networks learn hierarchical representations from raw data",
    "Text classification assigns predefined categories to documents",
]


# ══════════════════════════════════════════════════════
# COSINE SIMILARITY — manual implementation
# ══════════════════════════════════════════════════════

def cosine_sim(vec_a: list[float] | np.ndarray,
               vec_b: list[float] | np.ndarray) -> float:
    """
    Cosine similarity = dot(A, B) / (‖A‖ × ‖B‖)

    Returns a value in [-1, 1]:
      1.0  → same direction (identical meaning)
      0.0  → perpendicular (unrelated)
     -1.0  → opposite direction

    WHY use cosine and not Euclidean distance?
    "the cat sat" vs "the cat sat on the mat" — different magnitudes
    but similar DIRECTION (meaning) → cosine handles this correctly.

    Java equivalent:
        double dot = IntStream.range(0,n).mapToDouble(i -> a[i]*b[i]).sum();
        double normA = Math.sqrt(Arrays.stream(a).map(x->x*x).sum());
        ...
    """
    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def pairwise_similarity(docs: list[str]) -> np.ndarray:
    """Build N×N cosine similarity matrix from TF-IDF vectors."""
    vec = TfidfVectorizer(stop_words="english")
    tfidf = vec.fit_transform(docs)
    return cosine_similarity(tfidf)


def semantic_search(
    query: str, docs: list[str], top_n: int = 3
) -> list[tuple[str, float]]:
    """
    Find top-N documents most semantically similar to query.
    Simple but effective retrieval using TF-IDF + cosine similarity.
    Production systems use dense encoders (Sentence-BERT, etc.).
    """
    all_texts = [query] + docs
    vec = TfidfVectorizer(stop_words="english")
    vecs = vec.fit_transform(all_texts)

    query_vec = vecs[0]
    doc_vecs   = vecs[1:]
    scores = cosine_similarity(query_vec, doc_vecs)[0]

    ranked = sorted(zip(docs, scores.tolist()), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


# ══════════════════════════════════════════════════════
# WORD2VEC — gensim
# ══════════════════════════════════════════════════════

def train_word2vec(sentences: list[str]) -> "Word2Vec | None":
    """
    Train a tiny Word2Vec model on our corpus.

    Real models train on billions of sentences (Wikipedia, Common Crawl).
    We use 50 dimensions (real models: 100–300).

    Parameters:
      vector_size : embedding dimensionality
      window      : context window (words left + right of target)
      min_count   : ignore words with freq < this
      sg          : 0 = CBOW, 1 = Skip-gram
      epochs      : training passes over corpus
    """
    if not GENSIM_AVAILABLE:
        print("[Word2Vec] gensim not available, skipping.")
        return None

    tokenized = [s.lower().split() for s in sentences]
    model = Word2Vec(
        sentences=tokenized,
        vector_size=50,
        window=3,
        min_count=1,
        sg=1,
        epochs=300,
        seed=42,
    )
    print(f"[Word2Vec] Trained. Vocabulary: {len(model.wv)} unique words")
    return model


def explore_embeddings(model: "Word2Vec") -> None:
    """Show nearest neighbours and simple vector arithmetic."""
    wv = model.wv
    words_to_probe = ["learning", "machine", "neural", "language", "python", "data"]

    print("\n[Word2Vec] Nearest neighbours:")
    for word in words_to_probe:
        if word in wv:
            similar = wv.most_similar(word, topn=3)
            print(f"  '{word}' → {[(w, round(s, 3)) for w, s in similar]}")

    # Analogy: deep + learning ≈ ?
    if all(w in wv for w in ["deep", "learning", "machine"]):
        result = wv.most_similar(positive=["deep", "learning"], negative=["machine"], topn=3)
        print(f"\n[Word2Vec] deep + learning - machine ≈ {result}")


# ══════════════════════════════════════════════════════
# DOCUMENT PROJECTION (for visualisation)
# ══════════════════════════════════════════════════════

def project_documents_2d(docs: list[str]) -> np.ndarray:
    """Project TF-IDF document vectors to 2D via PCA."""
    vec = TfidfVectorizer(stop_words="english", max_features=50)
    tfidf = vec.fit_transform(docs).toarray()
    n_comp = min(2, tfidf.shape[0] - 1, tfidf.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    return pca.fit_transform(tfidf)


# ══════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("WORD EMBEDDINGS & COSINE SIMILARITY DEMO")
    print("=" * 60)

    # 1. Manual cosine similarity
    print("\n[Cosine Similarity]")
    print(f"  identical vectors  [1,2,3] vs [1,2,3] = {cosine_sim([1,2,3],[1,2,3]):.4f}")
    print(f"  different vectors  [1,2,3] vs [3,1,0] = {cosine_sim([1,2,3],[3,1,0]):.4f}")
    print(f"  orthogonal vectors [1,0,0] vs [0,1,0] = {cosine_sim([1,0,0],[0,1,0]):.4f}")

    # 2. Pairwise similarity matrix
    sim = pairwise_similarity(DOCUMENTS)
    print(f"\n[Pairwise Similarity Matrix] {sim.shape}")
    print(np.round(sim, 2))

    # 3. Semantic search
    query = "machine learning with Python"
    print(f"\n[Semantic Search] query: '{query}'")
    for doc, score in semantic_search(query, DOCUMENTS, top_n=3):
        print(f"  [{score:.3f}] {doc}")

    # 4. Word2Vec
    if GENSIM_AVAILABLE:
        print("\n[Word2Vec] Training on toy corpus (300 epochs)...")
        model = train_word2vec(TRAINING_SENTENCES)
        if model:
            explore_embeddings(model)

    # 5. 2D projection
    coords = project_documents_2d(DOCUMENTS)
    print(f"\n[PCA 2D Projection]")
    for doc, coord in zip(DOCUMENTS, coords):
        print(f"  ({coord[0]:6.3f}, {coord[1]:6.3f})  {doc[:55]}")
