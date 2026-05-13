"""
test_nlp.py — NLP + Embeddings tests
"""

import numpy as np
import pytest

from day1.nlp_basics import (
    CORPUS,
    bag_of_words,
    full_preprocess,
    normalize_tokens,
    remove_stopwords,
    tfidf_vectors,
    tokenize_sentences,
    tokenize_words,
)
from day1.embeddings import (
    cosine_sim,
    pairwise_similarity,
    project_documents_2d,
    semantic_search,
)


# ─────────────────────────────────────────────
# TOKENIZATION
# ─────────────────────────────────────────────

def test_word_tokenize_basic():
    tokens = tokenize_words("Hello world! How are you?")
    assert "Hello" in tokens
    assert "world" in tokens
    assert len(tokens) >= 5


def test_sentence_tokenize_count():
    text = "First sentence. Second sentence. Third sentence."
    sents = tokenize_sentences(text)
    assert len(sents) == 3


def test_normalize_lowercases():
    tokens = normalize_tokens(["Hello", "World", "Python"])
    assert all(t == t.lower() for t in tokens)


def test_normalize_removes_punctuation():
    tokens = normalize_tokens(["!", "?", ".", "hello"])
    assert "!" not in tokens
    assert "?" not in tokens
    assert "hello" in tokens


def test_stopword_removal_removes_the():
    tokens = ["the", "cat", "sat", "on", "the", "mat"]
    filtered = remove_stopwords(tokens)
    assert "the" not in filtered
    assert "on" not in filtered


def test_stopword_removal_preserves_content_words():
    tokens = ["machine", "learning", "python", "the", "is"]
    filtered = remove_stopwords(tokens)
    assert "machine" in filtered
    assert "learning" in filtered
    assert "python" in filtered


def test_full_preprocess_removes_stopwords():
    tokens = full_preprocess("The quick brown fox jumps over the lazy dog")
    assert "the" not in tokens
    assert len(tokens) > 0


def test_full_preprocess_returns_list_of_strings():
    tokens = full_preprocess("Machine learning is amazing!")
    assert isinstance(tokens, list)
    assert all(isinstance(t, str) for t in tokens)


# ─────────────────────────────────────────────
# BAG OF WORDS / TF-IDF
# ─────────────────────────────────────────────

def test_bow_matrix_row_count():
    bow, vocab = bag_of_words(CORPUS)
    assert bow.shape[0] == len(CORPUS)


def test_bow_vocab_max_features():
    bow, vocab = bag_of_words(CORPUS)
    assert len(vocab) <= 20


def test_tfidf_matrix_shape():
    tfidf, vocab = tfidf_vectors(CORPUS)
    assert tfidf.shape[0] == len(CORPUS)
    assert tfidf.shape[1] <= 20


def test_tfidf_values_bounded():
    tfidf, _ = tfidf_vectors(CORPUS)
    assert tfidf.min() >= 0.0
    assert tfidf.max() <= 1.0


# ─────────────────────────────────────────────
# COSINE SIMILARITY
# ─────────────────────────────────────────────

def test_cosine_identical_vectors():
    v = [1.0, 2.0, 3.0]
    assert cosine_sim(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_orthogonal_vectors():
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    assert cosine_sim(a, b) == pytest.approx(0.0, abs=1e-6)


def test_cosine_opposite_vectors():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert cosine_sim(a, b) == pytest.approx(-1.0, abs=1e-6)


def test_cosine_range_random():
    rng = np.random.default_rng(42)
    for _ in range(30):
        a = rng.standard_normal(10).tolist()
        b = rng.standard_normal(10).tolist()
        s = cosine_sim(a, b)
        assert -1.0 - 1e-6 <= s <= 1.0 + 1e-6


def test_cosine_zero_vector_returns_zero():
    assert cosine_sim([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == 0.0


# ─────────────────────────────────────────────
# SEMANTIC SEARCH
# ─────────────────────────────────────────────

def test_semantic_search_returns_correct_count(nlp_corpus):
    results = semantic_search("machine learning", nlp_corpus, top_n=2)
    assert len(results) == 2


def test_semantic_search_sorted_by_score(nlp_corpus):
    results = semantic_search("neural network deep learning", nlp_corpus, top_n=3)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_pairwise_similarity_shape(nlp_corpus):
    mat = pairwise_similarity(nlp_corpus)
    n = len(nlp_corpus)
    assert mat.shape == (n, n)


def test_pairwise_similarity_diagonal_ones(nlp_corpus):
    mat = pairwise_similarity(nlp_corpus)
    np.testing.assert_array_almost_equal(np.diag(mat), np.ones(len(nlp_corpus)))


def test_pairwise_similarity_symmetric(nlp_corpus):
    mat = pairwise_similarity(nlp_corpus)
    np.testing.assert_array_almost_equal(mat, mat.T)


def test_project_documents_2d_shape(nlp_corpus):
    coords = project_documents_2d(nlp_corpus)
    assert coords.shape == (len(nlp_corpus), 2)
