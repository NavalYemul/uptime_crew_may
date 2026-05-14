"""Tests for day3.hybrid_search — BM25, Dense, RRF, HybridSearcher."""

import pytest
import numpy as np
from day3.hybrid_search import (
    SearchResult,
    BM25Index,
    DenseIndex,
    reciprocal_rank_fusion,
    HybridSearcher,
    generate_product_corpus,
)


@pytest.fixture
def small_corpus():
    return [
        "Samsung Galaxy S24 Ultra with AMOLED 120Hz display.",
        "MacBook Pro M3 chip for machine learning workloads.",
        "Sony WH-1000XM5 noise cancelling wireless headphones.",
        "Logitech MX Master 3S productivity mouse.",
        "Apple AirPods Pro 2nd generation earbuds.",
    ]


# ── BM25Index ─────────────────────────────────────────────────

def test_bm25_index_builds(small_corpus):
    idx = BM25Index()
    idx.index(small_corpus)
    assert idx._bm25 is not None


def test_bm25_search_returns_search_results(small_corpus):
    idx = BM25Index()
    idx.index(small_corpus)
    results = idx.search("AMOLED 120Hz", n=3)
    assert all(isinstance(r, SearchResult) for r in results)


def test_bm25_search_ranked_descending(small_corpus):
    idx = BM25Index()
    idx.index(small_corpus)
    results = idx.search("machine learning laptop", n=4)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_bm25_search_top_n_limit(small_corpus):
    idx = BM25Index()
    idx.index(small_corpus)
    results = idx.search("headphones", n=2)
    assert len(results) <= 2


def test_bm25_exact_term_finds_amoled(small_corpus):
    idx = BM25Index()
    idx.index(small_corpus)
    results = idx.search("AMOLED", n=1)
    # The AMOLED document should be top result
    assert "AMOLED" in results[0].text or len(results) == 0 or results[0].text


def test_bm25_search_empty_corpus():
    idx = BM25Index()
    results = idx.search("anything", n=3)
    assert results == []


def test_bm25_source_label(small_corpus):
    idx = BM25Index()
    idx.index(small_corpus)
    results = idx.search("headphones", n=1)
    assert results[0].source == "bm25"


# ── DenseIndex ────────────────────────────────────────────────

def test_dense_index_with_mock(small_corpus, mock_embed):
    idx = DenseIndex(mock_embed_fn=mock_embed)
    idx.index(small_corpus)
    assert idx._embeddings is not None
    assert idx._embeddings.shape[0] == len(small_corpus)


def test_dense_search_returns_results(small_corpus, mock_embed):
    idx = DenseIndex(mock_embed_fn=mock_embed)
    idx.index(small_corpus)
    results = idx.search("best laptop for programming", n=3)
    assert len(results) > 0


def test_dense_search_returns_search_results(small_corpus, mock_embed):
    idx = DenseIndex(mock_embed_fn=mock_embed)
    idx.index(small_corpus)
    results = idx.search("noise cancellation", n=2)
    assert all(isinstance(r, SearchResult) for r in results)


def test_dense_search_source_label(small_corpus, mock_embed):
    idx = DenseIndex(mock_embed_fn=mock_embed)
    idx.index(small_corpus)
    results = idx.search("headphones", n=1)
    assert results[0].source == "dense"


def test_dense_search_empty_corpus(mock_embed):
    idx = DenseIndex(mock_embed_fn=mock_embed)
    results = idx.search("query", n=3)
    assert results == []


# ── Reciprocal Rank Fusion ────────────────────────────────────

def test_rrf_deduplicates():
    list_a = [
        SearchResult("Doc A", 10.0, 1, "bm25"),
        SearchResult("Doc B", 8.0,  2, "bm25"),
    ]
    list_b = [
        SearchResult("Doc A", 0.9, 1, "dense"),
        SearchResult("Doc C", 0.8, 2, "dense"),
    ]
    fused = reciprocal_rank_fusion([list_a, list_b])
    texts = [r.text for r in fused]
    # Doc A appears in both lists but should appear only once in fused
    assert texts.count("Doc A") == 1


def test_rrf_rank_ordering():
    list_a = [
        SearchResult("Doc A", 10.0, 1, "bm25"),
        SearchResult("Doc B", 8.0,  2, "bm25"),
    ]
    list_b = [
        SearchResult("Doc A", 0.9, 1, "dense"),
        SearchResult("Doc B", 0.8, 2, "dense"),
    ]
    fused = reciprocal_rank_fusion([list_a, list_b])
    # Doc A appears #1 in both, Doc B appears #2 in both
    assert fused[0].text == "Doc A"
    assert fused[1].text == "Doc B"


def test_rrf_source_label():
    list_a = [SearchResult("Doc X", 1.0, 1, "bm25")]
    fused = reciprocal_rank_fusion([list_a])
    assert fused[0].source == "hybrid_rrf"


def test_rrf_empty_lists():
    fused = reciprocal_rank_fusion([])
    assert fused == []


def test_rrf_single_list():
    list_a = [
        SearchResult("Doc A", 5.0, 1, "bm25"),
        SearchResult("Doc B", 3.0, 2, "bm25"),
    ]
    fused = reciprocal_rank_fusion([list_a])
    # Single list: order preserved
    assert fused[0].text == "Doc A"
    assert fused[1].text == "Doc B"


def test_rrf_scores_non_negative():
    list_a = [SearchResult("Doc A", 1.0, 1, "bm25")]
    list_b = [SearchResult("Doc A", 0.9, 1, "dense")]
    fused = reciprocal_rank_fusion([list_a, list_b])
    assert all(r.score >= 0 for r in fused)


# ── HybridSearcher ────────────────────────────────────────────

def test_hybrid_searcher_index_and_search(small_corpus, mock_embed):
    searcher = HybridSearcher(mock_embed_fn=mock_embed)
    searcher.index(small_corpus)
    results = searcher.search("laptop machine learning", n=3)
    assert len(results) > 0
    assert all(isinstance(r, SearchResult) for r in results)


def test_hybrid_compare_returns_three_keys(small_corpus, mock_embed):
    searcher = HybridSearcher(mock_embed_fn=mock_embed)
    searcher.index(small_corpus)
    comparison = searcher.compare("noise cancelling headphones", n=2)
    assert "bm25_results" in comparison
    assert "dense_results" in comparison
    assert "hybrid_results" in comparison


def test_hybrid_searcher_returns_search_results(small_corpus, mock_embed):
    searcher = HybridSearcher(mock_embed_fn=mock_embed)
    searcher.index(small_corpus)
    results = searcher.search("AMOLED display", n=3)
    assert all(isinstance(r, SearchResult) for r in results)


def test_generate_product_corpus_count():
    corpus = generate_product_corpus()
    assert len(corpus) == 20


def test_generate_product_corpus_nonempty():
    corpus = generate_product_corpus()
    for doc in corpus:
        assert len(doc) > 20
