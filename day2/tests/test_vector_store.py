"""test_vector_store.py — Vector store & similarity tests"""

import numpy as np
import pytest

from day2.vector_store import (
    distance_metrics_demo,
    HNSWConfig,
    ChromaVectorStore,
    KeywordSearch,
    generate_corpus,
    validate_index,
)


# ── Distance Metrics ──────────────────────────────────

def test_distance_metrics_keys():
    r = distance_metrics_demo()
    assert {"pair_a_b", "pair_a_c", "pair_a_d"} <= r.keys()


def test_cosine_similar_close_to_one():
    r = distance_metrics_demo()
    assert r["pair_a_b"]["cosine"] > 0.9


def test_cosine_orthogonal_is_zero():
    r = distance_metrics_demo()
    assert r["pair_a_c"]["cosine"] == pytest.approx(0.0, abs=1e-5)


def test_cosine_same_direction_is_one():
    r = distance_metrics_demo()
    assert r["pair_a_d"]["cosine"] == pytest.approx(1.0, abs=1e-4)


def test_euclidean_same_direction_nonzero():
    r = distance_metrics_demo()
    # Same direction but different magnitude → nonzero Euclidean
    assert r["pair_a_d"]["euclidean"] > 0


def test_dot_product_positive_for_similar():
    r = distance_metrics_demo()
    assert r["pair_a_b"]["dot_product"] > 0


# ── HNSW Config ───────────────────────────────────────

def test_hnsw_config_defaults():
    cfg = HNSWConfig()
    assert cfg.M == 16
    assert cfg.ef_construction == 200
    assert cfg.ef_search == 100
    assert cfg.space == "cosine"


def test_hnsw_recall_estimate_high_ef():
    cfg = HNSWConfig(ef_search=200)
    assert "99" in cfg.recall_estimate()


def test_hnsw_recall_estimate_low_ef():
    cfg = HNSWConfig(ef_search=30)
    assert "80" in cfg.recall_estimate()


# ── ChromaDB Vector Store ─────────────────────────────

@pytest.fixture
def small_store(small_corpus):
    store = ChromaVectorStore("test_small_store")
    store.add_documents(
        small_corpus,
        ids=[f"d{i}" for i in range(len(small_corpus))],
    )
    return store


def test_store_count(small_store, small_corpus):
    assert small_store.count() == len(small_corpus)


def test_store_search_returns_results(small_store):
    results = small_store.search("machine learning", n_results=3)
    assert len(results) == 3


def test_store_search_sorted_by_score(small_store):
    results = small_store.search("neural network deep learning", n_results=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_store_search_scores_bounded(small_store):
    results = small_store.search("any query", n_results=3)
    for r in results:
        assert 0.0 <= r["score"] <= 1.0


def test_store_search_has_required_keys(small_store):
    results = small_store.search("data pipeline", n_results=2)
    for r in results:
        assert {"rank", "id", "document", "distance", "score"} <= r.keys()


def test_store_get_all_ids(small_store, small_corpus):
    ids = small_store.get_all_ids()
    assert len(ids) == len(small_corpus)


# ── Corpus Generator ──────────────────────────────────

def test_generate_corpus_count():
    corpus = generate_corpus(120)
    assert len(corpus) == 120


def test_generate_corpus_structure():
    corpus = generate_corpus(10)
    for doc in corpus:
        assert {"text", "topic", "doc_id", "priority"} <= doc.keys()
        assert len(doc["text"]) > 10
        assert doc["topic"] in {"ml","dl","nlp","vectors","data_eng","rag"}


def test_generate_corpus_unique_ids():
    corpus = generate_corpus(60)
    ids = [d["doc_id"] for d in corpus]
    assert len(ids) == len(set(ids))


# ── Validate Index ────────────────────────────────────

def test_validate_index_pass(small_store, small_corpus):
    report = validate_index(small_store, expected_count=len(small_corpus))
    assert report["count_ok"] is True
    assert report["duplicate_ids"] == 0
    assert report["retrieval_ok"] is True


def test_validate_index_fail_wrong_count(small_store):
    report = validate_index(small_store, expected_count=9999)
    assert report["count_ok"] is False


# ── Keyword Search ────────────────────────────────────

def test_keyword_search_returns_results(small_corpus):
    kw = KeywordSearch()
    kw.index(small_corpus)
    results = kw.search("machine learning", n=3)
    assert len(results) == 3


def test_keyword_search_sorted(small_corpus):
    kw = KeywordSearch()
    kw.index(small_corpus)
    results = kw.search("vector embeddings similarity", n=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_keyword_search_exact_match_wins(small_corpus):
    # "HNSW" is an exact token in one doc — keyword should find it
    kw = KeywordSearch()
    kw.index(small_corpus)
    results = kw.search("HNSW nearest neighbour", n=3)
    top_doc = results[0]["document"]
    assert "HNSW" in top_doc or "hnsw" in top_doc.lower()
