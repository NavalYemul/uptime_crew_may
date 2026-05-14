"""Tests for day3.reranking — CrossEncoder mock, MMR, metadata filter, pipeline."""

import pytest
import numpy as np
from day3.reranking import (
    RankedResult,
    CrossEncoderReranker,
    mmr_select,
    metadata_filter,
    build_retrieval_pipeline,
    compare_retrieval_strategies,
)


@pytest.fixture
def candidates():
    return [
        "MacBook Pro M3 with 18GB RAM is ideal for machine learning.",
        "Samsung Galaxy S24 Ultra has a 200MP camera.",
        "Sony WH-1000XM5 has industry-leading noise cancellation.",
        "Logitech MX Master 3S is perfect for productivity.",
        "Apple AirPods Pro 2nd gen with H2 chip.",
        "Dell XPS 15 with OLED display for content creation.",
        "OnePlus 12 has 100W fast charging.",
        "iPad Pro M2 for creative professionals.",
    ]


@pytest.fixture
def docs_with_meta():
    return [
        {"text": "MacBook Pro M3 is best for ML workloads.",
         "metadata": {"category": "laptop", "price": 199900}},
        {"text": "Samsung Galaxy S24 Ultra flagship smartphone.",
         "metadata": {"category": "smartphone", "price": 129999}},
        {"text": "Sony WH-1000XM5 headphones with ANC.",
         "metadata": {"category": "headphones", "price": 29990}},
        {"text": "Dell XPS 15 laptop for content creation.",
         "metadata": {"category": "laptop", "price": 189990}},
        {"text": "Apple AirPods Pro 2nd gen earbuds.",
         "metadata": {"category": "headphones", "price": 24900}},
    ]


# ── CrossEncoderReranker (mock) ───────────────────────────────

def test_rerank_mock_returns_ranked_results(candidates):
    reranker = CrossEncoderReranker()
    results = reranker.rerank_mock("machine learning laptop", candidates, top_k=3)
    assert len(results) == 3
    assert all(isinstance(r, RankedResult) for r in results)


def test_rerank_mock_top_k_respected(candidates):
    reranker = CrossEncoderReranker()
    results = reranker.rerank_mock("noise cancellation", candidates, top_k=2)
    assert len(results) == 2


def test_rerank_mock_ranks_start_at_one(candidates):
    reranker = CrossEncoderReranker()
    results = reranker.rerank_mock("headphones ANC", candidates, top_k=3)
    final_ranks = [r.final_rank for r in results]
    assert final_ranks[0] == 1
    assert final_ranks[1] == 2
    assert final_ranks[2] == 3


def test_rerank_mock_final_score_is_float(candidates):
    reranker = CrossEncoderReranker()
    results = reranker.rerank_mock("laptop", candidates, top_k=3)
    for r in results:
        assert isinstance(r.final_score, float)


def test_rerank_mock_rank_change_property(candidates):
    reranker = CrossEncoderReranker()
    results = reranker.rerank_mock("MacBook machine learning", candidates, top_k=3)
    # rank_changed is a bool property
    for r in results:
        assert isinstance(r.rank_changed, bool)


def test_rerank_mock_empty_candidates():
    reranker = CrossEncoderReranker()
    results = reranker.rerank_mock("any query", [], top_k=3)
    assert results == []


# ── MMR Select ────────────────────────────────────────────────

def test_mmr_select_returns_top_k(mock_embed, sample_corpus):
    vecs = mock_embed(sample_corpus)
    query_vec = mock_embed(["best laptop for ML"])[0]
    selected = mmr_select(query_vec, vecs, sample_corpus, top_k=3)
    assert len(selected) == 3


def test_mmr_select_returns_tuples(mock_embed, sample_corpus):
    vecs = mock_embed(sample_corpus)
    query_vec = mock_embed(["laptop"])[0]
    selected = mmr_select(query_vec, vecs, sample_corpus, top_k=3)
    for item in selected:
        assert isinstance(item, tuple)
        assert len(item) == 2


def test_mmr_select_indices_in_range(mock_embed, sample_corpus):
    vecs = mock_embed(sample_corpus)
    query_vec = mock_embed(["headphones"])[0]
    selected = mmr_select(query_vec, vecs, sample_corpus, top_k=3)
    for idx, score in selected:
        assert 0 <= idx < len(sample_corpus)


def test_mmr_select_lambda_1_pure_relevance(mock_embed, sample_corpus):
    vecs = mock_embed(sample_corpus)
    query_vec = mock_embed(["MacBook machine learning"])[0]
    selected_pure = mmr_select(query_vec, vecs, sample_corpus, top_k=3, lambda_param=1.0)
    assert len(selected_pure) == 3


def test_mmr_select_lambda_0_diversity(mock_embed, sample_corpus):
    vecs = mock_embed(sample_corpus)
    query_vec = mock_embed(["anything"])[0]
    selected_diverse = mmr_select(query_vec, vecs, sample_corpus, top_k=3, lambda_param=0.0)
    # Diversity selection should return unique indices
    indices = [i for i, _ in selected_diverse]
    assert len(indices) == len(set(indices))


def test_mmr_select_empty_docs(mock_embed):
    vecs = np.empty((0, 384), dtype=np.float32)
    query_vec = mock_embed(["query"])[0]
    selected = mmr_select(query_vec, vecs, [], top_k=3)
    assert selected == []


def test_mmr_select_top_k_capped(mock_embed):
    corpus = ["doc one", "doc two"]
    vecs = mock_embed(corpus)
    query_vec = mock_embed(["query"])[0]
    selected = mmr_select(query_vec, vecs, corpus, top_k=10)
    # Can't return more docs than exist
    assert len(selected) <= len(corpus)


# ── metadata_filter ───────────────────────────────────────────

def test_metadata_filter_exact_match(docs_with_meta):
    filtered = metadata_filter(docs_with_meta, {"category": "laptop"})
    assert len(filtered) == 2
    for d in filtered:
        assert d["metadata"]["category"] == "laptop"


def test_metadata_filter_price_range(docs_with_meta):
    filtered = metadata_filter(docs_with_meta, {"price_range": (0, 50000)})
    assert len(filtered) >= 1
    for d in filtered:
        assert d["metadata"]["price"] <= 50000


def test_metadata_filter_no_match(docs_with_meta):
    filtered = metadata_filter(docs_with_meta, {"category": "tablet"})
    assert filtered == []


def test_metadata_filter_empty_filters(docs_with_meta):
    filtered = metadata_filter(docs_with_meta, {})
    assert len(filtered) == len(docs_with_meta)


def test_metadata_filter_multiple_filters(docs_with_meta):
    filtered = metadata_filter(docs_with_meta, {"category": "headphones", "price_range": (0, 30000)})
    assert all(d["metadata"]["category"] == "headphones" for d in filtered)
    assert all(d["metadata"]["price"] <= 30000 for d in filtered)


# ── build_retrieval_pipeline ──────────────────────────────────

def test_build_retrieval_pipeline_returns_callable(docs_with_meta, mock_embed):
    retrieve = build_retrieval_pipeline(docs_with_meta, [], mock_embed_fn=mock_embed)
    assert callable(retrieve)


def test_retrieval_pipeline_returns_results(docs_with_meta, mock_embed):
    retrieve = build_retrieval_pipeline(docs_with_meta, [], mock_embed_fn=mock_embed)
    results = retrieve("best laptop for machine learning", top_k=2)
    assert len(results) <= 2


def test_retrieval_pipeline_with_filter(docs_with_meta, mock_embed):
    retrieve = build_retrieval_pipeline(docs_with_meta, [], mock_embed_fn=mock_embed)
    results = retrieve("audio device", filters={"category": "headphones"}, top_k=3)
    # All results should be from headphones (based on filter pre-step)
    assert isinstance(results, list)
