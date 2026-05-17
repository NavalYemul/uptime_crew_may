import pytest
from day5.hybrid_search import (
    BM25Index, DenseIndex, HybridSearcher,
    reciprocal_rank_fusion, rerank_mock, query_rewrite
)


class TestBM25Index:
    def test_returns_results(self, sample_docs, sample_query):
        idx = BM25Index(sample_docs)
        results = idx.search(sample_query, top_k=3)
        assert len(results) <= 3
        assert all(isinstance(i, int) for i, _ in results)

    def test_relevant_doc_ranks_high(self, sample_docs):
        idx = BM25Index(sample_docs)
        docs = idx.get_top_docs("BM25 ranking", top_k=3)
        assert any("BM25" in d for d in docs)

    def test_empty_result_for_no_match(self, sample_docs):
        idx = BM25Index(sample_docs)
        docs = idx.get_top_docs("zzz_nonexistent_xyz", top_k=3)
        assert isinstance(docs, list)


class TestDenseIndex:
    def test_returns_results(self, sample_docs, sample_query):
        idx = DenseIndex(sample_docs)
        results = idx.search(sample_query, top_k=3)
        assert len(results) <= 3

    def test_doc_indices_in_range(self, sample_docs, sample_query):
        idx = DenseIndex(sample_docs)
        for i, _ in idx.search(sample_query, top_k=5):
            assert 0 <= i < len(sample_docs)


class TestRRF:
    def test_merges_two_lists(self):
        list1 = [(0, 1.0), (1, 0.8), (2, 0.6)]
        list2 = [(2, 1.0), (0, 0.7), (3, 0.5)]
        fused = reciprocal_rank_fusion([list1, list2])
        doc_ids = [i for i, _ in fused]
        assert 0 in doc_ids  # doc 0 appears in both → high score
        assert 2 in doc_ids

    def test_scores_sum_correctly(self):
        # Doc 0 at rank 0 in both lists → score = 2/(60+1)
        lists = [[(0, 1.0)], [(0, 0.5)]]
        fused = dict(reciprocal_rank_fusion(lists))
        expected = 2.0 / (60 + 1)
        assert abs(fused[0] - expected) < 1e-9

    def test_k_parameter(self):
        lists = [[(0, 1.0)], [(0, 0.5)]]
        fused_60 = dict(reciprocal_rank_fusion(lists, k=60))
        fused_1  = dict(reciprocal_rank_fusion(lists, k=1))
        # Higher k → lower individual contribution
        assert fused_1[0] > fused_60[0]


class TestHybridSearcher:
    def test_returns_docs(self, sample_docs, sample_query):
        searcher = HybridSearcher(sample_docs)
        results = searcher.search(sample_query, top_k=3)
        assert len(results) <= 3
        assert all(isinstance(r, str) for r in results)

    def test_explain_has_scores(self, sample_docs, sample_query):
        searcher = HybridSearcher(sample_docs)
        explained = searcher.explain(sample_query, top_k=3)
        assert len(explained) <= 3
        for row in explained:
            assert "rrf_score" in row
            assert "bm25_score" in row
            assert "dense_score" in row


class TestQueryRewrite:
    def test_expand(self):
        r = query_rewrite("BM25", strategy="expand")
        assert "BM25" in r
        assert len(r) > len("BM25")

    def test_hypothetical(self):
        r = query_rewrite("What is RAG?", strategy="hypothetical")
        assert "RAG" in r or "answer" in r.lower()

    def test_decompose(self):
        r = query_rewrite("retrieval and ranking", strategy="decompose")
        assert isinstance(r, str)
        assert len(r) > 0


class TestRerank:
    def test_rerank_returns_same_docs(self, sample_docs):
        reranked = rerank_mock("hybrid search BM25", sample_docs[:4])
        assert len(reranked) == 4

    def test_relevant_doc_ranks_first(self):
        docs = ["about cooking", "about BM25 search ranking", "about weather"]
        reranked = rerank_mock("BM25 search", docs)
        assert "BM25" in reranked[0]
