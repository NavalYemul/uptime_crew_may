import pytest
from day5.evaluation import (
    compute_faithfulness, compute_answer_relevancy,
    evaluate_rag_response, RAGASResult,
    RAGMonitor, CostTracker, MonitoringMetrics
)


class TestFaithfulness:
    def test_high_when_answer_in_context(self):
        contexts = ["BM25 is a ranking algorithm for information retrieval"]
        answer   = "BM25 is a ranking algorithm"
        score = compute_faithfulness(answer, contexts)
        assert score > 0.5

    def test_low_when_answer_not_in_context(self):
        contexts = ["Python is a programming language"]
        answer   = "BM25 is a ranking algorithm"
        score = compute_faithfulness(answer, contexts)
        assert score < 0.5

    def test_empty_inputs(self):
        assert compute_faithfulness("", []) == 0.0
        assert compute_faithfulness("answer", []) == 0.0


class TestAnswerRelevancy:
    def test_high_when_answer_addresses_question(self):
        score = compute_answer_relevancy(
            "What is BM25?",
            "BM25 is a ranking function used in search engines"
        )
        assert score > 0.3

    def test_returns_float_between_0_and_1(self):
        score = compute_answer_relevancy("What is RAG?", "RAG is retrieval augmented generation")
        assert 0.0 <= score <= 1.0


class TestFullEvaluation:
    def test_returns_ragas_result(self):
        result = evaluate_rag_response(
            question="What is hybrid search?",
            answer="Hybrid search combines BM25 and semantic retrieval",
            contexts=["Hybrid search combines keyword and semantic retrieval methods"]
        )
        assert isinstance(result, RAGASResult)
        assert 0.0 <= result.faithfulness <= 1.0
        assert 0.0 <= result.answer_relevancy <= 1.0
        assert 0.0 <= result.overall <= 1.0


class TestRAGMonitor:
    def test_records_query(self):
        monitor = RAGMonitor()
        monitor.record("What is BM25?", "BM25 is a ranking algorithm", ["BM25 ranks documents"])
        metrics = monitor.get_metrics()
        assert metrics.total_queries == 1

    def test_detects_drift(self):
        monitor = RAGMonitor(faithfulness_threshold=0.5)
        # Add 15 high-quality records
        for i in range(15):
            monitor._records.append({
                "query": "q", "answer": "a",
                "faithfulness": 0.9, "answer_relevancy": 0.8,
                "tokens": 100, "latency_ms": 50
            })
        # Add 10 low-quality recent records
        for i in range(10):
            monitor._records.append({
                "query": "q", "answer": "a",
                "faithfulness": 0.2, "answer_relevancy": 0.3,
                "tokens": 100, "latency_ms": 50
            })
        drift = monitor.detect_drift(window=10)
        assert drift["drifted"] == True
        assert drift["delta"] > 0.1

    def test_no_drift_stable_quality(self):
        monitor = RAGMonitor()
        for i in range(25):
            monitor._records.append({
                "query": "q", "answer": "a",
                "faithfulness": 0.8, "answer_relevancy": 0.7,
                "tokens": 100, "latency_ms": 40
            })
        drift = monitor.detect_drift(window=10)
        assert drift["drifted"] == False


class TestCostTracker:
    def test_records_cost(self):
        tracker = CostTracker("gpt-4o-mini")
        cost = tracker.record(1000, 200)
        assert cost > 0.0

    def test_total_cost_sums(self):
        tracker = CostTracker("gpt-4o-mini")
        c1 = tracker.record(500, 100)
        c2 = tracker.record(500, 100)
        assert abs(tracker.total_cost() - (c1 + c2)) < 1e-9

    def test_summary_has_required_keys(self):
        tracker = CostTracker()
        tracker.record(1000, 500, "test query")
        s = tracker.summary()
        assert "total_cost_usd" in s
        assert "sessions" in s
        assert "model" in s
