"""Tests for day3.evaluation — RAGAS proxy, EvalResult, RAGEvaluator, A/B test."""

import json
import pytest
from pathlib import Path

from day3.evaluation import (
    QAPair,
    EvalResult,
    compute_ragas_proxy,
    GOLDEN_DATASET,
    RAGEvaluator,
    ab_test,
)


# ── compute_ragas_proxy ───────────────────────────────────────

def test_compute_ragas_returns_eval_result():
    result = compute_ragas_proxy(
        question="What laptop is good for ML?",
        answer="MacBook Pro M3 is great for ML.",
        context="MacBook Pro M3 chip with 18GB RAM for ML workloads.",
        ground_truth="MacBook Pro M3 is best for ML.",
    )
    assert isinstance(result, EvalResult)


def test_compute_ragas_scores_in_range():
    result = compute_ragas_proxy(
        question="Which headphones have best ANC?",
        answer="Sony WH-1000XM5 has industry-leading ANC.",
        context="Sony WH-1000XM5 headphones with dual processors and ANC technology.",
        ground_truth="Sony WH-1000XM5 has industry-leading ANC with 8 microphones.",
    )
    assert 0.0 <= result.faithfulness <= 1.0
    assert 0.0 <= result.answer_relevancy <= 1.0
    assert 0.0 <= result.context_precision <= 1.0
    assert 0.0 <= result.context_recall <= 1.0


def test_eval_result_average_property():
    r = EvalResult(
        question="q", answer="a", ground_truth="gt",
        faithfulness=0.8, answer_relevancy=0.7,
        context_precision=0.6, context_recall=0.9,
    )
    expected = (0.8 + 0.7 + 0.6 + 0.9) / 4
    assert abs(r.average - expected) < 1e-4


def test_eval_result_passed_threshold():
    # Both thresholds met: faith >= 0.85, relevancy >= 0.7
    r = EvalResult(
        question="q", answer="a", ground_truth="gt",
        faithfulness=0.9, answer_relevancy=0.8,
        context_precision=0.7, context_recall=0.8,
    )
    assert r.passed is True


def test_eval_result_failed_low_faithfulness():
    r = EvalResult(
        question="q", answer="a", ground_truth="gt",
        faithfulness=0.5,  # below 0.85
        answer_relevancy=0.9,
        context_precision=0.7, context_recall=0.8,
    )
    assert r.passed is False


def test_eval_result_failed_low_relevancy():
    r = EvalResult(
        question="q", answer="a", ground_truth="gt",
        faithfulness=0.9,
        answer_relevancy=0.3,  # below 0.7
        context_precision=0.7, context_recall=0.8,
    )
    assert r.passed is False


# ── GOLDEN_DATASET ────────────────────────────────────────────

def test_golden_dataset_has_ten_items():
    assert len(GOLDEN_DATASET) == 10


def test_golden_dataset_all_qa_pairs():
    for qa in GOLDEN_DATASET:
        assert isinstance(qa, QAPair)
        assert len(qa.question) > 5
        assert len(qa.ground_truth) > 10
        assert len(qa.category) > 0


# ── RAGEvaluator ──────────────────────────────────────────────

def test_rag_evaluator_run_returns_list(mock_embed, sample_qa_pairs):
    from day3.rag_pipeline import build_product_rag
    pipeline = build_product_rag(mock_embed_fn=mock_embed)
    evaluator = RAGEvaluator(pipeline)
    results = evaluator.run(sample_qa_pairs)
    assert isinstance(results, list)
    assert len(results) == len(sample_qa_pairs)


def test_rag_evaluator_summary_keys(mock_embed, sample_qa_pairs):
    from day3.rag_pipeline import build_product_rag
    pipeline = build_product_rag(mock_embed_fn=mock_embed)
    evaluator = RAGEvaluator(pipeline)
    results = evaluator.run(sample_qa_pairs)
    summary = evaluator.summary(results)
    assert "faithfulness" in summary
    assert "answer_relevancy" in summary
    assert "pass_rate" in summary
    assert "num_evaluated" in summary


def test_rag_evaluator_failing_questions(mock_embed, sample_qa_pairs):
    from day3.rag_pipeline import build_product_rag
    pipeline = build_product_rag(mock_embed_fn=mock_embed)
    evaluator = RAGEvaluator(pipeline)
    results = evaluator.run(sample_qa_pairs)
    failing = evaluator.failing_questions(results)
    assert isinstance(failing, list)
    # All failing results should have passed=False
    for r in failing:
        assert r.passed is False


def test_rag_evaluator_summary_empty():
    from day3.rag_pipeline import build_product_rag  # dummy pipeline not needed

    class DummyPipeline:
        def query(self, q):
            from day3.rag_pipeline import AdvancedRAGResult
            return AdvancedRAGResult(q, "dummy", [], [], "mock", False, 0.0, 0.0, 0, 0)

    evaluator = RAGEvaluator(DummyPipeline())
    summary = evaluator.summary([])
    assert "error" in summary


def test_save_baseline_writes_json(tmp_path, mock_embed, sample_qa_pairs):
    from day3.rag_pipeline import build_product_rag
    pipeline = build_product_rag(mock_embed_fn=mock_embed)
    evaluator = RAGEvaluator(pipeline)
    results = evaluator.run(sample_qa_pairs)
    output_path = str(tmp_path / "test_baseline.json")
    baseline = evaluator.save_baseline(results, path=output_path)
    assert Path(output_path).exists()
    loaded = json.loads(Path(output_path).read_text())
    assert "summary" in loaded
    assert "num_queries" in loaded


# ── ab_test ───────────────────────────────────────────────────

def test_ab_test_returns_winner(mock_embed, sample_qa_pairs):
    from day3.rag_pipeline import build_product_rag
    pipeline_a = build_product_rag(mock_embed_fn=mock_embed)
    pipeline_b = build_product_rag(mock_embed_fn=mock_embed)
    result = ab_test(pipeline_a, pipeline_b, sample_qa_pairs,
                     name_a="hybrid", name_b="dense_only")
    assert "overall_winner" in result
    assert result["overall_winner"] in ("hybrid", "dense_only")


def test_ab_test_metric_winners(mock_embed, sample_qa_pairs):
    from day3.rag_pipeline import build_product_rag
    pipeline_a = build_product_rag(mock_embed_fn=mock_embed)
    pipeline_b = build_product_rag(mock_embed_fn=mock_embed)
    result = ab_test(pipeline_a, pipeline_b, sample_qa_pairs)
    assert "metric_winners" in result
    assert "faithfulness" in result["metric_winners"]
