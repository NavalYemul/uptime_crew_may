"""Tests for day4.langgraph_basics"""
import pytest
from day4.langgraph_basics import (
    build_bmi_graph, build_prompt_chain_graph, build_review_graph,
    inspect_graph, run_graph, BMIState, ChainState, ReviewState
)


class TestBMIGraph:
    def setup_method(self):
        self.graph = build_bmi_graph()

    def test_normal_bmi(self):
        result = run_graph(self.graph, {
            "weight_kg": 70, "height_m": 1.75, "bmi": None, "category": None, "recommendation": None
        })
        assert result["bmi"] == pytest.approx(22.86, abs=0.1)
        assert result["category"] == "normal"
        assert result["recommendation"] is not None

    def test_underweight(self):
        result = run_graph(self.graph, {
            "weight_kg": 45, "height_m": 1.70, "bmi": None, "category": None, "recommendation": None
        })
        assert result["category"] == "underweight"
        assert "caloric" in result["recommendation"].lower() or "nutrient" in result["recommendation"].lower()

    def test_overweight(self):
        result = run_graph(self.graph, {
            "weight_kg": 85, "height_m": 1.70, "bmi": None, "category": None, "recommendation": None
        })
        assert result["category"] == "overweight"

    def test_obese(self):
        result = run_graph(self.graph, {
            "weight_kg": 100, "height_m": 1.60, "bmi": None, "category": None, "recommendation": None
        })
        assert result["category"] == "obese"

    def test_bmi_calculation(self):
        # BMI = 90 / 1.8^2 = 27.78
        result = run_graph(self.graph, {
            "weight_kg": 90, "height_m": 1.80, "bmi": None, "category": None, "recommendation": None
        })
        assert result["bmi"] == pytest.approx(27.78, abs=0.1)

    def test_graph_has_required_nodes(self):
        info = inspect_graph(self.graph)
        assert info["num_nodes"] >= 4


class TestPromptChainGraph:
    def test_chain_runs_all_three_steps(self):
        call_log = []
        def mock_llm(prompt: str) -> str:
            call_log.append(prompt)
            if "report" in prompt.lower():
                return "Detailed report about the topic."
            elif "bullet" in prompt.lower() or "summarise" in prompt.lower():
                return "- Point 1\n- Point 2\n- Point 3"
            else:
                return "Short tweet about the topic! #AI"

        graph = build_prompt_chain_graph(mock_llm)
        result = graph.invoke({"topic": "AI in India", "detailed_report": None, "summary": None, "tweet": None})

        assert result["detailed_report"] is not None
        assert result["summary"] is not None
        assert result["tweet"] is not None
        assert len(call_log) == 3

    def test_tweet_truncated_to_280(self):
        def mock_llm(prompt: str) -> str:
            return "x" * 500
        graph = build_prompt_chain_graph(mock_llm)
        result = graph.invoke({"topic": "test", "detailed_report": None, "summary": None, "tweet": None})
        assert len(result["tweet"]) <= 280


class TestReviewGraph:
    def setup_method(self):
        self.graph = build_review_graph()

    def test_positive_review(self):
        result = run_graph(self.graph, {
            "review_text": "This product is amazing and great quality!", "sentiment": None, "reply": None
        })
        assert result["sentiment"] == "positive"
        assert result["reply"] is not None

    def test_negative_review(self):
        result = run_graph(self.graph, {
            "review_text": "Terrible experience, worst service ever.", "sentiment": None, "reply": None
        })
        assert result["sentiment"] == "negative"
        assert "support" in result["reply"].lower() or "sorry" in result["reply"].lower()

    def test_neutral_review(self):
        result = run_graph(self.graph, {
            "review_text": "It arrived on time.", "sentiment": None, "reply": None
        })
        assert result["sentiment"] == "neutral"

    def test_reply_generated_for_all_sentiments(self):
        reviews = [
            "Great amazing product love it!",
            "Awful terrible horrible experience.",
            "It was delivered.",
        ]
        for review in reviews:
            result = run_graph(self.graph, {"review_text": review, "sentiment": None, "reply": None})
            assert result["reply"] is not None and len(result["reply"]) > 10
