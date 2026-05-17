"""Tests for day4.prompt_engineering"""
import pytest
from langchain_core.prompts import PromptTemplate
from day4.prompt_engineering import (
    zero_shot_prompt, few_shot_prompt, chain_of_thought_prompt,
    count_tokens, estimate_cost,
    PersonInfo, extract_person_info_mock,
)


class TestPromptTemplates:
    def test_zero_shot_returns_prompt_template(self):
        pt = zero_shot_prompt("Classify sentiment")
        assert isinstance(pt, PromptTemplate)

    def test_zero_shot_has_input_variable(self):
        pt = zero_shot_prompt("Classify sentiment")
        assert "input" in pt.input_variables

    def test_zero_shot_formats(self):
        pt = zero_shot_prompt("Classify sentiment")
        result = pt.format(input="This is great!")
        assert "This is great!" in result
        assert "Classify sentiment" in result

    def test_few_shot_returns_prompt_template(self):
        examples = [{"input": "great", "output": "positive"}]
        pt = few_shot_prompt("Classify:", examples)
        assert isinstance(pt, PromptTemplate)

    def test_few_shot_includes_examples(self):
        examples = [
            {"input": "amazing", "output": "positive"},
            {"input": "terrible", "output": "negative"},
        ]
        pt = few_shot_prompt("Classify:", examples)
        formatted = pt.format(input="wonderful")
        assert "amazing" in formatted
        assert "positive" in formatted
        assert "terrible" in formatted

    def test_chain_of_thought_includes_step_by_step(self):
        pt = chain_of_thought_prompt("Solve the problem")
        formatted = pt.format(input="2 + 2")
        assert "step by step" in formatted.lower()

    def test_chain_of_thought_returns_prompt_template(self):
        pt = chain_of_thought_prompt("Solve it")
        assert isinstance(pt, PromptTemplate)


class TestTokenCounting:
    def test_count_tokens_returns_int(self):
        result = count_tokens("Hello, world!")
        assert isinstance(result, int)
        assert result > 0

    def test_count_tokens_empty_string(self):
        result = count_tokens("")
        assert result == 0

    def test_count_tokens_longer_text_more_tokens(self):
        short = count_tokens("Hi")
        long = count_tokens("This is a much longer sentence with many more words in it.")
        assert long > short

    def test_count_tokens_different_models(self):
        # Both should return positive int
        n1 = count_tokens("test text", model="gpt-4o")
        n2 = count_tokens("test text", model="gpt-4o-mini")
        assert n1 > 0
        assert n2 > 0


class TestCostEstimation:
    def test_estimate_cost_returns_dict(self):
        result = estimate_cost(1000)
        assert isinstance(result, dict)

    def test_estimate_cost_has_cost_usd(self):
        result = estimate_cost(1000, model="gpt-4o-mini")
        assert "cost_usd" in result
        assert isinstance(result["cost_usd"], float)

    def test_estimate_cost_has_input_tokens(self):
        result = estimate_cost(500)
        assert result["input_tokens"] == 500

    def test_estimate_cost_has_model(self):
        result = estimate_cost(100, model="gpt-4o")
        assert result["model"] == "gpt-4o"

    def test_estimate_cost_more_tokens_higher_cost(self):
        c1 = estimate_cost(100, model="gpt-4o-mini")
        c2 = estimate_cost(1000, model="gpt-4o-mini")
        assert c2["cost_usd"] > c1["cost_usd"]

    def test_estimate_cost_zero_tokens(self):
        result = estimate_cost(0)
        assert result["cost_usd"] == 0.0


class TestPersonExtraction:
    def test_extract_returns_person_info(self):
        text = "John Smith is 30 years old and works as a software engineer."
        person = extract_person_info_mock(text)
        assert isinstance(person, PersonInfo)

    def test_extract_age(self):
        text = "John is 25 years old and works as a doctor."
        person = extract_person_info_mock(text)
        assert person.age == 25

    def test_extract_name(self):
        text = "Name: Alice Johnson. She is 28 years old."
        person = extract_person_info_mock(text)
        assert "Alice" in person.name

    def test_extract_occupation(self):
        text = "Bob works as a teacher."
        person = extract_person_info_mock(text)
        assert "teacher" in person.occupation.lower()

    def test_person_info_model_fields(self):
        p = PersonInfo(name="Test User", age=40, occupation="analyst")
        assert p.name == "Test User"
        assert p.age == 40
        assert p.occupation == "analyst"

    def test_person_info_model_dump(self):
        p = PersonInfo(name="Jane", age=35, occupation="manager")
        d = p.model_dump()
        assert set(d.keys()) == {"name", "age", "occupation"}
