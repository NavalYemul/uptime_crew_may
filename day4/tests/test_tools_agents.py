"""Tests for day4.tools_agents"""
import pytest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from day4.tools_agents import (
    calculator, get_weather, search_web,
    build_tool_agent, extract_final_answer, count_tool_calls,
    MockLLMWithTools, AgentState
)


class TestTools:
    def test_calculator_basic(self):
        assert calculator.invoke({"expression": "2 + 2"}) == "4.0"

    def test_calculator_power(self):
        assert calculator.invoke({"expression": "2 ** 10"}) == "1024.0"

    def test_calculator_sqrt(self):
        result = float(calculator.invoke({"expression": "sqrt(144)"}))
        assert result == pytest.approx(12.0)

    def test_calculator_error(self):
        result = calculator.invoke({"expression": "1 / 0"})
        assert "Error" in result or "error" in result.lower() or result == "inf"

    def test_get_weather_known_city(self):
        result = get_weather.invoke({"city": "mumbai"})
        assert "mumbai" in result.lower() or "Mumbai" in result
        assert "C" in result

    def test_get_weather_unknown_city(self):
        result = get_weather.invoke({"city": "UnknownCity123"})
        assert len(result) > 5  # returns default weather

    def test_search_web_returns_string(self):
        result = search_web.invoke({"query": "langgraph"})
        assert isinstance(result, str)
        assert len(result) > 10


class TestMockLLM:
    def test_first_call_returns_tool_call(self):
        llm = MockLLMWithTools(tool_name="calculator", tool_args={"expression": "2+2"})
        response = llm.invoke([HumanMessage("What is 2+2?")])
        assert isinstance(response, AIMessage)
        assert len(response.tool_calls) > 0
        assert response.tool_calls[0]["name"] == "calculator"

    def test_second_call_returns_final_answer(self):
        llm = MockLLMWithTools(final_answer="The result is 4.")
        # First call
        llm.invoke([HumanMessage("2+2?")])
        # Second call with a ToolMessage (simulates tool result)
        response = llm.invoke([
            HumanMessage("2+2?"),
            AIMessage(content="", tool_calls=[{"id": "x", "name": "calculator", "args": {}, "type": "tool_call"}]),
            ToolMessage(content="4.0", tool_call_id="x"),
        ])
        assert response.content == "The result is 4."

    def test_bind_tools_returns_self(self):
        llm = MockLLMWithTools()
        result = llm.bind_tools([calculator])
        assert result is llm
        assert len(result.bound_tools) == 1


class TestToolAgent:
    def test_agent_completes(self):
        llm = MockLLMWithTools(
            tool_name    = "calculator",
            tool_args    = {"expression": "6 * 7"},
            final_answer = "6 * 7 = 42",
        )
        agent = build_tool_agent(llm, [calculator, get_weather])
        result = agent.invoke({"messages": [HumanMessage("What is 6 * 7?")]})
        assert "messages" in result
        assert len(result["messages"]) > 1

    def test_extract_final_answer(self):
        state = {
            "messages": [
                HumanMessage("Question"),
                AIMessage(content="", tool_calls=[{"id": "1", "name": "calc", "args": {}, "type": "tool_call"}]),
                ToolMessage(content="42", tool_call_id="1"),
                AIMessage(content="The answer is 42."),
            ]
        }
        assert extract_final_answer(state) == "The answer is 42."

    def test_count_tool_calls(self):
        state = {
            "messages": [
                HumanMessage("Question"),
                ToolMessage(content="result1", tool_call_id="1"),
                ToolMessage(content="result2", tool_call_id="2"),
                AIMessage(content="Done."),
            ]
        }
        assert count_tool_calls(state) == 2
