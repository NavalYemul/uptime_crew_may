"""
tools_agents.py — LangGraph Tool Use and ReAct Agents
======================================================
Covers: @tool decorator, ToolNode, tools_condition, bind_tools,
        ReAct agent pattern, streaming, mock LLM for tests.

Industry use:
  Every production LangGraph agent uses this exact pattern:
  LLM decides which tool to call -> ToolNode executes it -> LLM sees result -> loops.
  This is the foundation of Copilot, Claude's tool use, and all AI assistants.

Run: python -m day4.tools_agents
"""

from __future__ import annotations

import json
from typing import TypedDict, Annotated, Optional, Any, Callable
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# ══════════════════════════════════════════════════════
# EXAMPLE TOOLS
# ══════════════════════════════════════════════════════

@tool
def calculator(expression: str) -> str:
    """Evaluate a safe mathematical expression.

    Args:
        expression: A mathematical expression like "2 + 3 * 4" or "sqrt(16)".

    Returns:
        The result as a string.
    """
    import math
    try:
        # Safe eval with only math functions allowed
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(round(float(result), 6))
    except Exception as e:
        return f"Error: {e}"


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city (mock implementation).

    Args:
        city: City name.

    Returns:
        Weather description string.
    """
    # Mock data — in production: call a real weather API
    weather_data = {
        "mumbai":     "Humid, 32C, partly cloudy",
        "delhi":      "Hot, 38C, sunny",
        "bangalore":  "Pleasant, 24C, light breeze",
        "chennai":    "Hot and humid, 34C",
        "kolkata":    "Warm, 30C, chance of rain",
        "pune":       "Comfortable, 26C, clear",
        "hyderabad":  "Warm, 28C, sunny",
        "tokyo":      "Mild, 22C, overcast",
        "london":     "Cool, 14C, cloudy",
        "new york":   "Warm, 25C, partly sunny",
        "gurgaon":    "Hot, 36C, sunny",
        "default":    "20C, partly cloudy",
    }
    key = city.lower().strip()
    return weather_data.get(key, weather_data["default"]) + f" in {city}"


@tool
def search_web(query: str) -> str:
    """Search the web for information (mock implementation).

    Args:
        query: Search query string.

    Returns:
        Simulated search result string.
    """
    # Mock results — in production: use DuckDuckGoSearchRun or SerpAPI
    mock_results = {
        "dhurandhar": "Dhurandhar 2 is an upcoming Bollywood thriller film, release date: August 2025.",
        "kalpana chawla": "Kalpana Chawla was born in Karnal, Haryana, India. She was NASA's first Indian-American astronaut.",
        "india gdp": "India's GDP grew 7.2% in FY2024, making it the world's fastest-growing major economy.",
        "langchain": "LangChain is an open-source framework for building LLM-powered applications. Latest version: 0.3.x",
        "langgraph": "LangGraph is a library for building stateful, multi-actor LLM applications using graph-based state machines.",
    }
    query_lower = query.lower()
    for key, result in mock_results.items():
        if key in query_lower:
            return result
    return f"Search results for '{query}': Found 3 relevant articles. Key facts: This is a mock search result."


# ══════════════════════════════════════════════════════
# AGENT STATE
# ══════════════════════════════════════════════════════

class AgentState(TypedDict):
    """State for a tool-using agent. Uses add_messages reducer so messages accumulate."""
    messages: Annotated[list[BaseMessage], add_messages]


# ══════════════════════════════════════════════════════
# MOCK LLM FOR TESTS
# ══════════════════════════════════════════════════════

class MockLLMWithTools:
    """Deterministic mock LLM that simulates tool calling for tests.

    Instead of calling OpenAI, this mock:
    - Returns a tool_call on the first invocation
    - Returns a final answer on the second invocation (after seeing tool result)

    This lets tests run without API keys or network access.
    """

    def __init__(self, tool_name: str = "calculator", tool_args: dict = None, final_answer: str = "The answer is 42."):
        self._tool_name    = tool_name
        self._tool_args    = tool_args or {"expression": "6 * 7"}
        self._final_answer = final_answer
        self._call_count   = 0
        self.bound_tools: list = []

    def bind_tools(self, tools: list) -> "MockLLMWithTools":
        """Simulate .bind_tools() — returns self (tools noted but not actually bound)."""
        self.bound_tools = tools
        return self

    def invoke(self, messages: list) -> AIMessage:
        self._call_count += 1

        # First call: decide to call a tool
        has_tool_result = any(isinstance(m, ToolMessage) for m in messages)

        if not has_tool_result:
            # Return a message with a tool call
            tool_call = {
                "id":   f"call_{self._call_count:04d}",
                "name": self._tool_name,
                "args": self._tool_args,
                "type": "tool_call",
            }
            return AIMessage(
                content    = "",
                tool_calls = [tool_call],
            )
        else:
            # Return final answer after seeing tool result
            return AIMessage(content=self._final_answer)

    @property
    def call_count(self) -> int:
        return self._call_count


# ══════════════════════════════════════════════════════
# REACT AGENT GRAPH
# ══════════════════════════════════════════════════════

def build_tool_agent(llm, tools: list):
    """Build a ReAct-style agent graph with tool use.

    Pattern (industry standard):
      1. Agent node calls LLM with tool definitions.
      2. If LLM returns tool_calls -> route to ToolNode.
      3. ToolNode executes the tool, appends ToolMessage to state.
      4. Loop back to agent node — LLM sees tool result and decides next step.
      5. If LLM returns no tool_calls -> END.

    Args:
        llm:   LangChain LLM instance with .bind_tools() method.
        tools: List of @tool-decorated callables.

    Returns:
        Compiled LangGraph graph.
    """
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def agent_node(state: AgentState) -> dict:
        """Call the LLM with the current message history."""
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    g = StateGraph(AgentState)
    g.add_node("agent",     agent_node)
    g.add_node("tools",     tool_node)

    g.add_edge(START,   "agent")
    g.add_conditional_edges(
        "agent",
        tools_condition,   # langgraph built-in: checks if last message has tool_calls
        {"tools": "tools", END: END},
    )
    g.add_edge("tools", "agent")   # loop back after tool execution

    return g.compile()


# ══════════════════════════════════════════════════════
# TOOL RESULT FORMATTING
# ══════════════════════════════════════════════════════

def extract_final_answer(state: AgentState) -> str:
    """Extract the last AI message content from an agent state.

    Args:
        state: Final agent state dict with 'messages' key.

    Returns:
        The final text answer from the agent.
    """
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return ""


def count_tool_calls(state: AgentState) -> int:
    """Count how many tool calls were made during the agent run.

    Args:
        state: Final agent state.

    Returns:
        Number of ToolMessage instances in the message history.
    """
    return sum(1 for m in state.get("messages", []) if isinstance(m, ToolMessage))


# ══════════════════════════════════════════════════════
# MAIN DEMO
# ══════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("TOOL USE & REACT AGENTS DEMO")
    print("=" * 70)

    tools = [calculator, get_weather, search_web]

    print("\n[1] Individual tools")
    print(f"  calculator('2 ** 10')   = {calculator.invoke({'expression': '2 ** 10'})}")
    print(f"  get_weather('mumbai')   = {get_weather.invoke({'city': 'mumbai'})}")
    print(f"  search_web('langgraph') = {search_web.invoke({'query': 'langgraph'})[:70]}")

    print("\n[2] Tool-using agent with mock LLM")
    mock_llm = MockLLMWithTools(
        tool_name    = "calculator",
        tool_args    = {"expression": "18 * 9900"},
        final_answer = "The annual salary is Rs 1,78,200 (18 months x Rs 9,900).",
    )
    agent = build_tool_agent(mock_llm, tools)
    result = agent.invoke({"messages": [HumanMessage(content="What is 18 * 9900?")]})

    print(f"  Query: 'What is 18 * 9900?'")
    print(f"  Tool calls made: {count_tool_calls(result)}")
    print(f"  Final answer: {extract_final_answer(result)}")

    print("\n[3] Available tools:")
    for t in tools:
        print(f"  {t.name}: {t.description[:60]}")

    print("\n[4] Real agent (requires OPENAI_API_KEY):")
    print("  from langchain_openai import ChatOpenAI")
    print("  llm = ChatOpenAI(model='gpt-4o-mini')")
    print("  agent = build_tool_agent(llm, [calculator, get_weather, search_web])")
    print("  result = agent.invoke({'messages': [HumanMessage('Weather in Tokyo?')]})")


if __name__ == "__main__":
    main()
