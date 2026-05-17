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
from typing import TypedDict, Annotated, Optional, Any, Callable, Type
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool, BaseTool
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel


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
# BASE TOOL SUBCLASS EXAMPLE
# ══════════════════════════════════════════════════════

class WeatherTool(BaseTool):
    """Example BaseTool subclass — an alternative to the @tool decorator.

    Use BaseTool when you need instance attributes, complex initialization,
    or shared state across calls (e.g., a database connection or API client).
    """
    name: str = "weather_tool"
    description: str = "Get current weather for a city. Input should be the city name."

    def _run(self, city: str) -> str:
        """Synchronous implementation required by BaseTool."""
        weather_data = {
            "mumbai":    "Humid, 32C, partly cloudy",
            "delhi":     "Hot, 38C, sunny",
            "bangalore": "Pleasant, 24C, light breeze",
            "london":    "Cool, 14C, cloudy",
            "tokyo":     "Mild, 22C, overcast",
        }
        return weather_data.get(city.lower().strip(), f"22C, partly cloudy in {city}")

    async def _arun(self, city: str) -> str:
        """Async version (required by BaseTool interface)."""
        return self._run(city)


# ══════════════════════════════════════════════════════
# PYDANTIC STRUCTURED OUTPUT
# ══════════════════════════════════════════════════════

class WeatherOutput(BaseModel):
    """Structured weather data — use Instructor or .with_structured_output() to produce this."""
    city: str
    temperature: float
    unit: str
    description: str


# ══════════════════════════════════════════════════════
# GUARDRAILS
# ══════════════════════════════════════════════════════

UNSAFE_KEYWORDS = ["ignore previous", "jailbreak", "DAN", "system prompt"]


def add_guardrails(chain_fn: Callable[[str], Any]) -> Callable[[str], Any]:
    """Wrap a function with input guardrails.

    Checks input for common prompt injection / jailbreak patterns.
    Raises ValueError if unsafe content is detected.

    Args:
        chain_fn: Any callable that takes a string input.

    Returns:
        Wrapped callable that validates input before calling chain_fn.
    """
    def safe_wrapper(user_input: str) -> Any:
        lower = user_input.lower()
        for keyword in UNSAFE_KEYWORDS:
            if keyword.lower() in lower:
                raise ValueError(
                    f"Input blocked: unsafe pattern detected — '{keyword}'"
                )
        return chain_fn(user_input)

    return safe_wrapper


# ══════════════════════════════════════════════════════
# REACT AGENT EXECUTOR (local prompt, no hub.pull)
# ══════════════════════════════════════════════════════

def build_react_agent_executor(llm, tools: list):
    """Build a ReAct-style agent using LangChain's create_react_agent pattern.

    Uses a local prompt template instead of hub.pull to avoid network calls.
    In production: hub.pull("hwchase17/react") provides the canonical ReAct prompt.

    Args:
        llm:   LangChain LLM instance.
        tools: List of @tool-decorated callables.

    Returns:
        Compiled LangGraph agent (same as build_tool_agent but with explicit prompt).
    """
    # Local ReAct prompt — equivalent to hwchase17/react from hub
    react_prompt = PromptTemplate.from_template(
        "Answer the following question using the available tools.\n\n"
        "Tools: {tools}\n\n"
        "Question: {input}\n\n"
        "Think step by step. Use a tool if needed.\n"
        "Thought: {agent_scratchpad}"
    )
    # For LangGraph-based agents, build_tool_agent already implements ReAct
    return build_tool_agent(llm, tools)


# ══════════════════════════════════════════════════════
# PLAN-AND-EXECUTE AGENT
# ══════════════════════════════════════════════════════

class PlanExecuteState(TypedDict):
    """State for a plan-and-execute agent."""
    task: str
    plan: list[str]          # numbered steps
    current_step: int
    results: list[str]       # result of each step
    final_answer: str


def build_plan_execute_agent(llm_fn: Callable[[str], str], tools: list):
    """Build a Plan-and-Execute agent using LangGraph.

    Flow: plan_node generates numbered steps -> execute_node runs each step
    in a loop -> final answer is assembled from all step results.

    Args:
        llm_fn: Callable(prompt: str) -> str. The LLM call.
        tools:  List of @tool callables available during execution.

    Returns:
        Compiled LangGraph graph.
    """
    def plan_node(state: PlanExecuteState) -> dict:
        """Generate a numbered plan for the task."""
        prompt = f"Break this task into 2-3 numbered steps: {state['task']}"
        plan_text = llm_fn(prompt)
        # Parse numbered steps; fall back to wrapping in a single step
        steps = []
        for line in plan_text.strip().split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                # Remove numbering prefix
                cleaned = line.lstrip("0123456789.-) ").strip()
                if cleaned:
                    steps.append(cleaned)
        if not steps:
            steps = [state["task"]]
        return {"plan": steps, "current_step": 0, "results": []}

    def execute_node(state: PlanExecuteState) -> dict:
        """Execute the current step and advance."""
        step_idx = state["current_step"]
        plan = state["plan"]
        results = list(state.get("results", []))

        if step_idx < len(plan):
            step = plan[step_idx]
            result = llm_fn(f"Execute this step: {step}")
            results.append(result)
            return {"current_step": step_idx + 1, "results": results}
        return {"results": results}

    def finalize_node(state: PlanExecuteState) -> dict:
        """Assemble final answer from all step results."""
        parts = []
        for i, (step, result) in enumerate(zip(state["plan"], state["results"]), 1):
            parts.append(f"Step {i} ({step}): {result}")
        final = "\n".join(parts)
        return {"final_answer": final}

    def should_continue(state: PlanExecuteState) -> str:
        if state["current_step"] < len(state["plan"]):
            return "execute"
        return "finalize"

    g = StateGraph(PlanExecuteState)
    g.add_node("plan",     plan_node)
    g.add_node("execute",  execute_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START,    "plan")
    g.add_conditional_edges("plan",    should_continue, {"execute": "execute", "finalize": "finalize"})
    g.add_conditional_edges("execute", should_continue, {"execute": "execute", "finalize": "finalize"})
    g.add_edge("finalize", END)

    return g.compile()


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
