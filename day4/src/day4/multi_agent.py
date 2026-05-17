"""
multi_agent.py — Subgraphs and Multi-Agent Systems
===================================================
Covers: subgraph composition, supervisor routing, agent handoffs,
        shared state across agents, parallel execution.

Why multi-agent?
  Complex tasks are too large for a single context window.
  Different tasks require different tool sets (web search vs code execution).
  Parallel agents complete tasks faster than sequential ones.

Architecture patterns:
  1. Supervisor: one routing agent decides which specialised agent to call
  2. Subgraph: an independent graph used as a node inside a parent graph
  3. Handoff: one agent explicitly routes to another via state update
  4. Parallel: fan-out to multiple agents, fan-in to combine results

Industry use:
  Microsoft Autogen, LangChain multi-agent, Salesforce Agentforce all
  use variants of these patterns for enterprise AI systems.

Run: python -m day4.multi_agent
"""

from __future__ import annotations

from enum import Enum
from typing import TypedDict, Annotated, Optional, Callable
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ══════════════════════════════════════════════════════
# SUBGRAPH PATTERN
# ══════════════════════════════════════════════════════

class ResearchState(TypedDict):
    """State for a research subgraph agent."""
    messages:  Annotated[list[BaseMessage], add_messages]
    topic:     str
    findings:  Optional[str]


class WriterState(TypedDict):
    """State for a writer subgraph agent."""
    messages:  Annotated[list[BaseMessage], add_messages]
    topic:     str
    findings:  Optional[str]
    draft:     Optional[str]


def build_research_subgraph(research_fn: Callable[[str], str]):
    """Build a research agent subgraph.

    This graph can be used as a node inside a parent supervisor graph.

    Args:
        research_fn: Callable(topic: str) -> str. The research function.

    Returns:
        Compiled LangGraph subgraph.
    """
    def research_node(state: ResearchState) -> dict:
        findings = research_fn(state["topic"])
        return {
            "findings":  findings,
            "messages":  [AIMessage(content=f"Research complete: {findings[:100]}")],
        }

    g = StateGraph(ResearchState)
    g.add_node("research", research_node)
    g.add_edge(START,      "research")
    g.add_edge("research", END)
    return g.compile()


def build_writer_subgraph(write_fn: Callable[[str, str], str]):
    """Build a writer agent subgraph.

    Args:
        write_fn: Callable(topic: str, findings: str) -> str. Drafts content.

    Returns:
        Compiled LangGraph subgraph.
    """
    def write_node(state: WriterState) -> dict:
        draft = write_fn(state["topic"], state.get("findings", ""))
        return {
            "draft":    draft,
            "messages": [AIMessage(content=f"Draft written: {draft[:100]}")],
        }

    g = StateGraph(WriterState)
    g.add_node("write", write_node)
    g.add_edge(START,   "write")
    g.add_edge("write", END)
    return g.compile()


# ══════════════════════════════════════════════════════
# SUPERVISOR PATTERN
# ══════════════════════════════════════════════════════

class SupervisorState(TypedDict):
    """State for a supervisor graph managing multiple agents."""
    messages:      Annotated[list[BaseMessage], add_messages]
    task:          str
    next_agent:    Optional[str]    # supervisor routes to this agent
    research:      Optional[str]
    draft:         Optional[str]
    final_output:  Optional[str]


def build_supervisor_graph(
    supervisor_fn:  Callable[[str, dict], str],
    research_fn:    Callable[[str], str],
    write_fn:       Callable[[str, str], str],
):
    """Build a supervisor + researcher + writer multi-agent graph.

    Flow:
      supervisor -> [route] -> researcher -> supervisor -> writer -> supervisor -> END

    The supervisor decides which agent to call next based on task progress.

    Args:
        supervisor_fn: Callable(task, state) -> next_agent_name.
        research_fn:   Callable(topic) -> findings string.
        write_fn:      Callable(topic, findings) -> draft string.

    Returns:
        Compiled LangGraph graph.
    """
    def supervisor_node(state: SupervisorState) -> dict:
        next_agent = supervisor_fn(state["task"], state)
        return {
            "messages":   [AIMessage(content=f"[SUPERVISOR] Routing to: {next_agent}")],
            "next_agent": next_agent,
        }

    def research_node(state: SupervisorState) -> dict:
        findings = research_fn(state["task"])
        return {
            "research": findings,
            "messages": [AIMessage(content=f"[RESEARCHER] Found: {findings[:100]}")],
        }

    def write_node(state: SupervisorState) -> dict:
        draft = write_fn(state["task"], state.get("research", ""))
        return {
            "draft":    draft,
            "messages": [AIMessage(content=f"[WRITER] Draft: {draft[:100]}")],
        }

    def finalise_node(state: SupervisorState) -> dict:
        output = (
            f"Task: {state['task']}\n\n"
            f"Research:\n{state.get('research', 'N/A')}\n\n"
            f"Draft:\n{state.get('draft', 'N/A')}"
        )
        return {"final_output": output}

    def route_supervisor(state: SupervisorState) -> str:
        return state.get("next_agent", "finish")

    g = StateGraph(SupervisorState)
    g.add_node("supervisor", supervisor_node)
    g.add_node("researcher",  research_node)
    g.add_node("writer",      write_node)
    g.add_node("finish",      finalise_node)

    g.add_edge(START,       "supervisor")
    g.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {"researcher": "researcher", "writer": "writer", "finish": "finish"},
    )
    g.add_edge("researcher", "supervisor")
    g.add_edge("writer",     "supervisor")
    g.add_edge("finish",     END)

    return g.compile()


def build_research_team(research_fn=None, write_fn=None):
    """Build a complete research team with default mock functions.

    Convenience factory for demos and tests.

    Returns:
        Compiled supervisor graph.
    """
    _steps = [0]

    def default_supervisor(task: str, state: dict) -> str:
        _steps[0] += 1
        if not state.get("research"):
            return "researcher"
        elif not state.get("draft"):
            return "writer"
        else:
            return "finish"

    def default_research(topic: str) -> str:
        return f"Research findings on '{topic}': Key facts gathered from 5 sources. Market size: $10B. Growth rate: 15% YoY."

    def default_write(topic: str, findings: str) -> str:
        return f"Executive Summary: {topic}\n\n{findings}\n\nConclusion: Strong opportunity identified."

    return build_supervisor_graph(
        supervisor_fn = default_supervisor,
        research_fn   = research_fn or default_research,
        write_fn      = write_fn or default_write,
    )


# ══════════════════════════════════════════════════════
# AGENT TYPE ENUM
# ══════════════════════════════════════════════════════

class AgentType(Enum):
    """Enumeration of supported agent architecture patterns."""
    REACT            = "react"
    PLAN_AND_EXECUTE = "plan_and_execute"
    AGENTIC_RAG      = "agentic_rag"


# ══════════════════════════════════════════════════════
# AGENTIC RAG
# ══════════════════════════════════════════════════════

class RAGState(TypedDict):
    """State for the Agentic RAG graph."""
    query:         str
    documents:     list[str]
    answer:        str
    needs_rewrite: bool
    iteration:     int


def build_agentic_rag_graph(
    retrieve_fn:  Callable[[str], list[str]],
    generate_fn:  Callable[[str, list[str]], str],
):
    """Build an Agentic RAG graph with retrieve → grade → (rewrite → retrieve)* → generate.

    Flow:
      retrieve_node  — calls retrieve_fn(query) → sets documents
      grade_node     — checks relevance (len(documents) > 0), sets needs_rewrite
      rewrite_node   — rewrites query (appends " detailed explanation")
      generate_node  — calls generate_fn(query, documents) → sets answer

    Conditional: if needs_rewrite and iteration < 2 → rewrite → retrieve
                 else → generate

    Args:
        retrieve_fn: Callable(query: str) -> list[str]. Retrieves relevant documents.
        generate_fn: Callable(query: str, docs: list[str]) -> str. Generates an answer.

    Returns:
        Compiled LangGraph graph.
    """
    def retrieve_node(state: RAGState) -> dict:
        docs = retrieve_fn(state["query"])
        return {"documents": docs}

    def grade_node(state: RAGState) -> dict:
        # Simple relevance check: documents exist
        needs_rewrite = len(state["documents"]) == 0
        return {"needs_rewrite": needs_rewrite}

    def rewrite_node(state: RAGState) -> dict:
        rewritten = state["query"] + " detailed explanation"
        return {"query": rewritten, "iteration": state.get("iteration", 0) + 1}

    def generate_node(state: RAGState) -> dict:
        answer = generate_fn(state["query"], state["documents"])
        return {"answer": answer}

    def route_after_grade(state: RAGState) -> str:
        iteration = state.get("iteration", 0)
        if state.get("needs_rewrite") and iteration < 2:
            return "rewrite"
        return "generate"

    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade",    grade_node)
    g.add_node("rewrite",  rewrite_node)
    g.add_node("generate", generate_node)

    g.add_edge(START,      "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade",
        route_after_grade,
        {"rewrite": "rewrite", "generate": "generate"},
    )
    g.add_edge("rewrite",  "retrieve")
    g.add_edge("generate", END)

    return g.compile()


# ══════════════════════════════════════════════════════
# MAIN DEMO
# ══════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("MULTI-AGENT SYSTEMS DEMO")
    print("=" * 70)

    print("\n[1] Research subgraph")
    def mock_research(topic: str) -> str:
        return f"3 key findings about {topic}: market growing, tech improving, adoption increasing."

    research_graph = build_research_subgraph(mock_research)
    result = research_graph.invoke({
        "messages": [], "topic": "LangGraph multi-agent", "findings": None
    })
    print(f"  Topic: LangGraph multi-agent")
    print(f"  Findings: {result['findings'][:80]}")

    print("\n[2] Supervisor + Researcher + Writer pipeline")
    team = build_research_team()
    result = team.invoke({
        "messages":     [HumanMessage("Research the India AI market")],
        "task":         "India AI market overview",
        "next_agent":   None,
        "research":     None,
        "draft":        None,
        "final_output": None,
    })

    print(f"  Task: 'India AI market overview'")
    print(f"  Final output ({len(result['final_output'])} chars):")
    for line in result['final_output'].split('\n')[:5]:
        print(f"    {line}")

    print(f"\n  Agent messages ({len(result['messages'])} total):")
    for msg in result['messages']:
        print(f"    {msg.content[:80]}")

    print("\n[3] Multi-agent patterns:")
    patterns = [
        ("Supervisor",  "One agent routes to specialised sub-agents based on task state"),
        ("Subgraph",    "Independent graph used as a node inside a parent graph"),
        ("Handoff",     "Agent A explicitly passes control to Agent B via state update"),
        ("Parallel",    "Fan-out: multiple agents run simultaneously, fan-in combines results"),
        ("Reflection",  "Agent critiques its own output and revises until quality threshold"),
    ]
    for name, desc in patterns:
        print(f"  {name:<12}: {desc}")


if __name__ == "__main__":
    main()
