"""
Multi-agent orchestrator using LangGraph.

Architecture:
    query_rewrite → retrieval → [conditional: api or synthesize] → synthesis

Also includes a supervisor pattern for parallel agent execution.
"""

from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


# ---------------------------------------------------------------------------
# Shared state flowing through all agents
# ---------------------------------------------------------------------------

class OrchestratorState(TypedDict):
    query: str
    rewritten_query: Optional[str]
    retrieved_docs: list[str]
    api_result: Optional[str]
    synthesis: Optional[str]
    messages: Annotated[list[BaseMessage], add_messages]
    source: str  # "local", "api", "web", "hybrid"
    cost_tokens: int
    trace_id: Optional[str]


# ---------------------------------------------------------------------------
# Node builders
# ---------------------------------------------------------------------------

def build_query_rewriter(rewrite_fn=None):
    """Returns a node that rewrites the query for better retrieval."""
    def query_rewrite_node(state: OrchestratorState) -> OrchestratorState:
        query = state["query"]
        if rewrite_fn:
            rewritten = rewrite_fn(query)
        else:
            # Default: expand abbreviations, add context
            rewritten = query + " detailed explanation"
        return {"rewritten_query": rewritten}
    return query_rewrite_node


def build_retrieval_agent(retrieve_fn=None):
    """Returns a node that retrieves relevant documents."""
    def retrieval_node(state: OrchestratorState) -> OrchestratorState:
        q = state.get("rewritten_query") or state["query"]
        if retrieve_fn:
            docs = retrieve_fn(q)
        else:
            docs = [f"Mock doc about: {q}"]
        return {"retrieved_docs": docs, "source": "local"}
    return retrieval_node


def build_api_agent(api_fn=None):
    """Returns a node that calls an external API for structured data."""
    def api_node(state: OrchestratorState) -> OrchestratorState:
        query = state["query"]
        if api_fn:
            result = api_fn(query)
        else:
            result = f"API result for: {query}"
        return {"api_result": result}
    return api_node


def build_synthesis_agent(synthesize_fn=None):
    """Returns a node that synthesizes final answer from docs + api result."""
    def synthesis_node(state: OrchestratorState) -> OrchestratorState:
        docs = state.get("retrieved_docs", [])
        api = state.get("api_result", "")
        query = state["query"]
        if synthesize_fn:
            answer = synthesize_fn(query, docs, api)
        else:
            parts = []
            if docs:
                parts.append(f"From docs: {docs[0][:80]}")
            if api:
                parts.append(f"From API: {api[:60]}")
            answer = " | ".join(parts) if parts else f"No data found for: {query}"
        tokens = len(query.split()) + sum(len(d.split()) for d in docs)
        return {
            "synthesis": answer,
            "cost_tokens": tokens,
            "messages": [AIMessage(content=answer)]
        }
    return synthesis_node


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_retrieval(state: OrchestratorState) -> str:
    """Route to API if retrieval found nothing, else go straight to synthesis."""
    if not state.get("retrieved_docs"):
        return "api"
    return "synthesize"


# ---------------------------------------------------------------------------
# Main orchestrator builder
# ---------------------------------------------------------------------------

def build_orchestrator(
    rewrite_fn=None,
    retrieve_fn=None,
    api_fn=None,
    synthesize_fn=None
):
    """
    Build full multi-agent orchestrator:
    query_rewrite → retrieval → [conditional: api or synthesize] → synthesis
    """
    g = StateGraph(OrchestratorState)
    g.add_node("query_rewrite", build_query_rewriter(rewrite_fn))
    g.add_node("retrieval",     build_retrieval_agent(retrieve_fn))
    g.add_node("api_agent",     build_api_agent(api_fn))
    g.add_node("synthesis",     build_synthesis_agent(synthesize_fn))

    g.add_edge(START, "query_rewrite")
    g.add_edge("query_rewrite", "retrieval")
    g.add_conditional_edges("retrieval", route_after_retrieval, {
        "api": "api_agent",
        "synthesize": "synthesis"
    })
    g.add_edge("api_agent", "synthesis")
    g.add_edge("synthesis", END)

    return g.compile()


def run_orchestrator(graph, query: str) -> dict:
    """Run the orchestrator and return the final state."""
    initial = OrchestratorState(
        query=query,
        rewritten_query=None,
        retrieved_docs=[],
        api_result=None,
        synthesis=None,
        messages=[HumanMessage(content=query)],
        source="local",
        cost_tokens=0,
        trace_id=None
    )
    return graph.invoke(initial)


# ---------------------------------------------------------------------------
# Supervisor pattern
# ---------------------------------------------------------------------------

class SupervisorState(TypedDict):
    task: str
    agent_results: list[str]
    final_answer: str


def build_supervisor_orchestrator(agents: list):
    """
    Run multiple agents and let supervisor pick the best result.
    agents: list of (name, fn) tuples
    """
    def supervisor_node(state: SupervisorState) -> SupervisorState:
        results = []
        for name, fn in agents:
            try:
                result = fn(state["task"])
                results.append(f"[{name}]: {result}")
            except Exception as e:
                results.append(f"[{name}]: error - {e}")

        # Supervisor picks: prefer longest non-error result
        valid = [r for r in results if "error" not in r]
        best = max(valid, key=len) if valid else results[0]
        return {
            "agent_results": results,
            "final_answer": best
        }

    g = StateGraph(SupervisorState)
    g.add_node("supervisor", supervisor_node)
    g.set_entry_point("supervisor")
    g.add_edge("supervisor", END)
    return g.compile()
