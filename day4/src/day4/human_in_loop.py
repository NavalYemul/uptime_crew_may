"""
human_in_loop.py — Human-in-the-Loop (HITL) Patterns
=====================================================
Covers: interrupt(), Command, approval workflows, resume/reject,
        breakpoints, streaming with HITL.

Why HITL?
  Fully autonomous agents make mistakes. In high-stakes domains:
  - Medical: doctor must review AI diagnosis before acting
  - Finance: compliance officer must approve large transactions
  - Legal: lawyer reviews AI-drafted contract before sending
  - Support: agent drafts reply, human approves before sending to customer

  LangGraph's interrupt() pauses execution at any node and returns
  control to your application code. Command(resume=...) continues.

Industry standard:
  All production agents from Salesforce, Workday, ServiceNow etc.
  include HITL approval steps for high-risk actions.

Run: python -m day4.human_in_loop
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict, Annotated, Optional, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# ══════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════

class HITLState(TypedDict):
    """State for HITL workflows. Includes approval tracking."""
    messages:      Annotated[list[BaseMessage], add_messages]
    draft_action:  Optional[str]      # action the AI wants to take
    approved:      Optional[bool]     # human decision
    final_result:  Optional[str]      # outcome after approval


# ══════════════════════════════════════════════════════
# HITL GRAPH
# ══════════════════════════════════════════════════════

def build_hitl_graph(llm_fn, checkpointer=None):
    """Build a graph with a human approval step.

    Flow:
      START -> draft_action -> interrupt (human reviews) -> execute_or_reject -> END

    The interrupt() in human_review_node pauses execution and returns
    control to the caller. The caller inspects the draft, then resumes
    with Command(resume=True/False).

    Args:
        llm_fn:       Callable(messages) -> str. Generates the draft action.
        checkpointer: Required for interrupt() to work (stores graph state).
                      Defaults to a new MemorySaver() if None.

    Returns:
        Compiled LangGraph graph with HITL capability.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    def draft_action_node(state: HITLState) -> dict:
        """Agent drafts an action and stores it for human review."""
        response = llm_fn(state["messages"])
        return {
            "messages":     [AIMessage(content=f"[DRAFT] {response}")],
            "draft_action": response,
        }

    def human_review_node(state: HITLState) -> dict:
        """Pause here for human review. interrupt() suspends execution.

        The graph state is checkpointed at this point.
        Caller resumes with: graph.invoke(Command(resume=True), config=config)
        """
        approved = interrupt(
            value={
                "action":   state.get("draft_action", ""),
                "question": "Do you approve this action? (True/False)",
            }
        )
        return {"approved": approved}

    def execute_or_reject_node(state: HITLState) -> dict:
        """Execute the action if approved, reject otherwise."""
        if state.get("approved"):
            return {
                "messages":     [AIMessage(content=f"Action executed: {state['draft_action']}")],
                "final_result": f"EXECUTED: {state['draft_action']}",
            }
        else:
            return {
                "messages":     [AIMessage(content="Action rejected by human reviewer.")],
                "final_result": "REJECTED",
            }

    g = StateGraph(HITLState)
    g.add_node("draft_action",        draft_action_node)
    g.add_node("human_review",        human_review_node)
    g.add_node("execute_or_reject",   execute_or_reject_node)

    g.add_edge(START,               "draft_action")
    g.add_edge("draft_action",      "human_review")
    g.add_edge("human_review",      "execute_or_reject")
    g.add_edge("execute_or_reject", END)

    return g.compile(checkpointer=checkpointer, interrupt_before=["human_review"])


@dataclass
class ApprovalWorkflow:
    """Helper class for managing HITL approval workflows.

    Wraps the interrupt/resume pattern into a clean API:
      workflow = ApprovalWorkflow(graph)
      draft = workflow.submit(query, thread_id)  # pauses at human_review
      result = workflow.approve(thread_id)        # resumes with approval
      # OR
      result = workflow.reject(thread_id)         # resumes with rejection
    """

    graph: Any

    def submit(self, user_query: str, thread_id: str) -> Optional[str]:
        """Submit a request. Returns the draft action for human review.

        Args:
            user_query: The user's request.
            thread_id:  Unique thread identifier.

        Returns:
            The drafted action text (None if no draft generated).
        """
        config = {"configurable": {"thread_id": thread_id}}
        try:
            result = self.graph.invoke(
                {
                    "messages":     [HumanMessage(content=user_query)],
                    "draft_action": None,
                    "approved":     None,
                    "final_result": None,
                },
                config=config
            )
            # Graph paused at interrupt — check state for draft
            state = self.graph.get_state(config)
            return state.values.get("draft_action")
        except Exception:
            state = self.graph.get_state(config)
            return state.values.get("draft_action")

    def approve(self, thread_id: str) -> str:
        """Resume with human approval (True).

        Args:
            thread_id: The thread to resume.

        Returns:
            final_result string.
        """
        return self._resume(thread_id, approved=True)

    def reject(self, thread_id: str) -> str:
        """Resume with human rejection (False).

        Args:
            thread_id: The thread to resume.

        Returns:
            final_result string.
        """
        return self._resume(thread_id, approved=False)

    def _resume(self, thread_id: str, approved: bool) -> str:
        config = {"configurable": {"thread_id": thread_id}}
        try:
            result = self.graph.invoke(Command(resume=approved), config=config)
            return result.get("final_result", "COMPLETED")
        except Exception as e:
            return f"ERROR: {e}"


# ══════════════════════════════════════════════════════
# MAIN DEMO
# ══════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("HUMAN-IN-THE-LOOP (HITL) DEMO")
    print("=" * 70)

    actions = ["Send email to all 5000 customers with 20% discount offer",
               "Delete all logs older than 30 days from production",
               "Approve loan application #LN-20240915 for Rs 50,00,000"]

    call_count = [0]
    def mock_agent(messages):
        call_count[0] += 1
        last = messages[-1].content if messages else ""
        return actions[call_count[0] % len(actions)]

    checkpointer = MemorySaver()
    graph = build_hitl_graph(mock_agent, checkpointer=checkpointer)
    workflow = ApprovalWorkflow(graph)

    print("\n[1] Submit request (graph pauses at human_review)")
    draft = workflow.submit("Process high-value loan application", "thread_hitl_001")
    print(f"  Draft action: '{draft}'")
    print(f"  Graph paused — waiting for human decision")

    print("\n[2] Human APPROVES")
    result_approved = workflow.approve("thread_hitl_001")
    print(f"  Result: {result_approved}")

    print("\n[3] Different thread — human REJECTS")
    call_count[0] = 0
    draft2 = workflow.submit("Send bulk email campaign", "thread_hitl_002")
    print(f"  Draft: '{draft2}'")
    result_rejected = workflow.reject("thread_hitl_002")
    print(f"  Result: {result_rejected}")

    print("\n[4] HITL use cases in production:")
    use_cases = [
        ("Finance",    "Large transaction approval before execution"),
        ("Legal",      "Contract draft review before sending"),
        ("Medical",    "AI diagnosis validation before treatment recommendation"),
        ("Support",    "Customer reply approval before sending"),
        ("Data",       "Schema change approval before migration"),
    ]
    for domain, use_case in use_cases:
        print(f"  [{domain}] {use_case}")


if __name__ == "__main__":
    main()
