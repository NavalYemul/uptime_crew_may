"""
memory_persistence.py — LangGraph Memory & Persistence
=======================================================
Covers: InMemorySaver, SqliteSaver, thread_id, checkpointing,
        conversation history, time travel, state snapshots.

Why persistence?
  Without a checkpointer, each LangGraph .invoke() starts fresh.
  With InMemorySaver(thread_id="user_123"), the graph remembers prior messages.
  With SqliteSaver, memory persists across process restarts.

Industry use:
  Every chatbot needs thread-level memory. In production:
  - InMemorySaver: single-instance apps (dev/testing)
  - SqliteSaver:   single-user persisted apps
  - PostgresSaver: multi-user production (PlanetScale, Supabase)
  - Databricks:    Delta Lake checkpoints for distributed agents

Run: python -m day4.memory_persistence
"""

from __future__ import annotations

from typing import TypedDict, Annotated, Optional, Callable
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


# ══════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════

class ChatState(TypedDict):
    """Chat state. Messages use add_messages reducer so they accumulate."""
    messages: Annotated[list[BaseMessage], add_messages]


# ══════════════════════════════════════════════════════
# CHATBOT GRAPH BUILDER
# ══════════════════════════════════════════════════════

def build_chatbot_graph(llm_fn: Callable[[list[BaseMessage]], str], checkpointer=None):
    """Build a basic chatbot graph with optional persistence.

    Args:
        llm_fn:       Callable(messages: list) -> str. The LLM call.
        checkpointer: LangGraph checkpointer (InMemorySaver, SqliteSaver, etc.)
                      Pass None for stateless (no memory).

    Returns:
        Compiled LangGraph graph.
    """
    def chat_node(state: ChatState) -> dict:
        """Call the LLM with all messages in history."""
        response_text = llm_fn(state["messages"])
        return {"messages": [AIMessage(content=response_text)]}

    g = StateGraph(ChatState)
    g.add_node("chat", chat_node)
    g.add_edge(START, "chat")
    g.add_edge("chat", END)

    return g.compile(checkpointer=checkpointer)


def chat_turn(graph, user_message: str, thread_id: str) -> str:
    """Send one message and get a reply, preserving conversation history.

    The thread_id uniquely identifies this conversation. All messages for
    this thread are automatically stored and retrieved by the checkpointer.

    Args:
        graph:        Compiled chatbot graph with a checkpointer.
        user_message: The user's message text.
        thread_id:    Unique conversation identifier.

    Returns:
        The AI's reply text.
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_message)]},
        config=config
    )
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content
    return ""


def get_conversation_history(graph, thread_id: str) -> list[dict]:
    """Retrieve the full message history for a thread.

    Uses LangGraph's get_state() to read the checkpointed state
    without running any new nodes.

    Args:
        graph:     Compiled chatbot graph with a checkpointer.
        thread_id: The conversation thread to retrieve.

    Returns:
        List of {"role": "human"|"ai", "content": str} dicts.
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = graph.get_state(config)
        messages = state.values.get("messages", [])
        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "ai", "content": msg.content})
        return history
    except Exception:
        return []


def get_state_history(graph, thread_id: str) -> list[dict]:
    """Get all checkpoint snapshots for time travel.

    Each turn creates a new checkpoint. get_state_history() returns
    all of them in reverse chronological order.

    Args:
        graph:     Compiled chatbot graph with a checkpointer.
        thread_id: The conversation thread.

    Returns:
        List of checkpoint summary dicts with step and message count.
    """
    config = {"configurable": {"thread_id": thread_id}}
    history = []
    try:
        for snapshot in graph.get_state_history(config):
            messages = snapshot.values.get("messages", [])
            history.append({
                "step":          snapshot.metadata.get("step", 0),
                "message_count": len(messages),
                "checkpoint_id": snapshot.config.get("configurable", {}).get("checkpoint_id", ""),
            })
    except Exception:
        pass
    return history


# ══════════════════════════════════════════════════════
# SHORT-TERM MEMORY
# ══════════════════════════════════════════════════════

class ShortTermMemory:
    """In-memory short-term memory backed by a plain Python list.

    Suitable for within-session conversation context where you want
    fast access to the N most recent messages without persistence.
    """

    def __init__(self):
        self._messages: list[str] = []

    def add(self, message: str) -> None:
        """Append a message to memory."""
        self._messages.append(message)

    def get_recent(self, n: int) -> list[str]:
        """Return the n most recent messages.

        Args:
            n: Maximum number of messages to return.

        Returns:
            List of the n most recent messages (oldest first).
        """
        return self._messages[-n:] if n > 0 else []

    def clear(self) -> None:
        """Remove all messages from memory."""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)


# ══════════════════════════════════════════════════════
# LONG-TERM MEMORY
# ══════════════════════════════════════════════════════

class LongTermMemory:
    """In-memory long-term memory backed by a plain Python dict.

    Uses keyword matching for retrieval. In production, replace with
    a vector database (Pinecone, Weaviate, pgvector) for semantic search.
    """

    def __init__(self):
        self._store: dict[str, str] = {}

    def store(self, key: str, value: str) -> None:
        """Store a key-value pair.

        Args:
            key:   Identifier for the memory entry (e.g. "user_preference_color").
            value: The value to remember.
        """
        self._store[key] = value

    def retrieve(self, key: str) -> Optional[str]:
        """Retrieve a value by exact key.

        Args:
            key: The key to look up.

        Returns:
            The stored value, or None if not found.
        """
        return self._store.get(key)

    def search(self, query: str) -> list[dict[str, str]]:
        """Search for entries where key or value contains any query word.

        Simple keyword matching — good enough for demos and tests.
        In production: use embeddings + cosine similarity.

        Args:
            query: Space-separated keywords to search for.

        Returns:
            List of {"key": ..., "value": ...} dicts matching the query.
        """
        keywords = query.lower().split()
        results = []
        for k, v in self._store.items():
            combined = (k + " " + v).lower()
            if any(kw in combined for kw in keywords):
                results.append({"key": k, "value": v})
        return results

    def __len__(self) -> int:
        return len(self._store)


# ══════════════════════════════════════════════════════
# BUFFER MEMORY CHAIN (pure Python, no LangChain)
# ══════════════════════════════════════════════════════

def build_buffer_memory_chain(llm_fn: Callable[[list[dict]], str]):
    """Return a chat function that maintains a conversation buffer.

    Demonstrates the ConversationBufferMemory concept using only a plain
    Python list — no LangChain dependency required for understanding the idea.

    Args:
        llm_fn: Callable(messages: list[dict]) -> str.
                Each dict has "role" ("user"/"assistant") and "content".

    Returns:
        Callable(user_message: str) -> str that accumulates history.
    """
    history: list[dict] = []

    def chat(user_message: str) -> str:
        history.append({"role": "user", "content": user_message})
        reply = llm_fn(history)
        history.append({"role": "assistant", "content": reply})
        return reply

    return chat


# ══════════════════════════════════════════════════════
# MAIN DEMO
# ══════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("MEMORY & PERSISTENCE DEMO")
    print("=" * 70)

    # Mock LLM that echoes with a response
    call_count = [0]
    def mock_llm(messages: list) -> str:
        call_count[0] += 1
        last = messages[-1].content if messages else ""
        responses = [
            "Hi! I'm your assistant. How can I help?",
            f"You asked about '{last[:40]}'. Here's what I know: This is a mock response.",
            "Interesting follow-up! Based on our conversation so far, I can tell you more.",
            "Great question! In context of everything we've discussed, the answer is 42.",
        ]
        return responses[min(call_count[0] - 1, len(responses) - 1)]

    print("\n[1] Stateless graph (no memory)")
    stateless = build_chatbot_graph(mock_llm)
    r1 = stateless.invoke({"messages": [HumanMessage("Hello!")]})
    print(f"  Turn 1: {r1['messages'][-1].content[:60]}")

    print("\n[2] With InMemorySaver (remembers across turns)")
    checkpointer = MemorySaver()
    call_count[0] = 0
    chatbot = build_chatbot_graph(mock_llm, checkpointer=checkpointer)

    thread = "demo_thread_001"
    reply1 = chat_turn(chatbot, "My name is Naval.", thread)
    reply2 = chat_turn(chatbot, "What is LangGraph?",  thread)
    reply3 = chat_turn(chatbot, "Tell me more.",       thread)

    print(f"  User: 'My name is Naval.'")
    print(f"  AI:   '{reply1[:60]}'")
    print(f"  User: 'What is LangGraph?'")
    print(f"  AI:   '{reply2[:60]}'")
    print(f"  User: 'Tell me more.'")
    print(f"  AI:   '{reply3[:60]}'")

    print("\n[3] Conversation history (3 turns = 6 messages)")
    history = get_conversation_history(chatbot, thread)
    print(f"  Total messages stored: {len(history)}")
    for msg in history:
        print(f"  [{msg['role'].upper()}] {msg['content'][:60]}")

    print("\n[4] State snapshots (time travel)")
    snapshots = get_state_history(chatbot, thread)
    print(f"  Total checkpoints: {len(snapshots)}")
    for s in snapshots[:3]:
        print(f"  Step {s['step']}: {s['message_count']} messages")

    print("\n[5] Different thread (isolated from thread_001)")
    call_count[0] = 0
    reply_other = chat_turn(chatbot, "Hello, I'm a new user.", "other_thread_002")
    other_history = get_conversation_history(chatbot, "other_thread_002")
    print(f"  thread_002 messages: {len(other_history)} (isolated from thread_001)")


if __name__ == "__main__":
    main()
