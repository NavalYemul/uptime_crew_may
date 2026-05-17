"""
Streaming response helpers for FastAPI SSE (Server-Sent Events).

Provides token-by-token streaming compatible with:
- FastAPI StreamingResponse
- Vercel AI SDK useChat hook
- Any SSE consumer
"""

import json
import time
from typing import Generator, AsyncGenerator, Optional
from dataclasses import dataclass, asdict


# ---------------------------------------------------------------------------
# StreamChunk dataclass
# ---------------------------------------------------------------------------

@dataclass
class StreamChunk:
    """A single chunk in a streaming response."""
    chunk_type: str   # "token", "source", "done", "error"
    content: str
    metadata: dict = None

    def to_sse(self) -> str:
        """Format as Server-Sent Events string."""
        data = {"type": self.chunk_type, "content": self.content}
        if self.metadata:
            data["metadata"] = self.metadata
        return f"data: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Generator helpers
# ---------------------------------------------------------------------------

def stream_tokens(text: str, delay: float = 0.0) -> Generator[StreamChunk, None, None]:
    """
    Simulate token-by-token streaming.
    In production: use LLM stream=True and yield each chunk.
    """
    words = text.split()
    for i, word in enumerate(words):
        chunk = word + (" " if i < len(words) - 1 else "")
        yield StreamChunk(chunk_type="token", content=chunk)
        if delay > 0:
            time.sleep(delay)
    yield StreamChunk(chunk_type="done", content="", metadata={"total_tokens": len(words)})


def stream_with_sources(
    answer: str,
    sources: list[str],
    delay: float = 0.0
) -> Generator[StreamChunk, None, None]:
    """Stream answer tokens, then append source metadata."""
    yield from stream_tokens(answer, delay)
    for src in sources:
        yield StreamChunk(chunk_type="source", content=src)


def collect_stream(gen: Generator[StreamChunk, None, None]) -> dict:
    """Collect a stream into a final result dict."""
    tokens = []
    sources = []
    metadata = {}
    for chunk in gen:
        if chunk.chunk_type == "token":
            tokens.append(chunk.content)
        elif chunk.chunk_type == "source":
            sources.append(chunk.content)
        elif chunk.chunk_type == "done":
            metadata = chunk.metadata or {}
    return {
        "answer": "".join(tokens),
        "sources": sources,
        "metadata": metadata
    }


# ---------------------------------------------------------------------------
# Streaming orchestrator wrapper
# ---------------------------------------------------------------------------

class StreamingOrchestrator:
    """
    Wraps the multi-agent orchestrator to produce a streaming response.
    Compatible with FastAPI StreamingResponse and Vercel AI SDK.
    """

    def __init__(self, orchestrator_graph):
        self.graph = orchestrator_graph

    def stream(self, query: str) -> Generator[str, None, None]:
        """
        Run orchestrator, then stream the synthesis as SSE.
        Yields raw SSE strings for use with FastAPI StreamingResponse.
        """
        # Run the full orchestrator (non-streaming internally)
        result = self.graph.invoke({
            "query": query,
            "rewritten_query": None,
            "retrieved_docs": [],
            "api_result": None,
            "synthesis": None,
            "messages": [],
            "source": "local",
            "cost_tokens": 0,
            "trace_id": None
        })

        answer = result.get("synthesis", "No answer generated")
        docs   = result.get("retrieved_docs", [])

        # Stream the answer word by word
        for chunk in stream_with_sources(answer, docs[:2]):
            yield chunk.to_sse()


# ---------------------------------------------------------------------------
# Code pattern examples (returned as strings for notebooks / docs)
# ---------------------------------------------------------------------------

def fastapi_streaming_example() -> str:
    """Returns a code string showing FastAPI SSE streaming pattern."""
    return '''
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from day5.streaming import StreamingOrchestrator

app = FastAPI()

@app.get("/stream")
async def stream_answer(query: str):
    """Stream answer tokens via Server-Sent Events."""
    def generate():
        orchestrator = StreamingOrchestrator(graph)
        yield from orchestrator.stream(query)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
'''


def vercel_ai_sdk_example() -> str:
    """Returns a code string showing Vercel AI SDK integration."""
    return '''
// React component using Vercel AI SDK useChat hook
import { useChat } from "ai/react";

export default function ChatUI() {
  const { messages, input, handleInputChange, handleSubmit } = useChat({
    api: "http://localhost:8000/stream",  // Points to FastAPI SSE endpoint
  });

  return (
    <div>
      {messages.map(m => (
        <div key={m.id}>
          <strong>{m.role}:</strong> {m.content}
        </div>
      ))}
      <form onSubmit={handleSubmit}>
        <input value={input} onChange={handleInputChange} />
        <button type="submit">Send</button>
      </form>
    </div>
  );
}
'''
