import pytest
from day5.streaming import (
    StreamChunk, stream_tokens, stream_with_sources,
    collect_stream, StreamingOrchestrator,
    fastapi_streaming_example, vercel_ai_sdk_example
)


class TestStreamChunk:
    def test_to_sse_format(self):
        chunk = StreamChunk(chunk_type="token", content="hello")
        sse = chunk.to_sse()
        assert sse.startswith("data:")
        assert "hello" in sse
        assert sse.endswith("\n\n")

    def test_metadata_included(self):
        chunk = StreamChunk(chunk_type="done", content="", metadata={"tokens": 5})
        sse = chunk.to_sse()
        assert "tokens" in sse


class TestStreamTokens:
    def test_yields_token_chunks(self):
        chunks = list(stream_tokens("hello world"))
        token_chunks = [c for c in chunks if c.chunk_type == "token"]
        assert len(token_chunks) == 2  # "hello" and "world"

    def test_ends_with_done(self):
        chunks = list(stream_tokens("test"))
        assert chunks[-1].chunk_type == "done"

    def test_reassembled_text(self):
        chunks = list(stream_tokens("hello world test"))
        text = "".join(c.content for c in chunks if c.chunk_type == "token")
        assert "hello" in text
        assert "world" in text


class TestCollectStream:
    def test_collects_answer(self):
        gen = stream_tokens("the answer is 42")
        result = collect_stream(gen)
        assert "answer" in result["answer"]

    def test_collects_sources(self):
        gen = stream_with_sources("answer text", ["source1", "source2"])
        result = collect_stream(gen)
        assert len(result["sources"]) == 2


class TestStreamingOrchestrator:
    def test_yields_sse_strings(self, mock_retrieve_fn, mock_synthesize_fn):
        from day5.multi_agent_orchestrator import build_orchestrator
        graph = build_orchestrator(
            retrieve_fn=mock_retrieve_fn,
            synthesize_fn=mock_synthesize_fn
        )
        orch = StreamingOrchestrator(graph)
        chunks = list(orch.stream("hybrid search"))
        assert len(chunks) > 0
        assert all(c.startswith("data:") for c in chunks)


class TestCodeExamples:
    def test_fastapi_example_has_streaming_response(self):
        code = fastapi_streaming_example()
        assert "StreamingResponse" in code
        assert "text/event-stream" in code

    def test_vercel_example_has_usechat(self):
        code = vercel_ai_sdk_example()
        assert "useChat" in code
