import pytest
from langchain_core.messages import HumanMessage


@pytest.fixture
def sample_docs():
    return [
        "LangGraph is a framework for building stateful multi-agent workflows",
        "BM25 is a ranking algorithm based on probabilistic retrieval framework",
        "FastAPI is a modern Python web framework for building REST APIs",
        "Hybrid search combines keyword and semantic retrieval methods",
        "RAGAS evaluates RAG systems using faithfulness and answer relevancy",
        "Docker containers package applications with all their dependencies",
        "LangSmith provides tracing and evaluation for LLM applications",
        "Reciprocal Rank Fusion merges multiple ranked lists into one",
    ]


@pytest.fixture
def sample_query():
    return "How does hybrid search work?"


@pytest.fixture
def mock_retrieve_fn(sample_docs):
    def retrieve(query):
        return [d for d in sample_docs if any(w in d.lower() for w in query.lower().split())][:3]
    return retrieve


@pytest.fixture
def mock_synthesize_fn():
    def synthesize(query, docs, api_result):
        parts = [docs[0][:50] if docs else "no docs"]
        if api_result:
            parts.append(api_result[:30])
        return f"Answer: {' + '.join(parts)}"
    return synthesize
