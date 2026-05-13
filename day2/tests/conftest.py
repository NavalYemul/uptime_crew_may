"""conftest.py — Shared fixtures (no real model downloads in tests)"""

import numpy as np
import pandas as pd
import pytest


def mock_embed(texts: list[str], dim: int = 384) -> np.ndarray:
    """Reproducible mock embeddings — no model download required."""
    key = sum(hash(t) for t in texts) % (2**32)
    rng = np.random.default_rng(key)
    vecs = rng.standard_normal((len(texts), dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-9)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame({
        "id":        [f"TXN-{i:06d}" for i in range(n)],
        "name":      [f"Customer {i}" for i in range(n)],
        "amount":    rng.uniform(10, 5000, n).round(2),
        "category":  np.tile(["sales","refund","subscription","ad_spend","royalty"], 10),
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
        "year":      2024,
        "month":     np.repeat(np.arange(1, 6), 10),
    })


@pytest.fixture
def small_corpus() -> list[str]:
    return [
        "Machine learning learns patterns from data.",
        "Deep learning uses neural networks with layers.",
        "Natural language processing handles text.",
        "Word embeddings map words to vectors.",
        "Vector databases enable fast similarity search.",
        "RAG retrieves context before generating answers.",
        "ETL pipelines extract transform and load data.",
        "Batch processing handles data in bulk jobs.",
        "Stream processing handles real-time data flow.",
        "HNSW algorithm enables approximate nearest neighbour search.",
    ]


@pytest.fixture
def mock_embeddings(small_corpus) -> np.ndarray:
    return mock_embed(small_corpus)
