"""conftest.py — Shared fixtures (no real model downloads in tests)"""

import pytest
import numpy as np


@pytest.fixture
def mock_embed():
    """No model download in tests — deterministic random embeddings."""
    def _embed(texts):
        vecs = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**31))
            v = rng.standard_normal(384).astype(np.float32)
            v = v / np.linalg.norm(v)
            vecs.append(v)
        return np.array(vecs)
    return _embed


@pytest.fixture
def sample_corpus():
    return [
        "Samsung Galaxy S24 Ultra features a 200MP camera and Snapdragon 8 Gen 3 processor.",
        "Apple iPhone 15 Pro Max has A17 Pro chip and titanium design with 5x optical zoom.",
        "MacBook Pro 14 M3 Pro chip delivers 18 hours battery life for machine learning workloads.",
        "Sony WH-1000XM5 headphones offer industry-leading active noise cancellation.",
        "Dell XPS 15 with OLED display and RTX 4070 is ideal for content creation.",
        "OnePlus 12 has 100W fast charging and Hasselblad camera system.",
        "iPad Pro 12.9 M2 chip with Apple Pencil hover support for creative professionals.",
        "Logitech MX Master 3S mouse with MagSpeed scroll wheel for productivity.",
    ]


@pytest.fixture
def sample_qa_pairs():
    from day3.evaluation import QAPair
    return [
        QAPair("best laptop for ML?", "MacBook Pro M3 is best for ML with 18GB RAM.", "laptop"),
        QAPair("noise cancelling headphones?", "Sony WH-1000XM5 has industry-leading ANC.", "headphones"),
    ]
