"""test_embeddings.py — Embedding model tests (uses mock embeddings, no model download)"""

import numpy as np
import pytest

from day2.embeddings import (
    MODEL_CATALOGUE,
    EmbeddingModelSpec,
    model_selection_guide,
    dense_vs_sparse_comparison,
    BenchmarkResult,
)
from tests.conftest import mock_embed


# ── Model Catalogue ───────────────────────────────────

def test_catalogue_not_empty():
    assert len(MODEL_CATALOGUE) >= 4


def test_catalogue_all_have_required_fields():
    for m in MODEL_CATALOGUE:
        assert isinstance(m.name, str) and len(m.name) > 0
        assert isinstance(m.dims, int) and m.dims > 0
        assert isinstance(m.max_tokens, int) and m.max_tokens > 0
        assert m.speed in {"fast", "medium", "slow", "API"}


def test_catalogue_dims_reasonable():
    for m in MODEL_CATALOGUE:
        if isinstance(m.dims, int):
            assert 128 <= m.dims <= 4096


def test_catalogue_contains_minilm():
    names = [m.name for m in MODEL_CATALOGUE]
    assert any("MiniLM" in n for n in names)


def test_catalogue_contains_openai():
    names = [m.name for m in MODEL_CATALOGUE]
    assert any("text-embedding" in n for n in names)


# ── Model Selection ───────────────────────────────────

def test_model_selection_returns_spec():
    result = model_selection_guide(speed_priority=True)
    assert isinstance(result, EmbeddingModelSpec)


def test_model_selection_speed_has_smallest_dims():
    fast = model_selection_guide(speed_priority=True, budget="free")
    qual = model_selection_guide(speed_priority=False, budget="free")
    # Free/local speed model should have <= dims than quality model
    assert fast.dims <= qual.dims


def test_model_selection_multilingual():
    m = model_selection_guide(multilingual=True)
    assert m.multilingual is True


def test_model_selection_free_returns_local():
    m = model_selection_guide(budget="free")
    # Free model should not be API-only
    assert m.size_mb != "API"


# ── Dense vs Sparse ───────────────────────────────────

def test_dense_vs_sparse_keys(small_corpus):
    r = dense_vs_sparse_comparison(small_corpus)
    assert {"sparse_shape", "sparse_sparsity", "dense_shape"} <= r.keys()


def test_sparse_is_actually_sparse(small_corpus):
    r = dense_vs_sparse_comparison(small_corpus)
    assert r["sparse_sparsity"] > 0.5   # TF-IDF should be >50% zeros


def test_sparse_row_count_matches(small_corpus):
    r = dense_vs_sparse_comparison(small_corpus)
    assert r["sparse_shape"][0] == len(small_corpus)


# ── Mock Embedding Maths ───────────────────────────────

def test_mock_embed_shape(small_corpus):
    vecs = mock_embed(small_corpus)
    assert vecs.shape == (len(small_corpus), 384)


def test_mock_embed_unit_normalised(small_corpus):
    vecs = mock_embed(small_corpus)
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_array_almost_equal(norms, np.ones(len(small_corpus)), decimal=5)


def test_mock_embed_reproducible(small_corpus):
    v1 = mock_embed(small_corpus)
    v2 = mock_embed(small_corpus)
    np.testing.assert_array_equal(v1, v2)


def test_cosine_sim_identical_vectors():
    v = mock_embed(["test sentence"])
    sim = float(v[0] @ v[0])
    assert sim == pytest.approx(1.0, abs=1e-5)


def test_cosine_sim_range(small_corpus):
    vecs = mock_embed(small_corpus)
    sim_matrix = vecs @ vecs.T
    assert sim_matrix.min() >= -1.0 - 1e-5
    assert sim_matrix.max() <=  1.0 + 1e-5


def test_similarity_matrix_symmetric(small_corpus):
    vecs = mock_embed(small_corpus)
    sim  = vecs @ vecs.T
    np.testing.assert_array_almost_equal(sim, sim.T)


# ── BenchmarkResult ───────────────────────────────────

def test_benchmark_result_construction():
    r = BenchmarkResult(
        model_name     = "test-model",
        dims           = 384,
        encode_time_s  = 0.5,
        texts_per_sec  = 100.0,
    )
    assert r.model_name == "test-model"
    assert r.dims == 384
    assert r.texts_per_sec == pytest.approx(100.0)
