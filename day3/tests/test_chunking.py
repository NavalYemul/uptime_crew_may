"""Tests for day3.chunking — all strategies, metadata extraction, compare."""

import pytest
from day3.chunking import (
    ChunkResult,
    fixed_size_chunk,
    recursive_chunk,
    character_chunk,
    semantic_chunk,
    extract_metadata_from_chunk,
    compare_strategies,
)


SAMPLE = (
    "MacBook Pro 14 M3 Pro chip.\n\n"
    "It delivers 18 hours battery life for machine learning workloads.\n\n"
    "Price: ₹1,99,900. Product ID: P001. Category: Laptop."
)

LONG_TEXT = " ".join([f"Word{i}" for i in range(300)])


# ── fixed_size_chunk ──────────────────────────────────────────

def test_fixed_size_returns_chunks():
    chunks = fixed_size_chunk(SAMPLE, chunk_size=50, overlap=10)
    assert len(chunks) > 0


def test_fixed_size_chunk_ids_unique():
    chunks = fixed_size_chunk(SAMPLE, chunk_size=50, overlap=10)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_fixed_size_overlap_in_metadata():
    chunks = fixed_size_chunk(SAMPLE, chunk_size=50, overlap=10)
    for c in chunks:
        assert "start_char" in c.metadata
        assert "end_char" in c.metadata


def test_fixed_size_chunk_type():
    chunks = fixed_size_chunk(SAMPLE, chunk_size=100, overlap=20)
    for c in chunks:
        assert isinstance(c, ChunkResult)


def test_fixed_size_empty_string():
    chunks = fixed_size_chunk("", chunk_size=200, overlap=40)
    assert chunks == []


def test_fixed_size_word_count_populated():
    chunks = fixed_size_chunk(LONG_TEXT, chunk_size=200, overlap=40)
    for c in chunks:
        assert c.word_count > 0


# ── recursive_chunk ───────────────────────────────────────────

def test_recursive_chunk_returns_chunks():
    chunks = recursive_chunk(SAMPLE, chunk_size=100, overlap=20)
    assert len(chunks) > 0


def test_recursive_chunk_empty_string():
    chunks = recursive_chunk("", chunk_size=200, overlap=40)
    assert chunks == []


def test_recursive_chunk_no_empty_text():
    chunks = recursive_chunk(LONG_TEXT, chunk_size=100, overlap=20)
    for c in chunks:
        assert c.text.strip() != ""


def test_recursive_chunk_paragraph_split():
    # Text with explicit paragraph breaks should split on them
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
    chunks = recursive_chunk(text, chunk_size=400, overlap=0)
    # With large chunk_size the paragraph text should appear in chunks
    combined = " ".join(c.text for c in chunks)
    assert "First paragraph" in combined
    assert "Second paragraph" in combined


def test_recursive_chunk_source_in_id():
    chunks = recursive_chunk(SAMPLE, chunk_size=100, overlap=20, source="mysource")
    for c in chunks:
        assert "mysource" in c.chunk_id


def test_recursive_chunk_strategy_in_metadata():
    chunks = recursive_chunk(SAMPLE, chunk_size=100, overlap=20)
    for c in chunks:
        assert c.metadata["strategy"] == "recursive"


# ── character_chunk ───────────────────────────────────────────

def test_character_chunk_splits_on_separator():
    text = "Part one.\n\nPart two.\n\nPart three."
    chunks = character_chunk(text, separator="\n\n")
    assert len(chunks) == 3


def test_character_chunk_no_empty():
    text = "Part one.\n\nPart two.\n\nPart three."
    chunks = character_chunk(text, separator="\n\n")
    for c in chunks:
        assert c.text.strip() != ""


def test_character_chunk_custom_separator():
    text = "a|b|c|d"
    chunks = character_chunk(text, separator="|")
    assert len(chunks) == 4


def test_character_chunk_empty():
    chunks = character_chunk("", separator="\n\n")
    assert chunks == []


# ── semantic_chunk ────────────────────────────────────────────

def test_semantic_chunk_fallback_no_encoder():
    chunks = semantic_chunk(SAMPLE, encoder=None)
    assert len(chunks) > 0


def test_semantic_chunk_fallback_strategy():
    chunks = semantic_chunk(SAMPLE, encoder=None)
    # Fallback should be recursive or semantic_fallback strategy
    for c in chunks:
        assert c.metadata.get("strategy") in ("recursive", "semantic_fallback")


def test_semantic_chunk_with_encoder(mock_embed):
    chunks = semantic_chunk(SAMPLE, encoder=mock_embed, similarity_threshold=0.5)
    assert len(chunks) > 0


def test_semantic_chunk_empty():
    chunks = semantic_chunk("", encoder=None)
    assert chunks == []


# ── extract_metadata_from_chunk ───────────────────────────────

def test_extract_metadata_has_price():
    text = "MacBook Pro costs ₹1,99,900."
    meta = extract_metadata_from_chunk(text)
    assert meta["has_price"] is True


def test_extract_metadata_no_price():
    text = "This is a plain text document without pricing."
    meta = extract_metadata_from_chunk(text)
    assert meta["has_price"] is False


def test_extract_metadata_detects_product_id():
    text = "Product ID: P001 is in stock."
    meta = extract_metadata_from_chunk(text)
    assert meta["has_product_id"] is True


def test_extract_metadata_no_product_id():
    text = "No product ID here."
    meta = extract_metadata_from_chunk(text)
    assert meta["has_product_id"] is False


def test_extract_metadata_word_count():
    text = "Hello world this is five words"
    meta = extract_metadata_from_chunk(text)
    assert meta["word_count"] == 6


def test_extract_metadata_laptop_category():
    text = "The MacBook Pro laptop has an Intel processor."
    meta = extract_metadata_from_chunk(text)
    assert meta["dominant_category"] == "laptop"


def test_extract_metadata_phone_category():
    text = "The Samsung Galaxy smartphone has AMOLED screen."
    meta = extract_metadata_from_chunk(text)
    assert meta["dominant_category"] == "smartphone"


def test_extract_metadata_source():
    meta = extract_metadata_from_chunk("some text", source_file="products.txt", chunk_index=3)
    assert meta["source"] == "products.txt"
    assert meta["chunk_index"] == 3


# ── compare_strategies ────────────────────────────────────────

def test_compare_strategies_returns_all_four():
    result = compare_strategies(SAMPLE)
    assert "fixed" in result
    assert "recursive" in result
    assert "character" in result
    assert "semantic" in result


def test_compare_strategies_has_required_keys():
    result = compare_strategies(SAMPLE)
    for strategy, stats in result.items():
        assert "num_chunks" in stats
        assert "avg_words" in stats
        assert "min_words" in stats
        assert "max_words" in stats
        assert "sample_chunk" in stats
