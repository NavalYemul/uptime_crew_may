"""
chunking.py — Document Chunking Strategies for RAG
====================================================
Covers: fixed-size, recursive, character, semantic chunking.
Shows how chunk strategy affects retrieval quality.

Why chunking matters:
  - LLMs have context windows. You cannot feed a 100-page PDF as-is.
  - Smaller, focused chunks = more precise retrieval (less noise for the LLM).
  - Bad chunking = answers that miss key sentences split across boundaries.
  - The right strategy depends on your document structure and retrieval needs.

Run: python -m day3.chunking
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional


# ══════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════

@dataclass
class ChunkResult:
    """Represents a single chunk produced by any chunking strategy.

    Fields:
        text       : The actual chunk text.
        chunk_id   : Unique identifier for this chunk (source + index).
        start_char : Character offset where this chunk starts in the original.
        end_char   : Character offset where this chunk ends in the original.
        word_count : Number of whitespace-separated words in this chunk.
        metadata   : Arbitrary key-value metadata (source, strategy, etc.).
    """
    text:       str
    chunk_id:   str
    start_char: int
    end_char:   int
    word_count: int
    metadata:   dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════
# CHUNKING STRATEGIES
# ══════════════════════════════════════════════════════

def fixed_size_chunk(
    text:       str,
    chunk_size: int = 400,
    overlap:    int = 80,
    source:     str = "doc",
) -> list[ChunkResult]:
    """Split text into fixed-size character windows with overlap.

    Strategy: Fixed-Size (Character-Based)
    ---------------------------------------
    Splits text every `chunk_size` characters using a sliding window.
    Each subsequent chunk starts `chunk_size - overlap` characters after
    the previous one, so the tail of one chunk becomes the head of the next.

    Pros:
      - Extremely fast. O(n) time, minimal memory.
      - Predictable chunk sizes — good when you need uniform token budgets.
      - Works on any text format (code, raw logs, binary-decoded text).

    Cons:
      - Completely ignores sentence and paragraph boundaries.
      - A sentence like "The price is ₹24,900" may be split as
        "The price is ₹24," and "900 for AirPods Pro."
      - Retrieval quality suffers when answers span a boundary.

    When to use:
      - First-pass prototyping.
      - Text where structure is meaningless (log files, genomic sequences).
      - When downstream reranking can compensate for boundary noise.

    Args:
        text:       Input document text.
        chunk_size: Maximum characters per chunk.
        overlap:    Number of characters shared between consecutive chunks.
        source:     Identifier for the source document (stored in metadata).

    Returns:
        List of ChunkResult objects with character offsets.
    """
    if not text.strip():
        return []

    chunks = []
    start = 0
    idx = 0
    step = max(1, chunk_size - overlap)

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]

        if chunk_text.strip():
            chunks.append(ChunkResult(
                text=chunk_text,
                chunk_id=f"{source}_fixed_{idx:04d}",
                start_char=start,
                end_char=end,
                word_count=len(chunk_text.split()),
                metadata={
                    "strategy": "fixed_size",
                    "source": source,
                    "chunk_index": idx,
                    "start_char": start,
                    "end_char": end,
                },
            ))
        start += step
        idx += 1

    return chunks


def recursive_chunk(
    text:       str,
    chunk_size: int = 400,
    overlap:    int = 80,
    source:     str = "doc",
) -> list[ChunkResult]:
    """Recursively split text by trying separators from coarsest to finest.

    Strategy: Recursive Character Splitting (LangChain-style)
    ----------------------------------------------------------
    Attempts to split on separators in this priority order:
      1. "\\n\\n"  (paragraph breaks) — coarsest, most meaningful
      2. "\\n"     (line breaks)
      3. ". "      (sentence boundaries)
      4. " "       (word boundaries) — finest fallback

    For each candidate split, only proceeds to the next separator if
    the resulting pieces are still larger than `chunk_size`. This means
    natural structure (paragraphs, sentences) is preserved whenever possible,
    and character-level splitting only happens as a last resort.

    Pros:
      - Best general-purpose strategy for most document types.
      - Preserves semantic units (paragraphs, sentences) wherever possible.
      - Overlap is applied at the sentence/word level, not the character level.

    Cons:
      - Slightly more complex and slower than fixed-size.
      - Overlap logic is approximate at sentence boundaries.

    This replicates the logic of LangChain's RecursiveCharacterTextSplitter.

    Args:
        text:       Input document text.
        chunk_size: Target maximum characters per chunk.
        overlap:    Characters of overlap between consecutive chunks.
        source:     Source document identifier.

    Returns:
        List of ChunkResult with character offsets computed post-split.
    """
    if not text.strip():
        return []

    def _split_recursive(t: str, separators: list[str]) -> list[str]:
        """Try each separator; recurse if any piece is still oversized."""
        if not separators or len(t) <= chunk_size:
            return [t]

        sep = separators[0]
        remaining = separators[1:]
        pieces = t.split(sep)

        # Re-join small consecutive pieces to fill up to chunk_size
        merged: list[str] = []
        current = ""
        for piece in pieces:
            candidate = (current + sep + piece) if current else piece
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                # If piece itself is oversized, recurse on it
                if len(piece) > chunk_size:
                    merged.extend(_split_recursive(piece, remaining))
                    current = ""
                else:
                    current = piece
        if current:
            merged.append(current)

        return merged

    separators = ["\n\n", "\n", ". ", " "]
    raw_chunks = _split_recursive(text, separators)
    raw_chunks = [c.strip() for c in raw_chunks if c.strip()]

    # Add overlap: prepend tail of previous chunk
    final_texts: list[str] = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0 or overlap == 0:
            final_texts.append(chunk)
        else:
            prev = final_texts[-1]
            tail = prev[-overlap:] if len(prev) > overlap else prev
            final_texts.append(tail + " " + chunk)

    # Compute approximate character offsets
    chunks: list[ChunkResult] = []
    search_from = 0
    for idx, chunk_text in enumerate(final_texts):
        # Find start in original text (best-effort for overlapping chunks)
        core = raw_chunks[idx] if idx < len(raw_chunks) else chunk_text
        pos = text.find(core[:min(40, len(core))], max(0, search_from - overlap))
        start = pos if pos >= 0 else search_from
        end = min(start + len(chunk_text), len(text))

        chunks.append(ChunkResult(
            text=chunk_text,
            chunk_id=f"{source}_recursive_{idx:04d}",
            start_char=start,
            end_char=end,
            word_count=len(chunk_text.split()),
            metadata={
                "strategy": "recursive",
                "source": source,
                "chunk_index": idx,
                "start_char": start,
                "end_char": end,
            },
        ))
        search_from = end

    return chunks


def character_chunk(
    text:      str,
    separator: str = "\n\n",
    source:    str = "doc",
) -> list[ChunkResult]:
    """Split text on a single separator (paragraph breaks by default).

    Strategy: Single-Separator (Character Split)
    ---------------------------------------------
    Splits on exactly one separator string, producing chunks that are
    complete paragraphs (or whatever structural unit the separator marks).

    Pros:
      - Conceptually simple — each chunk is one paragraph.
      - Excellent for structured documents: markdown, reports, FAQs,
        product catalogs where paragraphs are natural retrieval units.
      - No size limit — chunk sizes vary based on paragraph length.

    Cons:
      - No overlap — answers at paragraph boundaries may be missed.
      - Very long paragraphs produce oversized chunks that waste LLM context.
      - Very short paragraphs (headers) produce noisy, near-empty chunks.

    When to use:
      - Markdown documents with clear section structure.
      - FAQ pages where each Q&A is a paragraph.
      - Product catalogs (each product description = one paragraph).

    Args:
        text:      Input document text.
        separator: The separator to split on. Default "\\n\\n" = paragraphs.
        source:    Source document identifier.

    Returns:
        List of ChunkResult objects.
    """
    if not text.strip():
        return []

    raw_pieces = text.split(separator)
    chunks: list[ChunkResult] = []
    current_pos = 0

    for idx, piece in enumerate(raw_pieces):
        piece_stripped = piece.strip()
        if not piece_stripped:
            current_pos += len(piece) + len(separator)
            continue

        # Find actual position in original text
        start = text.find(piece_stripped, current_pos)
        if start < 0:
            start = current_pos
        end = start + len(piece_stripped)

        chunks.append(ChunkResult(
            text=piece_stripped,
            chunk_id=f"{source}_char_{idx:04d}",
            start_char=start,
            end_char=end,
            word_count=len(piece_stripped.split()),
            metadata={
                "strategy": "character",
                "separator": repr(separator),
                "source": source,
                "chunk_index": idx,
                "start_char": start,
                "end_char": end,
            },
        ))
        current_pos = end

    return chunks


def semantic_chunk(
    text:                str,
    encoder:             Optional[Callable] = None,
    similarity_threshold: float = 0.7,
    source:              str = "doc",
) -> list[ChunkResult]:
    """Group sentences into semantically coherent chunks using embedding similarity.

    Strategy: Semantic Chunking
    ----------------------------
    1. Split text into individual sentences.
    2. Embed each sentence with a dense encoder.
    3. Compute cosine similarity between adjacent sentence pairs.
    4. When similarity drops below `similarity_threshold`, start a new chunk.

    This means all sentences in one chunk are semantically related.
    A topic change (low similarity) creates a boundary — even mid-paragraph.

    Pros:
      - Produces the most semantically cohesive chunks.
      - Boundaries correspond to genuine topic shifts, not arbitrary lengths.
      - Retrieval precision is highest because each chunk = one topic.

    Cons:
      - Slowest strategy — requires encoding every sentence.
      - Chunk sizes vary widely (1 sentence to many).
      - Requires a good embedding model to be effective.
      - threshold is sensitive: too high = many tiny chunks; too low = one big chunk.

    Fallback:
      If `encoder` is None (e.g., in tests without model downloads),
      this function falls back to recursive_chunk to avoid model downloads.

    Args:
        text:                 Input document text.
        encoder:              Callable(list[str]) -> np.ndarray of shape (n, d).
                              Pass None to use fallback (no model required).
        similarity_threshold: Cosine similarity below which a new chunk starts.
                              Typical range: 0.6 – 0.8.
        source:               Source document identifier.

    Returns:
        List of ChunkResult objects (semantically grouped sentences).
    """
    import numpy as np

    if not text.strip():
        return []

    # Fallback: no encoder provided (test environments, no model downloads)
    if encoder is None:
        fallback = recursive_chunk(text, chunk_size=400, overlap=80, source=source)
        for c in fallback:
            c.metadata["strategy"] = "semantic_fallback"
        return fallback

    # Step 1: Split into sentences
    sentence_pattern = re.compile(r'(?<=[.!?])\s+|\n+')
    raw_sentences = sentence_pattern.split(text)
    sentences = [s.strip() for s in raw_sentences if s.strip() and len(s.split()) >= 3]

    if not sentences:
        return recursive_chunk(text, chunk_size=400, overlap=80, source=source)

    # Step 2: Embed all sentences
    embeddings = encoder(sentences)  # shape: (n, d)

    def cosine_sim(a, b) -> float:
        a_norm = a / (np.linalg.norm(a) + 1e-9)
        b_norm = b / (np.linalg.norm(b) + 1e-9)
        return float(np.dot(a_norm, b_norm))

    # Step 3: Group by similarity boundary
    groups: list[list[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i - 1], embeddings[i])
        if sim >= similarity_threshold:
            groups[-1].append(sentences[i])
        else:
            groups.append([sentences[i]])

    # Step 4: Build ChunkResult objects
    chunks: list[ChunkResult] = []
    search_from = 0
    for idx, group in enumerate(groups):
        chunk_text = " ".join(group)
        first_sent = group[0][:40] if group[0] else ""
        pos = text.find(first_sent, max(0, search_from - 10))
        start = pos if pos >= 0 else search_from
        end = min(start + len(chunk_text), len(text))

        chunks.append(ChunkResult(
            text=chunk_text,
            chunk_id=f"{source}_semantic_{idx:04d}",
            start_char=start,
            end_char=end,
            word_count=len(chunk_text.split()),
            metadata={
                "strategy": "semantic",
                "source": source,
                "chunk_index": idx,
                "num_sentences": len(group),
                "start_char": start,
                "end_char": end,
            },
        ))
        search_from = end

    return chunks


# ══════════════════════════════════════════════════════
# METADATA EXTRACTION
# ══════════════════════════════════════════════════════

def extract_metadata_from_chunk(
    text:        str,
    source_file: str = "",
    chunk_index: int = 0,
) -> dict:
    """Extract structured metadata from chunk text for metadata filtering.

    Extracts semantic signals that can be used as ChromaDB/vector store filters:
      - word_count, char_count: chunk size signals
      - has_price: True if chunk mentions ₹ or $ prices
      - has_product_id: True if chunk has pattern P\\d{3} (e.g. P001)
      - dominant_category: inferred product category from keywords
      - source, chunk_index: provenance tracking

    This metadata is stored alongside embeddings and enables pre-retrieval
    filtering: e.g., "only search in laptop documents" before vector search.

    Args:
        text:        The chunk text to analyse.
        source_file: The source filename or document ID.
        chunk_index: Position of this chunk in its source document.

    Returns:
        Dictionary of metadata fields suitable for vector store storage.
    """
    words = text.split()

    # Price detection: ₹ or $ followed by digits (with commas/dots)
    has_price = bool(re.search(r'[₹$]\s*[\d,]+', text))

    # Product ID detection: P followed by 3+ digits
    has_product_id = bool(re.search(r'\bP\d{3,}\b', text))

    # Category detection: scan for category keywords
    text_lower = text.lower()
    category_keywords = {
        "smartphone": ["smartphone", "phone", "mobile", "android", "ios", "iphone",
                       "samsung", "pixel", "oneplus", "snapdragon", "5g", "amoled"],
        "laptop": ["laptop", "macbook", "notebook", "thinkpad", "xps", "asus",
                   "dell", "lenovo", "processor", "cpu", "ssd", "ram"],
        "headphones": ["headphone", "earphone", "earbuds", "airpods", "sony",
                       "bose", "noise cancell", "anc", "bluetooth audio", "wh-"],
        "tablet": ["tablet", "ipad", "galaxy tab", "surface", "stylus", "pencil"],
        "accessories": ["mouse", "keyboard", "powerbank", "charger", "cable",
                        "hub", "dock", "logitech", "anker", "accessory"],
    }

    scores: dict[str, int] = {cat: 0 for cat in category_keywords}
    for cat, kws in category_keywords.items():
        for kw in kws:
            if kw in text_lower:
                scores[cat] += 1

    dominant_category = max(scores, key=lambda c: scores[c])
    if scores[dominant_category] == 0:
        dominant_category = "general"

    return {
        "word_count":        len(words),
        "char_count":        len(text),
        "has_price":         has_price,
        "has_product_id":    has_product_id,
        "dominant_category": dominant_category,
        "source":            source_file,
        "chunk_index":       chunk_index,
    }


# ══════════════════════════════════════════════════════
# STRATEGY COMPARISON
# ══════════════════════════════════════════════════════

def compare_strategies(text: str) -> dict:
    """Run all four chunking strategies on the same text and compare results.

    Returns a dictionary keyed by strategy name. Each value contains:
      - num_chunks : total chunks produced
      - avg_words  : mean word count across chunks
      - min_words  : minimum word count
      - max_words  : maximum word count
      - sample_chunk: first 120 characters of the first chunk

    Use this to quickly see how different strategies carve up your document
    before committing to one for production.

    Args:
        text: Any document text to compare strategies on.

    Returns:
        Dict[str, dict] with keys: fixed, recursive, character, semantic
    """
    strategies = {
        "fixed":     fixed_size_chunk(text, chunk_size=400, overlap=80, source="cmp"),
        "recursive": recursive_chunk(text, chunk_size=400, overlap=80, source="cmp"),
        "character": character_chunk(text, separator="\n\n", source="cmp"),
        "semantic":  semantic_chunk(text, encoder=None, source="cmp"),  # fallback in tests
    }

    results = {}
    for name, chunks in strategies.items():
        if not chunks:
            results[name] = {
                "num_chunks": 0, "avg_words": 0,
                "min_words": 0, "max_words": 0, "sample_chunk": "",
            }
            continue

        word_counts = [c.word_count for c in chunks]
        results[name] = {
            "num_chunks":   len(chunks),
            "avg_words":    round(sum(word_counts) / len(word_counts), 1),
            "min_words":    min(word_counts),
            "max_words":    max(word_counts),
            "sample_chunk": chunks[0].text[:120],
        }

    return results


# ══════════════════════════════════════════════════════
# DEMO TEXT
# ══════════════════════════════════════════════════════

SAMPLE_TEXT = """
MacBook Pro 14 M3 Pro — Product Overview

The MacBook Pro 14-inch with M3 Pro chip is designed for professionals who demand peak performance.
It features 18GB unified memory and up to 1TB SSD storage, delivering blazing-fast workflows for
machine learning, video editing, and software development.

Key Specifications:
- Processor: Apple M3 Pro (12-core CPU, 18-core GPU)
- Memory: 18GB unified memory
- Storage: 512GB / 1TB / 2TB SSD
- Display: 14.2-inch Liquid Retina XDR, 120Hz ProMotion
- Battery: Up to 18 hours
- Price: ₹1,99,900

Performance Highlights:
The M3 Pro chip delivers up to 40% faster CPU performance than M1 Pro.
Neural Engine handles machine learning workloads at 18 TOPS.
Hardware-accelerated ray tracing for creative professionals.
Thunderbolt 4 ports support dual external displays.

Ideal Use Cases:
Data scientists running large Jupyter notebooks benefit from the 18GB unified memory.
Video editors appreciate the 120Hz ProMotion display and H.264/HEVC hardware encoding.
Software developers enjoy native compilation speeds that rival desktop workstations.

The MacBook Pro 14 M3 Pro is priced at ₹1,99,900 and is available at all Apple Authorised Resellers.
Product ID: P001. Category: Laptop. In stock.

Samsung Galaxy S24 Ultra — Product Overview

The Galaxy S24 Ultra is Samsung's flagship smartphone for 2024, featuring a 200MP main camera
and the Snapdragon 8 Gen 3 processor. The titanium frame and built-in S Pen make it unique
among Android flagships.

Key Specifications:
- Processor: Snapdragon 8 Gen 3 for Galaxy
- Display: 6.8-inch Dynamic AMOLED 2X, 120Hz
- Camera: 200MP main + 12MP ultrawide + 50MP 5x telephoto + 10MP 3x telephoto
- Battery: 5000mAh with 45W fast charging
- RAM: 12GB / Storage: 256GB, 512GB, 1TB
- Price: ₹1,29,999

The S Pen is now integrated directly into the phone body, enabling precise note-taking and
creative work. The AI-powered photo editing can remove objects, reshape subjects, and generate
fill content using Samsung's Galaxy AI platform.

Product ID: P002. Category: Smartphone. In stock.
"""


# ══════════════════════════════════════════════════════
# MAIN DEMO
# ══════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("DOCUMENT CHUNKING STRATEGIES DEMO")
    print("=" * 70)

    print("\n[1] Fixed-Size Chunking (400 chars, 80 overlap)")
    fixed = fixed_size_chunk(SAMPLE_TEXT, chunk_size=400, overlap=80, source="demo")
    for i, c in enumerate(fixed[:3]):
        print(f"  Chunk {i}: {c.word_count} words | chars [{c.start_char}:{c.end_char}]")
        print(f"    '{c.text[:80]}...'")

    print(f"\n  Total: {len(fixed)} chunks")

    print("\n[2] Recursive Chunking (400 chars, 80 overlap)")
    recursive = recursive_chunk(SAMPLE_TEXT, chunk_size=400, overlap=80, source="demo")
    for i, c in enumerate(recursive[:3]):
        print(f"  Chunk {i}: {c.word_count} words | chars [{c.start_char}:{c.end_char}]")
        print(f"    '{c.text[:80]}...'")
    print(f"\n  Total: {len(recursive)} chunks")

    print("\n[3] Character Chunking (paragraph separator)")
    char_chunks = character_chunk(SAMPLE_TEXT, separator="\n\n", source="demo")
    for i, c in enumerate(char_chunks[:3]):
        print(f"  Chunk {i}: {c.word_count} words")
        print(f"    '{c.text[:80]}...'")
    print(f"\n  Total: {len(char_chunks)} chunks")

    print("\n[4] Semantic Chunking (no encoder — fallback to recursive)")
    semantic = semantic_chunk(SAMPLE_TEXT, encoder=None, source="demo")
    print(f"  Total: {len(semantic)} chunks (fallback mode — pass encoder= for real semantic)")

    print("\n[5] Metadata Extraction")
    meta = extract_metadata_from_chunk(
        SAMPLE_TEXT[:500], source_file="product_catalog.txt", chunk_index=0
    )
    for k, v in meta.items():
        print(f"  {k:25s}: {v}")

    print("\n[6] Strategy Comparison Table")
    comparison = compare_strategies(SAMPLE_TEXT)
    print(f"  {'Strategy':<12} {'Chunks':>6} {'AvgWords':>9} {'Min':>5} {'Max':>5}")
    print(f"  {'-'*12} {'-'*6} {'-'*9} {'-'*5} {'-'*5}")
    for name, stats in comparison.items():
        print(f"  {name:<12} {stats['num_chunks']:>6} {stats['avg_words']:>9.1f} "
              f"{stats['min_words']:>5} {stats['max_words']:>5}")


if __name__ == "__main__":
    main()
