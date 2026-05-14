"""
rag_pipeline.py — Advanced RAG Pipeline
========================================
Full production-grade RAG: hybrid search + cross-encoder reranking
+ metadata filtering + semantic caching + LangSmith tracing.

Architecture:
  User Query
    ↓
  Semantic Cache check (skip pipeline if similar query answered before)
    ↓ (cache miss)
  Metadata Pre-filter
    ↓
  Hybrid Retrieval (BM25 + Dense + RRF)
    ↓
  Cross-Encoder Reranking
    ↓
  Prompt Construction
    ↓
  LLM Call (Claude / OpenAI / Mock)
    ↓
  Store in Semantic Cache
    ↓
  Log to LangSmith / Local JSON
    ↓
  Return AdvancedRAGResult

Run: python -m day3.rag_pipeline
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv

load_dotenv()


# ══════════════════════════════════════════════════════
# SEMANTIC CACHE
# ══════════════════════════════════════════════════════

@dataclass
class CachedQuery:
    """One entry in the semantic cache.

    Fields:
        query:      The original query string.
        query_vec:  Embedding of the query (for similarity lookup).
        answer:     The generated answer to return on cache hit.
        timestamp:  Unix timestamp when this was cached.
        hit_count:  Number of times this cache entry has been retrieved.
    """
    query:     str
    query_vec: list[float]
    answer:    str
    timestamp: float
    hit_count: int = 0


class SemanticCache:
    """In-memory semantic cache using embedding similarity for lookup.

    Traditional caches require exact key matches.
    Semantic caches match queries by MEANING — so "What is the MacBook price?"
    returns the cached answer for "How much does the MacBook cost?".

    Architecture:
      - Stores query embeddings alongside answers.
      - On new query: embed it, compute cosine similarity with all cached queries.
      - If max similarity > threshold → cache hit, return cached answer.
      - Else → cache miss, run pipeline, store new embedding+answer.

    Threshold guidance:
      0.95+ : only near-identical rephrasing matches (very conservative)
      0.90  : same question with minor wording differences (recommended)
      0.80  : semantically similar but potentially different questions (aggressive)

    Production note:
      Replace the in-memory list with a vector database (Chroma, Redis + vector)
      for large-scale caches with millions of queries.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        max_size:             int   = 100,
    ):
        """
        Args:
            similarity_threshold: Cosine similarity above which a cache hit is declared.
            max_size:             Maximum number of entries. LRU eviction at capacity.
        """
        self._threshold = similarity_threshold
        self._max_size  = max_size
        self._entries:  list[CachedQuery] = []
        self._hits   = 0
        self._misses = 0

    def _get_embedding(self, text: str) -> list[float]:
        """Embed text for cache key. Uses hash-based mock for reliability.

        In production: use the same embedding model as the retrieval pipeline.
        Consistency between query and cache embeddings is critical.
        """
        # Deterministic hash-based embedding — no model download required
        # Sufficient for teaching; replace with real embeddings in production
        import hashlib
        import numpy as np

        # Create a pseudo-embedding from multiple hash seeds
        digest = hashlib.sha256(text.encode()).hexdigest()
        seed = int(digest[:8], 16)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(384).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        return vec.tolist()

    def get(self, query: str) -> Optional[str]:
        """Check cache for a semantically similar query.

        Args:
            query: Incoming query to look up.

        Returns:
            Cached answer string if hit, None if miss.
        """
        import numpy as np

        if not self._entries:
            self._misses += 1
            return None

        query_vec = np.array(self._get_embedding(query), dtype=np.float32)

        best_sim = -1.0
        best_entry = None

        for entry in self._entries:
            cached_vec = np.array(entry.query_vec, dtype=np.float32)
            sim = float(np.dot(query_vec, cached_vec))
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_sim >= self._threshold and best_entry is not None:
            best_entry.hit_count += 1
            self._hits += 1
            return best_entry.answer

        self._misses += 1
        return None

    def put(self, query: str, answer: str) -> None:
        """Store a new query-answer pair in the cache.

        Evicts the oldest entry (LRU approximation) when at capacity.

        Args:
            query:  The query string to cache.
            answer: The generated answer to store.
        """
        # Evict oldest entry if at capacity
        if len(self._entries) >= self._max_size:
            # Sort by (hit_count ASC, timestamp ASC) — evict least-used, oldest
            self._entries.sort(key=lambda e: (e.hit_count, e.timestamp))
            self._entries.pop(0)

        entry = CachedQuery(
            query=query,
            query_vec=self._get_embedding(query),
            answer=answer,
            timestamp=time.time(),
        )
        self._entries.append(entry)

    def stats(self) -> dict:
        """Return cache performance statistics.

        Returns:
            Dict with hits, misses, size, hit_rate.
        """
        total = self._hits + self._misses
        return {
            "hits":     self._hits,
            "misses":   self._misses,
            "size":     len(self._entries),
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
        }


# ══════════════════════════════════════════════════════
# LANGSMITH TRACER
# ══════════════════════════════════════════════════════

class LangSmithTracer:
    """LangSmith tracing with local JSON fallback.

    LangSmith traces every LLM call with full input/output/latency data.
    Without LANGCHAIN_API_KEY, falls back to local JSON file logging.
    This lets you develop and review traces locally before connecting cloud.

    Production: connect LANGCHAIN_API_KEY and view traces at smith.langchain.com
    Classroom: traces written to traces/rag_traces.json for review.
    """

    def __init__(self, project_name: str = "day3-advanced-rag"):
        self.project_name = project_name
        self._enabled     = False
        self._client      = None
        self._local_log:  list[dict] = []

        api_key = os.getenv("LANGCHAIN_API_KEY")
        if api_key:
            try:
                from langsmith import Client
                self._client  = Client()
                self._enabled = True
                print(f"[LangSmith] Connected. Project: '{project_name}'")
            except ImportError:
                print("[LangSmith] langsmith package not installed — using local logging.")
        else:
            print("[LangSmith] No LANGCHAIN_API_KEY — using local JSON logging.")

    def log_rag_call(
        self,
        query:      str,
        context:    str,
        answer:     str,
        latency_ms: float,
        metadata:   Optional[dict] = None,
    ) -> str:
        """Log a single RAG call. Returns trace_id."""
        trace_id = str(uuid.uuid4())[:8]
        entry = {
            "trace_id":    trace_id,
            "timestamp":   datetime.now().isoformat(),
            "project":     self.project_name,
            "query":       query,
            "context_len": len(context),
            "answer_len":  len(answer),
            "latency_ms":  round(latency_ms, 1),
            "metadata":    metadata or {},
        }

        if self._enabled and self._client:
            try:
                self._client.create_run(
                    name     = "advanced_rag_pipeline",
                    run_type = "chain",
                    inputs   = {"query": query, "context": context[:500]},
                    outputs  = {"answer": answer},
                    extra    = metadata or {},
                )
            except Exception as e:
                print(f"[LangSmith] Log failed: {e} — falling back to local.")

        self._local_log.append(entry)
        return trace_id

    def save_traces(self, path: str = "traces/rag_traces.json") -> None:
        """Save local traces to disk.

        Args:
            path: File path to write JSON traces.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self._local_log, indent=2))
        print(f"[LangSmith] Saved {len(self._local_log)} traces → {path}")

    def get_traces(self) -> list[dict]:
        """Return all locally logged traces."""
        return self._local_log


# ══════════════════════════════════════════════════════
# RESULT DATA CLASS
# ══════════════════════════════════════════════════════

@dataclass
class AdvancedRAGResult:
    """Full result from the AdvancedRAGPipeline.

    Fields:
        query:              The original user question.
        answer:             LLM-generated answer.
        retrieved_chunks:   Documents from hybrid retrieval (before reranking).
        reranked_chunks:    Documents after cross-encoder reranking (final context).
        retrieval_method:   Which retrieval path was used ("hybrid", "dense_only", etc.).
        cache_hit:          True if this answer came from the semantic cache.
        latency_ms:         End-to-end pipeline latency in milliseconds.
        faithfulness_proxy: Proxy faithfulness score (answer supported by context).
        num_retrieved:      Number of documents from first-stage retrieval.
        num_reranked:       Number of documents after reranking.
    """
    query:              str
    answer:             str
    retrieved_chunks:   list[str]
    reranked_chunks:    list[str]
    retrieval_method:   str
    cache_hit:          bool
    latency_ms:         float
    faithfulness_proxy: float
    num_retrieved:      int
    num_reranked:       int


# ══════════════════════════════════════════════════════
# ADVANCED RAG PIPELINE
# ══════════════════════════════════════════════════════

class AdvancedRAGPipeline:
    """Production-grade RAG pipeline with all Day 3 components.

    Components:
      1. SemanticCache    — avoid redundant LLM calls
      2. BM25Index        — keyword retrieval
      3. DenseIndex       — semantic retrieval
      4. RRF fusion       — merge BM25 + dense results
      5. CrossEncoder     — rerank with higher accuracy
      6. Claude/mock LLM  — generate final answer
      7. LangSmithTracer  — observability
    """

    def __init__(
        self,
        use_hybrid:    bool               = True,
        use_reranker:  bool               = True,
        use_cache:     bool               = True,
        top_k:         int                = 10,
        rerank_top_k:  int                = 3,
        mock_embed_fn: Optional[Callable] = None,
    ):
        """
        Args:
            use_hybrid:    Enable hybrid BM25+dense retrieval (vs dense-only).
            use_reranker:  Enable cross-encoder reranking stage.
            use_cache:     Enable semantic cache.
            top_k:         Number of candidates from first-stage retrieval.
            rerank_top_k:  Number of final documents after reranking.
            mock_embed_fn: Pass mock embedding function to avoid model downloads.
        """
        from day3.hybrid_search import BM25Index, DenseIndex, reciprocal_rank_fusion
        from day3.reranking import CrossEncoderReranker
        from day3.chunking import extract_metadata_from_chunk

        self.use_hybrid   = use_hybrid
        self.use_reranker = use_reranker
        self.use_cache    = use_cache
        self.top_k        = top_k
        self.rerank_top_k = rerank_top_k

        self._bm25       = BM25Index()
        self._dense      = DenseIndex(mock_embed_fn=mock_embed_fn)
        self._reranker   = CrossEncoderReranker()
        self._cache      = SemanticCache() if use_cache else None
        self._tracer     = LangSmithTracer()

        self._documents:  list[dict] = []   # {"text": ..., "metadata": ...}
        self._all_texts:  list[str]  = []
        self._extract_meta = extract_metadata_from_chunk

        self._rrf = reciprocal_rank_fusion

    def index(self, documents: list[dict]) -> None:
        """Index documents: extract metadata, build BM25 + dense index.

        Args:
            documents: List of {"text": str, "metadata": dict} dicts.
                       metadata may be partial — missing fields are extracted
                       automatically from text.
        """
        self._documents = []
        for i, doc in enumerate(documents):
            text = doc.get("text", "")
            meta = doc.get("metadata", {})
            # Augment metadata with extracted signals
            extracted = self._extract_meta(text, source_file=meta.get("source", ""), chunk_index=i)
            merged_meta = {**extracted, **meta}  # user-supplied metadata takes precedence
            self._documents.append({"text": text, "metadata": merged_meta})

        self._all_texts = [d["text"] for d in self._documents]
        self._bm25.index(self._all_texts)
        self._dense.index(self._all_texts)
        print(f"[AdvancedRAG] Indexed {len(self._documents)} documents.")

    def query(
        self,
        question:         str,
        metadata_filters: dict = {},
    ) -> AdvancedRAGResult:
        """Run the full advanced RAG pipeline.

        Pipeline:
          1. Check semantic cache.
          2. Apply metadata pre-filter.
          3. Hybrid retrieval (BM25 + dense + RRF) or dense-only.
          4. Cross-encoder reranking.
          5. Build prompt.
          6. Call LLM (Claude or mock).
          7. Store result in cache.
          8. Log trace.
          9. Return AdvancedRAGResult.

        Args:
            question:         User's question.
            metadata_filters: Pre-filter documents by metadata before retrieval.

        Returns:
            AdvancedRAGResult with answer and all pipeline metadata.
        """
        from day3.reranking import metadata_filter

        t0 = time.perf_counter()

        # Step 1: Semantic cache check
        if self._cache is not None:
            cached_answer = self._cache.get(question)
            if cached_answer is not None:
                latency = (time.perf_counter() - t0) * 1000
                return AdvancedRAGResult(
                    query=question,
                    answer=cached_answer,
                    retrieved_chunks=[],
                    reranked_chunks=[],
                    retrieval_method="cache_hit",
                    cache_hit=True,
                    latency_ms=round(latency, 2),
                    faithfulness_proxy=1.0,  # cached = previously verified
                    num_retrieved=0,
                    num_reranked=0,
                )

        # Step 2: Metadata pre-filter
        if metadata_filters:
            filtered_docs = metadata_filter(self._documents, metadata_filters)
            filtered_texts = [d["text"] for d in filtered_docs]
        else:
            filtered_docs = self._documents
            filtered_texts = self._all_texts

        if not filtered_texts:
            return self._empty_result(question, t0)

        # Step 3: Retrieval
        if self.use_hybrid:
            from day3.hybrid_search import BM25Index, DenseIndex

            # If we filtered, build sub-indices on filtered set
            if metadata_filters and len(filtered_texts) != len(self._all_texts):
                sub_bm25 = BM25Index()
                sub_bm25.index(filtered_texts)
                from day3.hybrid_search import DenseIndex
                sub_dense = DenseIndex(mock_embed_fn=self._dense._mock_embed_fn)
                sub_dense.index(filtered_texts)

                bm25_results  = sub_bm25.search(question, n=self.top_k)
                dense_results = sub_dense.search(question, n=self.top_k)
            else:
                bm25_results  = self._bm25.search(question, n=self.top_k)
                dense_results = self._dense.search(question, n=self.top_k)

            fused = self._rrf([bm25_results, dense_results])
            retrieved_texts = [r.text for r in fused[:self.top_k]]
            method = "hybrid_rrf"
        else:
            dense_results = self._dense.search(question, n=self.top_k)
            retrieved_texts = [r.text for r in dense_results]
            method = "dense_only"

        # Step 4: Cross-encoder reranking
        if self.use_reranker and retrieved_texts:
            reranked = self._reranker.rerank_mock(
                question, retrieved_texts, top_k=self.rerank_top_k
            )
            final_context_chunks = [r.text for r in reranked]
        else:
            final_context_chunks = retrieved_texts[:self.rerank_top_k]

        # Step 5: Build prompt
        prompt = self._build_prompt(question, final_context_chunks)

        # Step 6: LLM call
        answer = self._call_llm(prompt)

        latency = (time.perf_counter() - t0) * 1000

        # Step 7: Faithfulness proxy
        faith = self._faithfulness_proxy(answer, final_context_chunks)

        # Step 8: Cache the result
        if self._cache is not None:
            self._cache.put(question, answer)

        # Step 9: Trace
        context_str = "\n\n".join(final_context_chunks)
        self._tracer.log_rag_call(
            query=question,
            context=context_str,
            answer=answer,
            latency_ms=latency,
            metadata={
                "method":         method,
                "num_retrieved":  len(retrieved_texts),
                "num_reranked":   len(final_context_chunks),
                "filters":        str(metadata_filters),
            },
        )

        return AdvancedRAGResult(
            query=question,
            answer=answer,
            retrieved_chunks=retrieved_texts,
            reranked_chunks=final_context_chunks,
            retrieval_method=method,
            cache_hit=False,
            latency_ms=round(latency, 2),
            faithfulness_proxy=faith,
            num_retrieved=len(retrieved_texts),
            num_reranked=len(final_context_chunks),
        )

    def _build_prompt(self, question: str, context_chunks: list[str]) -> str:
        """Build a structured prompt with retrieved context.

        Args:
            question:       User's question.
            context_chunks: Retrieved and reranked document chunks.

        Returns:
            Complete prompt string ready for LLM.
        """
        context_str = "\n\n".join(
            f"[Document {i+1}]\n{chunk}"
            for i, chunk in enumerate(context_chunks)
        )
        return (
            "You are a helpful product expert. Answer the question using ONLY "
            "the provided context. If the context does not contain the answer, "
            "say 'I don't have enough information to answer this question.'\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

    def _call_llm(self, prompt: str) -> str:
        """Call Anthropic Claude API with fallback to mock response.

        Tries Claude claude-3-haiku-20240307 (fast, cheap) first.
        Falls back to mock_claude_response if no API key or on error.

        Args:
            prompt: Complete prompt string.

        Returns:
            Generated answer string.
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                message = client.messages.create(
                    model="claude-haiku-4-5",
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                return message.content[0].text
            except Exception as e:
                print(f"[LLM] Anthropic API error: {e} — using mock response.")

        # Fallback: extract context and generate mock answer
        return mock_claude_response(prompt, "")

    def _faithfulness_proxy(self, answer: str, context_chunks: list[str]) -> float:
        """Estimate faithfulness: fraction of answer words in context.

        Real faithfulness requires an LLM judge. This proxy is sufficient
        for classroom demonstration of the concept.

        Args:
            answer:         Generated answer text.
            context_chunks: Context chunks used to generate the answer.

        Returns:
            Proxy faithfulness score between 0.0 and 1.0.
        """
        if not context_chunks or not answer:
            return 0.0
        context = " ".join(context_chunks).lower()
        answer_words = set(answer.lower().split())
        context_words = set(context.split())
        if not answer_words:
            return 0.0
        overlap = len(answer_words & context_words) / len(answer_words)
        return round(min(1.0, overlap * 1.5), 4)  # scale up slightly (common words boost overlap)

    def _empty_result(self, question: str, t0: float) -> AdvancedRAGResult:
        """Return an empty result when no documents pass the filter."""
        latency = (time.perf_counter() - t0) * 1000
        return AdvancedRAGResult(
            query=question,
            answer="No documents matched the given filters.",
            retrieved_chunks=[],
            reranked_chunks=[],
            retrieval_method="filtered_empty",
            cache_hit=False,
            latency_ms=round(latency, 2),
            faithfulness_proxy=0.0,
            num_retrieved=0,
            num_reranked=0,
        )

    @property
    def cache_stats(self) -> dict:
        """Return semantic cache statistics."""
        return self._cache.stats() if self._cache else {"enabled": False}


# ══════════════════════════════════════════════════════
# MOCK LLM
# ══════════════════════════════════════════════════════

def mock_claude_response(prompt: str, context: str) -> str:
    """Mock LLM response for tests and environments without an API key.

    Extracts the first meaningful sentence from the prompt's context section
    and wraps it in a plausible answer format.

    Args:
        prompt:  The full prompt passed to the LLM.
        context: Additional context string (may be empty if context is in prompt).

    Returns:
        A plausible-looking templated answer string.
    """
    # Extract context from prompt if it's embedded
    ctx = context
    if not ctx and "Context:" in prompt:
        try:
            ctx_start = prompt.index("Context:") + len("Context:")
            ctx_end   = prompt.index("Question:") if "Question:" in prompt else len(prompt)
            ctx = prompt[ctx_start:ctx_end].strip()
        except ValueError:
            ctx = ""

    if ctx:
        # Take first sentence of first document block
        first_line = ctx.replace("[Document 1]", "").strip()
        first_sentence = first_line.split(".")[0] + "." if "." in first_line else first_line[:100]
        return (
            f"Based on the retrieved product information: {first_sentence} "
            "Please refer to the full product specifications for complete details."
        )

    return "Based on available information, I cannot provide a specific answer without relevant product context."


# ══════════════════════════════════════════════════════
# CONVENIENCE FACTORY
# ══════════════════════════════════════════════════════

def build_product_rag(mock_embed_fn: Optional[Callable] = None) -> AdvancedRAGPipeline:
    """Build and return an AdvancedRAGPipeline pre-loaded with 20 product docs.

    Args:
        mock_embed_fn: Pass for testing without model downloads.

    Returns:
        Ready-to-query AdvancedRAGPipeline.
    """
    from day3.hybrid_search import generate_product_corpus

    pipeline = AdvancedRAGPipeline(
        use_hybrid=True,
        use_reranker=True,
        use_cache=True,
        top_k=10,
        rerank_top_k=3,
        mock_embed_fn=mock_embed_fn,
    )

    corpus = generate_product_corpus()
    documents = [{"text": text, "metadata": {"source": "product_catalog"}} for text in corpus]
    pipeline.index(documents)
    return pipeline


# ══════════════════════════════════════════════════════
# MAIN DEMO
# ══════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("ADVANCED RAG PIPELINE DEMO")
    print("=" * 70)

    def _demo_embed(texts):
        import numpy as np
        vecs = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**31))
            v = rng.standard_normal(384).astype(np.float32)
            v = v / np.linalg.norm(v)
            vecs.append(v)
        return np.array(vecs)

    print("\n[1] Building product RAG pipeline...")
    pipeline = build_product_rag(mock_embed_fn=_demo_embed)

    queries = [
        ("What is the best laptop for machine learning?", {}),
        ("Which headphones have the best noise cancellation?", {}),
        ("Tell me about MacBook Pro price and specs.", {}),
    ]

    print("\n[2] Running queries...")
    for question, filters in queries:
        result = pipeline.query(question, metadata_filters=filters)
        print(f"\n  Q: {question}")
        print(f"  A: {result.answer[:120]}")
        print(f"  Method: {result.retrieval_method} | "
              f"Retrieved: {result.num_retrieved} → Reranked: {result.num_reranked} | "
              f"Latency: {result.latency_ms:.1f}ms | "
              f"Cache: {'HIT' if result.cache_hit else 'MISS'} | "
              f"Faithfulness: {result.faithfulness_proxy:.3f}")

    print("\n[3] Cache hit demo (same question again)...")
    repeat_result = pipeline.query(queries[0][0])
    print(f"  Cache HIT: {repeat_result.cache_hit} | Latency: {repeat_result.latency_ms:.1f}ms")

    print("\n[4] Cache stats:")
    stats = pipeline.cache_stats
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n[5] Saving traces...")
    pipeline._tracer.save_traces("traces/demo_traces.json")


if __name__ == "__main__":
    main()
