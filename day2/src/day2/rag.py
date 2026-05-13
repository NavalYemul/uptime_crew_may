"""
rag.py — RAG Pipeline, LangSmith Tracing & RAGAS Evaluation
=============================================================
Covers: RAG architecture, LLM observability, tracing every LLM call,
        RAGAS metrics (faithfulness, answer relevance, context precision/recall),
        baseline recording.

All core demos work WITHOUT API keys.
LangSmith tracing and RAGAS require optional keys (see .env.example).

Run:  python -m day2.rag
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()


# ══════════════════════════════════════════════════════
# RAG ARCHITECTURE
# ══════════════════════════════════════════════════════
"""
RAG = Retrieval Augmented Generation

Problem: LLMs have a knowledge cutoff and hallucinate facts.
Solution: Give the LLM relevant context RETRIEVED from your own documents.

Pipeline:
  INDEXING (offline, run once):
    Documents → Chunk → Embed → Store in VectorDB

  QUERYING (online, per request):
    User Question
      → Embed question
      → Retrieve top-k similar chunks from VectorDB
      → Build prompt: [System] + [Context chunks] + [Question]
      → Send to LLM
      → Return answer + source citations

  EVALUATION:
    Faithfulness       — is the answer grounded in the retrieved context?
    Answer Relevancy   — does the answer actually address the question?
    Context Precision  — are retrieved chunks relevant? (signal/noise ratio)
    Context Recall     — did we retrieve all the info needed to answer?
"""


# ══════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════

@dataclass
class Document:
    text:     str
    doc_id:   str
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    text:     str
    doc_id:   str
    score:    float
    rank:     int
    metadata: dict = field(default_factory=dict)


@dataclass
class RAGResult:
    query:            str
    answer:           str
    retrieved_chunks: list[RetrievedChunk]
    model_used:       str
    latency_ms:       float
    trace_id:         str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def context_text(self) -> str:
        return "\n\n".join(c.text for c in self.retrieved_chunks)

    @property
    def num_chunks(self) -> int:
        return len(self.retrieved_chunks)


@dataclass
class RAGEvalResult:
    """Stores RAGAS evaluation scores for one query."""
    query:              str
    answer:             str
    faithfulness:       float     # 0–1: is answer supported by context?
    answer_relevancy:   float     # 0–1: does answer address the question?
    context_precision:  float     # 0–1: are top chunks relevant?
    context_recall:     float     # 0–1: did we get all needed info?

    @property
    def average(self) -> float:
        return round(
            (self.faithfulness + self.answer_relevancy +
             self.context_precision + self.context_recall) / 4, 4
        )


# ══════════════════════════════════════════════════════
# CHUNKING
# ══════════════════════════════════════════════════════

def chunk_document(
    text: str,
    chunk_size: int = 200,
    overlap: int = 40,
) -> list[str]:
    """
    Split a document into overlapping chunks.

    Why overlap?
      An answer may span a chunk boundary. Overlapping ensures no sentence
      is 'cut off' and lost. Typical overlap: 10–20% of chunk_size.

    Why chunk at all?
      LLMs have context limits. Also, smaller chunks = more precise retrieval.
      Retrieving a 10-page document for a question about one sentence = noise.

    Production: use LangChain RecursiveCharacterTextSplitter which splits
    on paragraphs first, then sentences, then words — more semantically aware.
    """
    words  = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return [c for c in chunks if len(c.split()) >= 10]


def chunk_corpus(documents: list[Document], **kwargs) -> list[Document]:
    """Chunk all documents and return flat list of chunk documents."""
    chunks = []
    for doc in documents:
        for j, chunk_text in enumerate(chunk_document(doc.text, **kwargs)):
            chunks.append(Document(
                text     = chunk_text,
                doc_id   = f"{doc.doc_id}_chunk_{j:03d}",
                metadata = {**doc.metadata, "source_doc": doc.doc_id, "chunk": j},
            ))
    return chunks


# ══════════════════════════════════════════════════════
# MOCK LLM (works without API key)
# ══════════════════════════════════════════════════════

def mock_llm(prompt: str, context: str) -> str:
    """
    Mock LLM for classroom demo — returns a templated answer based on context.
    In production, replace with:
      - OpenAI: openai.chat.completions.create(...)
      - Azure OpenAI: same API via Azure endpoint
      - Databricks: requests to /serving-endpoints/{model}/invocations
      - Claude: anthropic.messages.create(...)
    """
    # Extract first sentence of context as the 'answer'
    first_sentence = context.split(".")[0] + "." if context else "No context available."
    return (
        f"Based on the provided context: {first_sentence} "
        f"This information comes from the retrieved documents."
    )


# ══════════════════════════════════════════════════════
# LANGSMITH TRACING
# ══════════════════════════════════════════════════════

class LangSmithTracer:
    """
    LangSmith traces every LLM call with:
      - Input prompt
      - Retrieved context
      - Model output
      - Latency, token counts
      - User feedback

    Without LANGCHAIN_API_KEY, falls back to local JSON logging.
    This lets you develop and review traces locally before connecting cloud.
    """

    def __init__(self, project_name: str = "day2-rag-baseline"):
        self.project_name = project_name
        self._enabled     = False
        self._client      = None
        self._local_log: list[dict] = []

        api_key = os.getenv("LANGCHAIN_API_KEY")
        if api_key:
            try:
                from langsmith import Client
                self._client  = Client()
                self._enabled = True
                print(f"[LangSmith] Connected. Project: '{project_name}'")
            except ImportError:
                print("[LangSmith] langsmith package not installed. Using local logging.")
        else:
            print("[LangSmith] No LANGCHAIN_API_KEY found. Using local JSON logging.")

    def log_rag_call(
        self,
        query:   str,
        context: str,
        answer:  str,
        latency: float,
        metadata: dict | None = None,
    ) -> str:
        """Log a single RAG call. Returns trace_id."""
        trace_id = str(uuid.uuid4())[:8]
        entry = {
            "trace_id":   trace_id,
            "timestamp":  datetime.now().isoformat(),
            "project":    self.project_name,
            "query":      query,
            "context_len": len(context),
            "answer_len":  len(answer),
            "latency_ms":  round(latency * 1000, 1),
            "metadata":    metadata or {},
        }

        if self._enabled and self._client:
            try:
                run = self._client.create_run(
                    name      = "rag_pipeline",
                    run_type  = "chain",
                    inputs    = {"query": query, "context": context},
                    outputs   = {"answer": answer},
                    extra     = metadata or {},
                )
                return str(run.id)
            except Exception as e:
                print(f"[LangSmith] Logging failed: {e}. Falling back to local.")

        self._local_log.append(entry)
        return trace_id

    def get_local_traces(self) -> list[dict]:
        return self._local_log

    def save_local_traces(self, path: str = "traces.json") -> None:
        Path(path).write_text(json.dumps(self._local_log, indent=2))
        print(f"[LangSmith] Saved {len(self._local_log)} traces → {path}")


# ══════════════════════════════════════════════════════
# RAG PIPELINE
# ══════════════════════════════════════════════════════

class RAGPipeline:
    """
    End-to-end RAG pipeline:
      index() → ingest documents into vector store
      query() → retrieve + generate + trace + return result
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k:           int = 3,
        llm_fn           = None,    # callable(prompt, context) -> str
        tracer:          LangSmithTracer | None = None,
    ):
        from day2.vector_store import ChromaVectorStore
        self.store   = ChromaVectorStore("rag_pipeline", embedding_model)
        self.top_k   = top_k
        self.llm     = llm_fn or mock_llm
        self.tracer  = tracer
        self._model  = embedding_model

    def index(self, documents: list[Document]) -> None:
        """Chunk all documents and load into vector store."""
        chunks = chunk_corpus(documents, chunk_size=80, overlap=15)
        texts  = [c.text     for c in chunks]
        ids    = [c.doc_id   for c in chunks]
        metas  = [c.metadata for c in chunks]
        self.store.add_documents(texts, ids=ids, metadatas=metas)
        print(f"[RAGPipeline] Indexed {len(documents)} docs → "
              f"{len(chunks)} chunks → {self.store.count()} in vector store")

    def query(self, question: str) -> RAGResult:
        """
        Retrieve relevant chunks, build prompt, call LLM, log trace.
        """
        t0 = time.perf_counter()

        # 1. Retrieve
        hits = self.store.search(question, n_results=self.top_k)
        chunks = [
            RetrievedChunk(
                text=h["document"], doc_id=h["id"],
                score=h["score"], rank=h["rank"],
                metadata=h.get("metadata", {}),
            )
            for h in hits
        ]

        # 2. Build context
        context = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(chunks))

        # 3. Generate
        answer = self.llm(question, context)

        latency = time.perf_counter() - t0

        # 4. Trace
        trace_id = "no_tracer"
        if self.tracer:
            trace_id = self.tracer.log_rag_call(
                query   = question,
                context = context,
                answer  = answer,
                latency = latency,
            )

        return RAGResult(
            query            = question,
            answer           = answer,
            retrieved_chunks = chunks,
            model_used       = self._model,
            latency_ms       = round(latency * 1000, 2),
            trace_id         = trace_id,
        )


# ══════════════════════════════════════════════════════
# RAGAS METRICS (proxy versions — no API key needed)
# ══════════════════════════════════════════════════════

def ragas_proxy_metrics(result: RAGResult, ground_truth: str = "") -> RAGEvalResult:
    """
    Proxy implementations of RAGAS metrics using only embeddings + heuristics.
    Real RAGAS calls an LLM judge (GPT-4) for more nuanced evaluation.
    These give the same conceptual signal at zero cost.

    Real RAGAS (requires OPENAI_API_KEY):
      from ragas import evaluate
      from ragas.metrics import faithfulness, answer_relevancy
      evaluate(dataset, metrics=[faithfulness, answer_relevancy, ...])
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    import numpy as np

    context   = result.context_text
    answer    = result.answer
    query     = result.query

    def tfidf_sim(a: str, b: str) -> float:
        if not a.strip() or not b.strip():
            return 0.0
        vec = TfidfVectorizer().fit([a, b])
        m   = vec.transform([a, b])
        return float(sk_cosine(m[0], m[1])[0, 0])

    def word_overlap(a: str, b: str) -> float:
        wa = set(a.lower().split())
        wb = set(b.lower().split())
        return len(wa & wb) / max(len(wb), 1)

    # Faithfulness proxy: what fraction of answer words appear in context?
    faithfulness = min(1.0, word_overlap(context, answer) * 2)

    # Answer relevancy proxy: TF-IDF similarity between query and answer
    answer_relevancy = tfidf_sim(query, answer)

    # Context precision proxy: average score of retrieved chunks
    if result.retrieved_chunks:
        scores = [c.score for c in result.retrieved_chunks]
        context_precision = float(np.mean(scores))
    else:
        context_precision = 0.0

    # Context recall proxy: TF-IDF overlap of context with ground truth
    if ground_truth:
        context_recall = tfidf_sim(context, ground_truth)
    else:
        context_recall = min(1.0, context_precision * 1.1)  # estimate

    return RAGEvalResult(
        query             = query,
        answer            = answer,
        faithfulness      = round(faithfulness,      4),
        answer_relevancy  = round(answer_relevancy,  4),
        context_precision = round(context_precision, 4),
        context_recall    = round(context_recall,    4),
    )


def real_ragas_evaluation(
    questions:      list[str],
    answers:        list[str],
    contexts_list:  list[list[str]],
    ground_truths:  list[str],
) -> dict | None:
    """
    Run real RAGAS evaluation using OpenAI as judge.
    Requires: pip install ragas openai && OPENAI_API_KEY in .env

    Returns None if RAGAS is not available/configured.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("[RAGAS] No OPENAI_API_KEY found. Returning None.")
        return None

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy, context_precision,
            context_recall, faithfulness,
        )

        dataset = Dataset.from_dict({
            "question":    questions,
            "answer":      answers,
            "contexts":    contexts_list,
            "ground_truth": ground_truths,
        })

        results = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        return results.to_pandas().mean().to_dict()

    except ImportError:
        print("[RAGAS] Install with: uv pip install 'day2-embeddings-rag[ragas]'")
        return None


# ══════════════════════════════════════════════════════
# BASELINE RECORDER
# ══════════════════════════════════════════════════════

def record_baseline(
    eval_results: list[RAGEvalResult],
    model_name:   str,
    output_path:  str = "rag_baseline.json",
) -> dict:
    """
    Save a reproducible baseline for this model + dataset combination.
    Run this once → commit the JSON → compare future models against it.

    This is your Day 2 baseline. Day 3 should beat it.
    """
    import numpy as np

    scores = {
        "faithfulness":      [r.faithfulness      for r in eval_results],
        "answer_relevancy":  [r.answer_relevancy   for r in eval_results],
        "context_precision": [r.context_precision  for r in eval_results],
        "context_recall":    [r.context_recall     for r in eval_results],
    }

    baseline = {
        "model":           model_name,
        "timestamp":       datetime.now().isoformat(),
        "num_queries":     len(eval_results),
        "metrics": {
            k: {
                "mean": round(float(np.mean(v)), 4),
                "std":  round(float(np.std(v)),  4),
                "min":  round(float(np.min(v)),  4),
                "max":  round(float(np.max(v)),  4),
            }
            for k, v in scores.items()
        },
        "overall_mean": round(float(np.mean([r.average for r in eval_results])), 4),
    }

    Path(output_path).write_text(json.dumps(baseline, indent=2))
    print(f"[Baseline] Saved → {output_path}")
    return baseline


# ══════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════

SAMPLE_DOCUMENTS = [
    Document("Machine learning uses algorithms to learn patterns from data without explicit programming. "
             "Supervised learning requires labelled examples. Unsupervised learning finds hidden structure. "
             "Reinforcement learning trains agents through rewards and penalties.",
             doc_id="ml_intro", metadata={"topic": "ml"}),

    Document("Neural networks are composed of layers of interconnected nodes. "
             "Deep learning uses many hidden layers to learn hierarchical representations. "
             "Convolutional networks excel at image recognition. Recurrent networks model sequences. "
             "Transformers use attention mechanisms and power modern LLMs.",
             doc_id="dl_intro", metadata={"topic": "dl"}),

    Document("RAG combines retrieval with language generation to reduce hallucinations. "
             "The pipeline chunks documents, embeds them, stores in a vector database, "
             "then retrieves relevant chunks at query time and provides them as context to the LLM. "
             "Key metrics: faithfulness, answer relevancy, context precision, context recall.",
             doc_id="rag_intro", metadata={"topic": "rag"}),

    Document("Data pipelines can be batch or streaming. Batch processes data in bulk at scheduled intervals. "
             "Streaming processes data continuously as it arrives. Apache Kafka is a popular streaming platform. "
             "Apache Spark supports both batch and streaming with a unified API.",
             doc_id="pipeline_intro", metadata={"topic": "data_eng"}),

    Document("Vector databases store high-dimensional embeddings and support fast nearest-neighbour search. "
             "HNSW (Hierarchical Navigable Small World) is the algorithm used by Chroma, Weaviate, and Qdrant. "
             "Cosine similarity measures the angle between vectors. Lower distance = higher similarity.",
             doc_id="vector_intro", metadata={"topic": "vectors"}),
]

EVAL_QUESTIONS = [
    ("What is supervised learning?",
     "Supervised learning requires labelled examples to train predictive models."),
    ("How does RAG reduce hallucinations?",
     "RAG provides retrieved document context to the LLM so answers are grounded in real data."),
    ("What algorithm do vector databases use for fast search?",
     "HNSW (Hierarchical Navigable Small World) enables fast approximate nearest-neighbour search."),
]


if __name__ == "__main__":
    import pprint

    print("=" * 60)
    print("RAG PIPELINE + LANGSMITH + RAGAS DEMO")
    print("=" * 60)

    # Setup
    tracer   = LangSmithTracer("day2-demo")
    pipeline = RAGPipeline(top_k=3, tracer=tracer)

    # Index
    print("\n[1] Indexing documents...")
    pipeline.index(SAMPLE_DOCUMENTS)

    # Query + evaluate
    print("\n[2] Querying + Evaluating...")
    eval_results: list[RAGEvalResult] = []

    for question, ground_truth in EVAL_QUESTIONS:
        result = pipeline.query(question)
        metrics = ragas_proxy_metrics(result, ground_truth)
        eval_results.append(metrics)

        print(f"\n  Q: {question}")
        print(f"  A: {result.answer[:80]}...")
        print(f"  Chunks retrieved: {result.num_chunks}")
        print(f"  Latency: {result.latency_ms:.1f}ms | Trace: {result.trace_id}")
        print(f"  Faithfulness={metrics.faithfulness:.4f} | "
              f"Relevancy={metrics.answer_relevancy:.4f} | "
              f"Precision={metrics.context_precision:.4f}")

    # Baseline
    print("\n[3] Recording baseline...")
    baseline = record_baseline(eval_results, "all-MiniLM-L6-v2")
    print(f"  Overall mean score: {baseline['overall_mean']:.4f}")
    pprint.pprint(baseline["metrics"])

    # Traces
    print(f"\n[4] Local traces: {len(tracer.get_local_traces())}")
    for t in tracer.get_local_traces():
        print(f"  [{t['trace_id']}] latency={t['latency_ms']}ms q='{t['query'][:50]}'")
