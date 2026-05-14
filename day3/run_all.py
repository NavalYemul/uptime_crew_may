"""
run_all.py — Day 3 instructor demo script
==========================================
Runs all six Day 3 modules in sequence with clear section headers.
Uses mock embeddings so no API keys or model downloads are required.

Usage: uv run python run_all.py
"""

import sys
import traceback
sys.path.insert(0, 'src')

import numpy as np


def demo_embed(texts):
    """Deterministic mock embeddings for standalone demo."""
    vecs = []
    for t in texts:
        rng = np.random.default_rng(abs(hash(t)) % (2**31))
        v = rng.standard_normal(384).astype(np.float32)
        v = v / np.linalg.norm(v)
        vecs.append(v)
    return np.array(vecs)


def section(title, n=1):
    print(f"\n{'═' * 70}")
    print(f"  MODULE {n}: {title}")
    print(f"{'═' * 70}")


def run_module(name, fn):
    try:
        fn()
        print(f"\n  [OK] {name} completed successfully.")
    except Exception as e:
        print(f"\n  [ERROR] {name} failed: {e}")
        traceback.print_exc()


# ══════════════════════════════════════════════════════
# MODULE 1: CHUNKING
# ══════════════════════════════════════════════════════

def run_chunking():
    section("Chunking Strategies", 1)
    from day3.chunking import (
        fixed_size_chunk, recursive_chunk, character_chunk,
        semantic_chunk, extract_metadata_from_chunk, compare_strategies
    )

    sample = (
        "MacBook Pro 14 M3 Pro features 18GB RAM for ML workloads.\n\n"
        "The Liquid Retina XDR display runs at 120Hz ProMotion.\n\n"
        "Price: ₹1,99,900. Product ID: P001. Category: Laptop.\n\n"
        "Samsung Galaxy S24 Ultra has 200MP camera and Snapdragon 8 Gen 3.\n\n"
        "It supports 45W fast charging and has 5000mAh battery."
    )

    print("\n[1] Chunking Strategy Comparison:")
    comparison = compare_strategies(sample)
    print(f"  {'Strategy':<12} {'Chunks':>6} {'AvgWords':>9} {'Min':>5} {'Max':>5}")
    print(f"  {'-'*12} {'-'*6} {'-'*9} {'-'*5} {'-'*5}")
    for name, stats in comparison.items():
        print(f"  {name:<12} {stats['num_chunks']:>6} {stats['avg_words']:>9.1f} "
              f"{stats['min_words']:>5} {stats['max_words']:>5}")

    print("\n[2] Metadata Extraction:")
    meta = extract_metadata_from_chunk(
        "MacBook Pro costs ₹1,99,900. Product ID: P001.",
        source_file="products.txt", chunk_index=0
    )
    print(f"  has_price:         {meta['has_price']}")
    print(f"  has_product_id:    {meta['has_product_id']}")
    print(f"  dominant_category: {meta['dominant_category']}")


# ══════════════════════════════════════════════════════
# MODULE 2: HYBRID SEARCH
# ══════════════════════════════════════════════════════

def run_hybrid_search():
    section("Hybrid Search (BM25 + Dense + RRF)", 2)
    from day3.hybrid_search import (
        BM25Index, DenseIndex, HybridSearcher,
        reciprocal_rank_fusion, SearchResult, generate_product_corpus
    )

    corpus = generate_product_corpus()
    print(f"\n[1] Corpus: {len(corpus)} product documents")

    print("\n[2] BM25 Search — 'AMOLED 120Hz' (exact technical terms):")
    bm25 = BM25Index()
    bm25.index(corpus)
    for r in bm25.search("AMOLED 120Hz display", n=2):
        print(f"  [{r.rank}] score={r.score:.3f} | {r.text[:80]}")

    print("\n[3] Dense Search — 'best laptop for ML' (semantic):")
    dense = DenseIndex(mock_embed_fn=demo_embed)
    dense.index(corpus)
    for r in dense.search("best laptop for machine learning", n=2):
        print(f"  [{r.rank}] sim={r.score:.4f} | {r.text[:80]}")

    print("\n[4] Hybrid (RRF) — 'lightweight device for travel':")
    hybrid = HybridSearcher(mock_embed_fn=demo_embed)
    hybrid.index(corpus)
    results = hybrid.search("lightweight device for travel", n=3)
    for r in results:
        print(f"  [{r.rank}] rrf={r.score:.6f} | {r.text[:80]}")

    print("\n[5] RRF Formula Demo:")
    list_a = [SearchResult("Doc A", 10.0, 1, "bm25"), SearchResult("Doc B", 8.0, 2, "bm25")]
    list_b = [SearchResult("Doc B", 0.9, 1, "dense"), SearchResult("Doc A", 0.8, 2, "dense")]
    fused = reciprocal_rank_fusion([list_a, list_b])
    for r in fused:
        print(f"  [{r.rank}] {r.text} | RRF={r.score:.6f}")


# ══════════════════════════════════════════════════════
# MODULE 3: RERANKING
# ══════════════════════════════════════════════════════

def run_reranking():
    section("Reranking, MMR & Metadata Filtering", 3)
    from day3.hybrid_search import generate_product_corpus, DenseIndex
    from day3.reranking import (
        CrossEncoderReranker, mmr_select, metadata_filter
    )

    corpus = generate_product_corpus()

    print("\n[1] Cross-Encoder Reranking (mock):")
    reranker = CrossEncoderReranker()
    candidates = corpus[:6]
    query = "noise cancelling wireless headphones"
    reranked = reranker.rerank_mock(query, candidates, top_k=3)
    for r in reranked:
        delta = r.initial_rank - r.final_rank
        symbol = ("↑" if delta > 0 else "↓" if delta < 0 else "=")
        print(f"  [{r.final_rank}] {symbol} (was #{r.initial_rank}) "
              f"score={r.final_score:.4f} | {r.text[:70]}")

    print("\n[2] MMR — balanced (lambda=0.5):")
    dense = DenseIndex(mock_embed_fn=demo_embed)
    dense.index(corpus)
    qv = dense._encode(["laptop for developers"])[0]
    dv = dense._embeddings
    for idx, score in mmr_select(qv, dv, corpus, top_k=3, lambda_param=0.5):
        print(f"  [{score:.4f}] {corpus[idx][:80]}")

    print("\n[3] Metadata Filtering:")
    docs_with_meta = [
        {"text": c, "metadata": {"category": "laptop" if "laptop" in c.lower() else "other"}}
        for c in corpus
    ]
    laptop_docs = metadata_filter(docs_with_meta, {"category": "laptop"})
    print(f"  Total: {len(corpus)} | Laptop only: {len(laptop_docs)}")


# ══════════════════════════════════════════════════════
# MODULE 4: ADVANCED RAG PIPELINE
# ══════════════════════════════════════════════════════

def run_rag_pipeline():
    section("Advanced RAG Pipeline (Hybrid + Rerank + Cache)", 4)
    from day3.rag_pipeline import build_product_rag

    print("\n[1] Building pipeline...")
    pipeline = build_product_rag(mock_embed_fn=demo_embed)

    queries = [
        "What is the best laptop for machine learning?",
        "Which headphones have the best noise cancellation?",
        "Tell me about the Samsung Galaxy S24 Ultra camera.",
    ]

    print("\n[2] Queries:")
    for q in queries:
        result = pipeline.query(q)
        print(f"\n  Q: {q}")
        print(f"  A: {result.answer[:110]}")
        print(f"  Method={result.retrieval_method} | "
              f"Retrieved={result.num_retrieved} → Reranked={result.num_reranked} | "
              f"Latency={result.latency_ms:.0f}ms | Cache={'HIT' if result.cache_hit else 'MISS'}")

    print("\n[3] Cache hit demo:")
    repeat = pipeline.query(queries[0])
    print(f"  Same question → Cache HIT: {repeat.cache_hit} | Latency: {repeat.latency_ms:.0f}ms")

    print("\n[4] Cache stats:")
    stats = pipeline.cache_stats
    print(f"  hits={stats['hits']} misses={stats['misses']} "
          f"size={stats['size']} hit_rate={stats['hit_rate']:.1%}")


# ══════════════════════════════════════════════════════
# MODULE 5: EVALUATION
# ══════════════════════════════════════════════════════

def run_evaluation():
    section("RAGAS Evaluation, Golden Dataset & A/B Test", 5)
    from day3.evaluation import GOLDEN_DATASET, RAGEvaluator, ab_test, compute_ragas_proxy
    from day3.rag_pipeline import build_product_rag

    print(f"\n[1] Golden dataset: {len(GOLDEN_DATASET)} questions across "
          f"{len(set(qa.category for qa in GOLDEN_DATASET))} categories")

    print("\n[2] RAGAS proxy metrics (single example):")
    result = compute_ragas_proxy(
        question="Which headphones have best ANC?",
        answer="Sony WH-1000XM5 has industry-leading noise cancellation.",
        context="Sony WH-1000XM5 with 8 microphones and dual processors delivers industry-leading ANC.",
        ground_truth="Sony WH-1000XM5 has industry-leading ANC with 8 microphones.",
    )
    print(f"  faithfulness={result.faithfulness:.4f} | "
          f"answer_relevancy={result.answer_relevancy:.4f} | "
          f"passed={result.passed}")

    print("\n[3] Full evaluation on golden dataset...")
    pipeline = build_product_rag(mock_embed_fn=demo_embed)
    evaluator = RAGEvaluator(pipeline)
    results = evaluator.run(GOLDEN_DATASET)
    summary = evaluator.summary(results)

    print(f"  {'Metric':<22} {'Mean':>6} {'Std':>6}")
    print(f"  {'-'*22} {'-'*6} {'-'*6}")
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        m = summary[metric]
        print(f"  {metric:<22} {m['mean']:>6.4f} {m['std']:>6.4f}")
    print(f"\n  Pass rate: {summary['pass_rate']:.1%} ({summary['num_passed']}/{summary['num_evaluated']})")

    print("\n[4] Saving baseline → day3_baseline.json")
    evaluator.save_baseline(results, "day3_baseline.json")


# ══════════════════════════════════════════════════════
# MODULE 6: AIRFLOW
# ══════════════════════════════════════════════════════

def run_airflow():
    section("Airflow DAG Patterns for RAG Pipelines", 6)
    from day3.airflow_demo import (
        SimpleDAGRunner, extract_products, chunk_and_embed,
        index_to_vectorstore, validate_index, notify_complete,
        create_rag_ingestion_dag_definition, show_airflow_code, try_real_airflow
    )
    from datetime import datetime

    execution_date = datetime.now().date().isoformat()

    print("\n[1] DAG Definition:")
    dag_def = create_rag_ingestion_dag_definition()
    print(f"  DAG ID:    {dag_def['dag_id']}")
    print(f"  Schedule:  {dag_def['schedule_interval']} ({dag_def['cron_expression']} IST)")
    print(f"  Tasks:     {[t['task_id'] for t in dag_def['tasks']]}")

    print("\n[2] SimpleDAGRunner simulation:")
    runner = SimpleDAGRunner("rag_product_ingestion")
    runner.add_task("extract",  extract_products,    depends_on=[])
    runner.add_task("chunk",    chunk_and_embed,      depends_on=["extract"])
    runner.add_task("index",    index_to_vectorstore, depends_on=["chunk"])
    runner.add_task("validate", validate_index,       depends_on=["index"])
    runner.add_task("notify",   notify_complete,      depends_on=["validate"])

    results = runner.run(execution_date)
    for r in results:
        icon = "✓" if r.status == "success" else "✗"
        print(f"  {icon} {r.task_id:<12} {r.status:<8} {r.duration_s:.3f}s")

    print("\n[3] Idempotency (re-run same date):")
    result2 = extract_products(execution_date=execution_date)
    print(f"  Re-extracted: {result2['count']} records (from idempotent cache)")

    print("\n[4] Airflow status:")
    airflow_status = try_real_airflow()
    if isinstance(airflow_status, dict):
        print(f"  Airflow installed: {airflow_status.get('dag_id')}")
    else:
        print("  Airflow not installed — code snippet available via show_airflow_code()")

    # Cleanup
    import shutil
    shutil.rmtree(f"/tmp/dag_outputs/{execution_date}", ignore_errors=True)


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║" + "  DAY 3 — ADVANCED RAG: ALL MODULES DEMO".center(68) + "║")
    print("║" + "  Hybrid Search | Reranking | Evaluation | Airflow".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    run_module("Module 1: Chunking",         run_chunking)
    run_module("Module 2: Hybrid Search",    run_hybrid_search)
    run_module("Module 3: Reranking",        run_reranking)
    run_module("Module 4: RAG Pipeline",     run_rag_pipeline)
    run_module("Module 5: Evaluation",       run_evaluation)
    run_module("Module 6: Airflow Demo",     run_airflow)

    print("\n" + "═" * 70)
    print("  All Day 3 modules completed.")
    print("  Run individual modules: uv run python -m day3.<module_name>")
    print("  Run tests:              uv run pytest tests/ -v")
    print("═" * 70)
