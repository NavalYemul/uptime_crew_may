"""
run_all.py — Day 2 instructor demo script
Usage: python run_all.py
"""
import sys
sys.path.insert(0, 'src')

print("=" * 65)
print("  DAY 2 — Embeddings, Vector Search, RAG & Data Pipelines")
print("=" * 65)

print("\n▶  MODULE 1: NumPy & Pandas")
print("─" * 65)
from day2.numpy_pandas import (
    array_basics, broadcasting_demo, create_ai_dataset,
    groupby_operations, file_format_comparison
)
r = array_basics()
print(f"Array sum={r['sum']}, dot={r['dot_ab']}")
r = broadcasting_demo()
print(f"Broadcasting diagonal_all_ones={r['diagonal_all_ones']}")
df = create_ai_dataset()
print(f"Dataset: {df.shape}")
print(groupby_operations(df).head(3).to_string(index=False))
r = file_format_comparison(df)
print(f"Parquet vs CSV size: {r['parquet_vs_csv_ratio']:.2f}x")

print("\n▶  MODULE 2: Embedding Models")
print("─" * 65)
from day2.embeddings import MODEL_CATALOGUE, model_selection_guide, dense_vs_sparse_comparison
print(f"Catalogue: {len(MODEL_CATALOGUE)} models")
m = model_selection_guide(speed_priority=True)
print(f"Speed model: {m.name} ({m.dims} dims)")
corpus = ["Machine learning learns from data.", "Dogs run fast.", "Canines sprint quickly."]
r = dense_vs_sparse_comparison(corpus)
print(f"Sparse sparsity: {r['sparse_sparsity']:.1%}")

print("\n▶  MODULE 3: Vector Store & HNSW")
print("─" * 65)
from day2.vector_store import distance_metrics_demo, generate_corpus, ChromaVectorStore, validate_index
r = distance_metrics_demo()
print(f"Cosine(same dir): {r['pair_a_d']['cosine']:.4f}")
corpus_docs = generate_corpus(30)
store = ChromaVectorStore("run_all_demo")
store.add_documents([d['text'] for d in corpus_docs])
vr = validate_index(store, 30)
print(f"Index count_ok={vr['count_ok']}, retrieval_ok={vr['retrieval_ok']}")

print("\n▶  MODULE 4: RAG Pipeline + Tracing + RAGAS")
print("─" * 65)
from day2.rag import RAGPipeline, LangSmithTracer, ragas_proxy_metrics, SAMPLE_DOCUMENTS, EVAL_QUESTIONS
tracer   = LangSmithTracer("run_all_demo")
pipeline = RAGPipeline(top_k=3, tracer=tracer)
pipeline.index(SAMPLE_DOCUMENTS)
result  = pipeline.query(EVAL_QUESTIONS[0][0])
metrics = ragas_proxy_metrics(result)
print(f"Query: {EVAL_QUESTIONS[0][0][:50]}")
print(f"Faithfulness={metrics.faithfulness:.4f} | Relevancy={metrics.answer_relevancy:.4f}")

print("\n▶  MODULE 5: Data Pipelines + DAG + Quality")
print("─" * 65)
from day2.pipeline import extract_from_source, etl_transform, elt_load_raw, build_ingestion_dag
records = extract_from_source(30)
df_etl  = etl_transform(records)
bronze  = elt_load_raw(records)
dag     = build_ingestion_dag(records)
outputs = dag.run()
print(f"ETL rows: {len(df_etl)}, Bronze: {len(bronze)}")
print(f"DAG report: {outputs.get('generate_report')}")

from day2.data_quality import run_validation_suite
reports = run_validation_suite(df_etl)
all_pass = all(r.passed for r in reports)
print(f"Validation suite: all_passed={all_pass}")

print("\n" + "=" * 65)
print("  All Day 2 demos complete!")
print("  Tests:    pytest tests/ -v")
print("  Notebook: jupyter lab notebooks/")
print("=" * 65)
