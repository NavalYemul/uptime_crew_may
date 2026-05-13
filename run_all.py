"""
run_all.py — Instructor demo script
=====================================
Runs all Day 1 modules in sequence.
Usage: python run_all.py
"""

import sys

print("=" * 65)
print("  DAY 1 — Python for AI Systems: End-to-End Demo")
print("=" * 65)

# 1. Models
print("\n\n▶  MODULE 1: Dataclasses & Pydantic")
print("─" * 65)
from day1.models import (
    Course, Coordinate, Dataset, DataPoint,
    MLExperiment, Student, TrainingRequest, DatasetStats,
)

s = Student("Alice", 25, "alice@ai.com", gpa=3.8)
print(f"Student: {s}, passing={s.is_passing()}")

c1, c2 = Coordinate(0, 0), Coordinate(3, 4)
print(f"Distance 3-4-5 triangle: {c1.distance_to(c2):.1f}")

course = Course("AI101", "Intro to AI", max_students=2)
course.enroll("Alice"); course.enroll("Bob")
print(f"Course seats left: {course.available_seats}")

exp = MLExperiment("run1", "RF", accuracy=0.93)
print(f"Experiment level: {exp.level.value}")

req = TrainingRequest(
    model_name="clf", algorithm="random_forest",
    features=["f1", "f2"], target="label"
)
print(f"Pydantic request: {req.model_name}, algo={req.algorithm}")


# 2. ML Workflow
print("\n\n▶  MODULE 2: ML Workflow")
print("─" * 65)
from day1.ml_workflow import run_classification_demo, run_regression_demo
lr_res, rf_res = run_classification_demo()
reg_res = run_regression_demo()


# 3. NLP Basics
print("\n\n▶  MODULE 3: NLP Basics")
print("─" * 65)
from day1.nlp_basics import (
    full_preprocess, bag_of_words, tfidf_vectors,
    word_frequency, CORPUS, SAMPLE_TEXT
)
tokens = full_preprocess(SAMPLE_TEXT)
print(f"Preprocessed tokens: {tokens[:10]}")
bow, vocab = bag_of_words(CORPUS)
print(f"BoW matrix: {bow.shape}  |  Vocab: {vocab[:8]}")
tfidf, _ = tfidf_vectors(CORPUS)
print(f"TF-IDF matrix: {tfidf.shape}")
freq = word_frequency(full_preprocess(" ".join(CORPUS)), top_n=5)
print(f"Top 5 words: {freq}")


# 4. Embeddings
print("\n\n▶  MODULE 4: Embeddings & Cosine Similarity")
print("─" * 65)
from day1.embeddings import (
    cosine_sim, pairwise_similarity, semantic_search,
    train_word2vec, TRAINING_SENTENCES, DOCUMENTS
)
print(f"Cosine [1,2,3] vs [1,2,3]: {cosine_sim([1,2,3],[1,2,3]):.4f}")
print(f"Cosine [1,0,0] vs [0,1,0]: {cosine_sim([1,0,0],[0,1,0]):.4f}")

query = "Python for machine learning"
print(f"\nSemantic search: '{query}'")
for doc, score in semantic_search(query, DOCUMENTS, top_n=3):
    print(f"  [{score:.3f}] {doc}")


# 5. Visualizations
print("\n\n▶  MODULE 5: Visualizations (5 charts)")
print("─" * 65)
print("Generating charts — close each window to continue...")
from day1.visualizations import run_all_visualizations
run_all_visualizations()


print("\n" + "=" * 65)
print("  All Day 1 demos complete!")
print("  Run tests with: pytest tests/ -v")
print("  Open notebook:  jupyter lab notebooks/sentence_embeddings.ipynb")
print("=" * 65)
