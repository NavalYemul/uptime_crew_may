"""
numpy_pandas.py — NumPy & Pandas Essentials for AI/Data Engineering
=====================================================================
Covers: array ops, broadcasting, DataFrame wrangling, groupby, merge,
        Parquet / CSV / JSON interchange formats.

Run:  python -m day2.numpy_pandas
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════
# PART 1 — NUMPY
# ══════════════════════════════════════════════════════

def array_basics() -> dict:
    """
    NumPy arrays vs Python lists:
    - Fixed dtype → less memory
    - Contiguous in RAM → fast vectorised ops (no Python loop overhead)
    - Foundation of every ML library: sklearn, PyTorch, TensorFlow all
      accept/return numpy arrays.
    """
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.arange(1, 6, dtype=float)

    return {
        "sum":      float(a.sum()),
        "mean":     float(a.mean()),
        "std":      float(a.std()),
        "dot_ab":   float(np.dot(a, b)),         # dot product
        "element_mul": (a * b).tolist(),          # element-wise multiply
        "reshaped": a.reshape(1, 5).shape,        # matrix shape
    }


def broadcasting_demo() -> dict:
    """
    Broadcasting: NumPy automatically expands smaller arrays to match
    larger ones for element-wise operations — NO explicit loop needed.

    Rule: shapes are compatible right-to-left. Missing dims are treated as 1.

    Critical for ML:
      - Adding bias vector to every row of a weight matrix
      - Normalising a batch of embeddings
      - Computing pairwise differences
    """
    # Simulate a batch of 4 embedding vectors (dim=3)
    embeddings = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [2.0, 4.0, 6.0],
    ])  # shape (4, 3)

    bias = np.array([0.1, 0.2, 0.3])   # shape (3,) ← broadcast across rows

    # ── Normalise each row to unit length (L2 norm) ──────────────────────
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)  # shape (4, 1)
    normalised = embeddings / norms    # (4,3) / (4,1) → broadcasts correctly

    # ── Pairwise cosine similarity matrix ─────────────────────────────────
    # dot(A, A.T) when rows are already unit-normalised = cosine similarity
    sim_matrix = normalised @ normalised.T   # shape (4, 4)

    return {
        "with_bias":    (embeddings + bias).tolist(),
        "normalised":   normalised.round(4).tolist(),
        "sim_matrix":   sim_matrix.round(4).tolist(),
        "diagonal_all_ones": bool(np.allclose(np.diag(sim_matrix), 1.0)),
    }


def numpy_for_ml() -> dict:
    """
    Common NumPy patterns used daily in ML pipelines:
    - random data generation (synthetic datasets)
    - argmax / argmin (model predictions)
    - clip (bound activations)
    - where (conditional assignment)
    - percentile (feature capping / outlier detection)
    """
    rng = np.random.default_rng(42)

    # Simulate model output logits for 5 samples, 3 classes
    logits = rng.standard_normal((5, 3))
    preds  = logits.argmax(axis=1)           # predicted class indices

    # Softmax (convert logits to probabilities)
    def softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(axis=1, keepdims=True))  # subtract max for stability
        return e / e.sum(axis=1, keepdims=True)

    probs = softmax(logits)

    # Outlier capping: any value > 95th percentile gets capped
    feature = rng.normal(loc=0, scale=1, size=1000)
    p95     = np.percentile(feature, 95)
    capped  = np.clip(feature, -np.inf, p95)

    return {
        "predictions":       preds.tolist(),
        "max_confidence":    probs.max(axis=1).round(3).tolist(),
        "probabilities_sum": probs.sum(axis=1).round(6).tolist(),  # should all be 1.0
        "p95_threshold":     round(float(p95), 3),
        "values_capped":     int((feature > p95).sum()),
    }


# ══════════════════════════════════════════════════════
# PART 2 — PANDAS DATA WRANGLING
# ══════════════════════════════════════════════════════

def create_ai_dataset() -> pd.DataFrame:
    """
    Synthetic dataset simulating an AI model registry.
    Each row = one experiment run.
    """
    rng = np.random.default_rng(42)
    n   = 120

    algorithms  = ["logistic_regression", "random_forest", "xgboost", "svm", "neural_net"]
    datasets    = ["iris", "titanic", "mnist_small", "fraud_detection", "sentiment"]
    teams       = ["nlp_team", "cv_team", "tabular_team"]
    environments= ["dev", "staging", "prod"]

    df = pd.DataFrame({
        "run_id":       [f"run_{i:04d}" for i in range(n)],
        "algorithm":    rng.choice(algorithms, n),
        "dataset":      rng.choice(datasets, n),
        "team":         rng.choice(teams, n),
        "environment":  rng.choice(environments, n, p=[0.5, 0.3, 0.2]),
        "accuracy":     rng.uniform(0.60, 0.99, n).round(4),
        "f1_score":     rng.uniform(0.55, 0.98, n).round(4),
        "latency_ms":   rng.uniform(10, 500, n).round(1),
        "training_time_s": rng.uniform(5, 300, n).round(1),
        "num_features": rng.choice([4, 8, 16, 32, 64, 128], n),
        "deployed":     rng.choice([True, False], n, p=[0.25, 0.75]),
    })
    return df


def wrangling_essentials(df: pd.DataFrame) -> dict:
    """
    Core Pandas operations every data engineer must know.
    Think of DataFrame as a spreadsheet with superpowers:
      - filter rows → WHERE clause
      - select columns → SELECT columns
      - add columns → derived/computed columns
      - sort → ORDER BY
    """
    # ── Filtering ──────────────────────────────────────
    high_acc = df[df["accuracy"] > 0.90]
    prod_runs = df[df["environment"] == "prod"]
    deployed_rf = df[(df["deployed"]) & (df["algorithm"] == "random_forest")]

    # ── Column operations ──────────────────────────────
    df2 = df.copy()
    df2["accuracy_pct"]    = (df2["accuracy"] * 100).round(1)
    df2["fast"]            = df2["latency_ms"] < 100
    df2["score_composite"] = (df2["accuracy"] * 0.7 + df2["f1_score"] * 0.3).round(4)

    # ── Sorting ────────────────────────────────────────
    top5 = df2.nlargest(5, "score_composite")[["run_id", "algorithm", "score_composite"]]

    # ── Missing values ─────────────────────────────────
    df_with_nulls = df.copy()
    null_idx = df_with_nulls.sample(10, random_state=99).index
    df_with_nulls.loc[null_idx, "accuracy"] = np.nan
    null_count = df_with_nulls["accuracy"].isna().sum()
    filled = df_with_nulls["accuracy"].fillna(df_with_nulls["accuracy"].median())

    return {
        "total_rows":         len(df),
        "high_acc_count":     len(high_acc),
        "prod_count":         len(prod_runs),
        "deployed_rf_count":  len(deployed_rf),
        "top5":               top5.to_dict("records"),
        "null_count":         int(null_count),
        "filled_nulls":       int(filled.notna().sum()),
    }


def groupby_operations(df: pd.DataFrame) -> pd.DataFrame:
    """
    groupby = SQL GROUP BY.
    Core pattern for aggregating metrics by dimension.

    Real-world: 'What is the average accuracy per algorithm?'
    This is how you build a model leaderboard.
    """
    summary = (
        df.groupby("algorithm")
        .agg(
            runs          = ("run_id",         "count"),
            avg_accuracy  = ("accuracy",        "mean"),
            max_accuracy  = ("accuracy",        "max"),
            avg_f1        = ("f1_score",        "mean"),
            avg_latency   = ("latency_ms",      "mean"),
            deployed_count= ("deployed",        "sum"),
        )
        .round(4)
        .sort_values("avg_accuracy", ascending=False)
        .reset_index()
    )
    return summary


def merge_operations() -> pd.DataFrame:
    """
    merge = SQL JOIN.

    Two tables:
      runs    — experiment run results
      models  — model metadata (owner, version, tags)

    Inner join → only runs that have a matching model entry.
    Left join  → all runs, model info where available (NaN where missing).
    """
    runs = pd.DataFrame({
        "run_id":     ["r1", "r2", "r3", "r4"],
        "algorithm":  ["random_forest", "xgboost", "svm", "neural_net"],
        "accuracy":   [0.92, 0.95, 0.88, 0.97],
    })

    models = pd.DataFrame({
        "algorithm": ["random_forest", "xgboost", "neural_net"],
        "owner":     ["alice",         "bob",       "carol"],
        "version":   ["v2.1",          "v1.0",      "v3.5"],
        "tags":      [["tree", "ensemble"], ["boost"], ["dl", "gpu"]],
    })

    # Inner join — only matched rows
    inner = runs.merge(models, on="algorithm", how="inner")
    # Left join — all runs, NaN for unmatched
    left  = runs.merge(models, on="algorithm", how="left")

    return left


# ══════════════════════════════════════════════════════
# PART 3 — FILE FORMATS (Parquet, CSV, JSON)
# ══════════════════════════════════════════════════════

def file_format_comparison(df: pd.DataFrame) -> dict:
    """
    Parquet vs CSV vs JSON — why it matters for data engineering:

    CSV:     Human readable, universal, NO schema, slow, large.
    JSON:    Semi-structured, nested data OK, verbose, slow for analytics.
    Parquet: Columnar, compressed, typed schema, 10–100x faster for analytics.
             Industry standard for data lakes (Databricks, Snowflake, BigQuery).

    Rule of thumb:
      - Raw ingestion from APIs: JSON
      - Spreadsheet handoffs / legacy systems: CSV
      - Everything in your data lake / ML feature store: Parquet
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)

        # Write
        df.to_csv(base / "data.csv",       index=False)
        df.to_json(base / "data.json",      orient="records", indent=2)
        df.to_parquet(base / "data.parquet", index=False, compression="snappy")

        # File sizes
        csv_size     = (base / "data.csv").stat().st_size
        json_size    = (base / "data.json").stat().st_size
        parquet_size = (base / "data.parquet").stat().st_size

        # Read back
        df_csv     = pd.read_csv(base / "data.csv")
        df_json    = pd.read_json(base / "data.json")
        df_parquet = pd.read_parquet(base / "data.parquet")

        return {
            "rows":         len(df),
            "csv_bytes":    csv_size,
            "json_bytes":   json_size,
            "parquet_bytes": parquet_size,
            "parquet_vs_csv_ratio": round(parquet_size / csv_size, 3),
            "schemas_match": list(df_parquet.columns) == list(df.columns),
            "parquet_dtypes": {col: str(t) for col, t in df_parquet.dtypes.items()},
        }


# ══════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    import pprint

    print("=" * 60)
    print("NUMPY & PANDAS ESSENTIALS DEMO")
    print("=" * 60)

    print("\n[1] Array basics")
    pprint.pprint(array_basics())

    print("\n[2] Broadcasting")
    r = broadcasting_demo()
    print(f"  diagonal all 1s: {r['diagonal_all_ones']}")
    print(f"  sim_matrix:\n{np.array(r['sim_matrix']).round(3)}")

    print("\n[3] NumPy for ML")
    pprint.pprint(numpy_for_ml())

    print("\n[4] DataFrame wrangling")
    df = create_ai_dataset()
    pprint.pprint(wrangling_essentials(df))

    print("\n[5] GroupBy leaderboard")
    print(groupby_operations(df).to_string(index=False))

    print("\n[6] Merge (SQL JOIN)")
    print(merge_operations().to_string(index=False))

    print("\n[7] File format comparison")
    pprint.pprint(file_format_comparison(df))
