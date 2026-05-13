"""test_numpy_pandas.py — NumPy + Pandas tests"""

import numpy as np
import pandas as pd
import pytest

from day2.numpy_pandas import (
    array_basics,
    broadcasting_demo,
    numpy_for_ml,
    create_ai_dataset,
    wrangling_essentials,
    groupby_operations,
    merge_operations,
    file_format_comparison,
)


# ── NumPy ────────────────────────────────────────────

def test_array_basics_keys():
    r = array_basics()
    assert {"sum", "mean", "std", "dot_ab", "element_mul", "reshaped"} <= r.keys()


def test_array_basics_values():
    r = array_basics()
    assert r["sum"]   == pytest.approx(15.0)
    assert r["mean"]  == pytest.approx(3.0)
    assert r["dot_ab"] == pytest.approx(55.0)


def test_broadcasting_normalised():
    r = broadcasting_demo()
    norms = [sum(x**2 for x in row) ** 0.5 for row in r["normalised"]]
    for norm in norms:
        assert norm == pytest.approx(1.0, abs=1e-3)


def test_broadcasting_diagonal_ones():
    r = broadcasting_demo()
    assert r["diagonal_all_ones"] is True


def test_broadcasting_sim_matrix_shape():
    r = broadcasting_demo()
    matrix = r["sim_matrix"]
    assert len(matrix) == 4
    assert all(len(row) == 4 for row in matrix)


def test_broadcasting_symmetric():
    r = broadcasting_demo()
    m = np.array(r["sim_matrix"])
    np.testing.assert_array_almost_equal(m, m.T)


def test_numpy_for_ml_predictions_shape():
    r = numpy_for_ml()
    assert len(r["predictions"]) == 5
    assert all(p in [0, 1, 2] for p in r["predictions"])


def test_numpy_for_ml_probs_sum_to_one():
    r = numpy_for_ml()
    for s in r["probabilities_sum"]:
        assert s == pytest.approx(1.0, abs=1e-5)


def test_numpy_for_ml_capping():
    r = numpy_for_ml()
    assert r["values_capped"] > 0
    assert r["values_capped"] < 100   # should cap ~5% of 1000 samples


# ── Pandas ───────────────────────────────────────────

def test_dataset_shape():
    df = create_ai_dataset()
    assert df.shape == (120, 11)


def test_dataset_columns():
    df = create_ai_dataset()
    expected = {"run_id", "algorithm", "dataset", "team", "environment",
                "accuracy", "f1_score", "latency_ms", "training_time_s",
                "num_features", "deployed"}
    assert expected <= set(df.columns)


def test_dataset_accuracy_range():
    df = create_ai_dataset()
    assert df["accuracy"].between(0.0, 1.0).all()


def test_wrangling_essentials_keys(sample_df):
    df = create_ai_dataset()
    r  = wrangling_essentials(df)
    assert {"total_rows", "high_acc_count", "prod_count", "top5"} <= r.keys()


def test_wrangling_essentials_counts(sample_df):
    df = create_ai_dataset()
    r  = wrangling_essentials(df)
    assert r["total_rows"] == 120
    assert 0 <= r["high_acc_count"] <= 120


def test_wrangling_top5_length():
    df = create_ai_dataset()
    r  = wrangling_essentials(df)
    assert len(r["top5"]) == 5


def test_groupby_has_all_algorithms():
    df      = create_ai_dataset()
    summary = groupby_operations(df)
    algos   = set(summary["algorithm"].tolist())
    assert algos == {"logistic_regression", "random_forest", "xgboost", "svm", "neural_net"}


def test_groupby_avg_accuracy_in_range():
    df      = create_ai_dataset()
    summary = groupby_operations(df)
    assert summary["avg_accuracy"].between(0.0, 1.0).all()


def test_groupby_run_counts_sum_to_total():
    df      = create_ai_dataset()
    summary = groupby_operations(df)
    assert summary["runs"].sum() == 120


def test_merge_left_preserves_all_runs():
    merged = merge_operations()
    assert len(merged) == 4   # all 4 runs


def test_merge_inner_excludes_unmatched():
    merged = merge_operations()
    # 'svm' has no match in models table → should have NaN owner
    svm_row = merged[merged["algorithm"] == "svm"]
    assert svm_row["owner"].isna().all()


def test_file_format_parquet_smaller_than_json(sample_df):
    df = create_ai_dataset()
    r  = file_format_comparison(df)
    assert r["parquet_bytes"] < r["json_bytes"]


def test_file_format_schemas_match(sample_df):
    df = create_ai_dataset()
    r  = file_format_comparison(df)
    assert r["schemas_match"] is True


def test_file_format_row_count(sample_df):
    df = create_ai_dataset()
    r  = file_format_comparison(df)
    assert r["rows"] == 120
