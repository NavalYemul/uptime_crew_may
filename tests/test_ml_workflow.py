"""
test_ml_workflow.py — ML pipeline tests
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from day1.ml_workflow import (
    build_pipeline,
    evaluate_classifier,
    evaluate_regressor,
    load_classification_data,
    load_regression_data,
    split_data,
    train_linear_regression,
    train_logistic_regression,
    train_random_forest,
)


def test_classification_data_shape():
    X, y = load_classification_data()
    assert X.shape == (150, 4)
    assert len(y) == 150


def test_classification_data_classes():
    _, y = load_classification_data()
    assert set(y.unique()) == {"setosa", "versicolor", "virginica"}


def test_split_respects_test_size(iris_split):
    X_train, X_test, y_train, y_test = iris_split
    total = len(X_train) + len(X_test)
    assert abs(len(X_test) / total - 0.2) < 0.05


def test_classifier_produces_predictions(iris_split):
    X_train, X_test, y_train, y_test = iris_split
    model = train_logistic_regression(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)


def test_logistic_regression_accuracy(iris_split):
    X_train, X_test, y_train, y_test = iris_split
    model = train_logistic_regression(X_train, y_train)
    result = evaluate_classifier(model, X_train, X_test, y_train, y_test, "LR")
    assert result.metrics["accuracy"] > 0.85


def test_random_forest_accuracy(iris_split):
    X_train, X_test, y_train, y_test = iris_split
    model = train_random_forest(X_train, y_train)
    result = evaluate_classifier(model, X_train, X_test, y_train, y_test, "RF")
    assert result.metrics["accuracy"] >= 0.90


def test_classification_metrics_in_range(iris_split):
    X_train, X_test, y_train, y_test = iris_split
    model = train_logistic_regression(X_train, y_train)
    result = evaluate_classifier(model, X_train, X_test, y_train, y_test, "test")
    for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        assert key in result.metrics
        assert 0.0 <= result.metrics[key] <= 1.0


def test_auc_roc_present(iris_split):
    X_train, X_test, y_train, y_test = iris_split
    model = train_logistic_regression(X_train, y_train)
    result = evaluate_classifier(model, X_train, X_test, y_train, y_test, "LR")
    assert "auc_roc" in result.metrics
    assert result.metrics["auc_roc"] > 0.95


def test_cross_val_scores_populated(iris_split):
    X_train, X_test, y_train, y_test = iris_split
    model = train_logistic_regression(X_train, y_train)
    result = evaluate_classifier(model, X_train, X_test, y_train, y_test, "LR")
    assert len(result.cv_scores) == 5
    assert result.cv_mean > 0.80


def test_regression_r2_positive():
    X, y = load_regression_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model  = train_linear_regression(X_train, y_train)
    result = evaluate_regressor(model, X_test, y_test)
    assert result.metrics["r2_score"] > 0.80


def test_regression_rmse_reasonable():
    X, y = load_regression_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model  = train_linear_regression(X_train, y_train)
    result = evaluate_regressor(model, X_test, y_test)
    assert result.metrics["rmse"] < 200    # synthetic noise is ±10


def test_pipeline_is_sklearn_pipeline():
    pipe = build_pipeline()
    assert isinstance(pipe, Pipeline)
    assert "scaler" in pipe.named_steps
    assert "clf" in pipe.named_steps


def test_pipeline_trains_and_predicts():
    iris = load_iris()
    pipe = build_pipeline()
    pipe.fit(iris.data[:120], iris.target[:120])
    preds = pipe.predict(iris.data[120:])
    assert len(preds) == 30
    assert set(preds).issubset({0, 1, 2})
