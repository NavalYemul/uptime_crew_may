"""
conftest.py — Shared pytest fixtures
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from day1.models import Course, DataPoint, Dataset, Student, TrainingRequest


@pytest.fixture
def sample_student() -> Student:
    return Student("Alice", 25, "alice@example.com", gpa=3.5)


@pytest.fixture
def sample_course() -> Course:
    return Course("CS101", "Intro to AI", max_students=3)


@pytest.fixture
def sample_training_request() -> TrainingRequest:
    return TrainingRequest(
        model_name="iris_clf",
        algorithm="random_forest",
        features=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        target="species",
    )


@pytest.fixture
def xor_dataset() -> Dataset:
    ds = Dataset("XOR")
    for features, label in [([0.0, 0.0], "F"), ([1.0, 1.0], "F"),
                              ([0.0, 1.0], "T"), ([1.0, 0.0], "T")]:
        ds.add(DataPoint(features, label))
    return ds


@pytest.fixture
def iris_split():
    """Returns (X_train, X_test, y_train, y_test) scaled and split."""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target,
        test_size=0.2, random_state=42, stratify=iris.target,
    )
    scaler = StandardScaler()
    return (
        scaler.fit_transform(X_train),
        scaler.transform(X_test),
        y_train,
        y_test,
    )


@pytest.fixture
def nlp_corpus() -> list[str]:
    return [
        "machine learning is powerful and widely used",
        "deep learning uses neural networks with many layers",
        "python is great for data science and machine learning",
        "natural language processing enables text understanding",
        "word embeddings capture semantic meaning of words",
    ]
