"""
ml_workflow.py — End-to-End ML Pipeline
=========================================
Covers: AI vs ML vs DL taxonomy, supervised learning,
        ML workflow, classification vs regression,
        evaluation metrics (precision/recall/F1/AUC-ROC/confusion matrix),
        cross-validation, sklearn Pipeline.

Run:  python -m day1.ml_workflow
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════
# CONCEPT MAP
# ══════════════════════════════════════════════════════
"""
AI  (Artificial Intelligence)
 └── ML  (Machine Learning)          ← we focus here today
      └── Deep Learning              ← neural nets, many layers

AI  = any technique making machines behave intelligently (incl. rule-based)
ML  = machines that LEARN from data; no hard-coded rules
DL  = ML using deep neural networks; excels at images, text, audio

Supervised Learning   → labelled data; learn input→output mapping
  Classification      → output is a CATEGORY  (spam/not-spam, species)
  Regression          → output is a NUMBER    (house price, temperature)

Unsupervised Learning → no labels; find hidden structure
  Clustering (k-means), Dimensionality Reduction (PCA), Anomaly Detection

ML Workflow:
  1. Data Collection
  2. Preprocessing (scaling, encoding, cleaning)
  3. Train / Test Split
  4. Model Training
  5. Evaluation (metrics, cross-validation)
  6. Iterate
"""


# ══════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════

@dataclass
class MLResult:
    """Container for model evaluation results."""
    model_name: str
    task_type: str                       # "classification" | "regression"
    metrics: dict[str, float]
    cv_scores: list[float] = field(default_factory=list)

    @property
    def cv_mean(self) -> float:
        return float(np.mean(self.cv_scores)) if self.cv_scores else 0.0

    @property
    def cv_std(self) -> float:
        return float(np.std(self.cv_scores)) if self.cv_scores else 0.0

    def summary(self) -> str:
        lines = [f"\n{'─'*45}", f"  Model   : {self.model_name} ({self.task_type})"]
        for k, v in self.metrics.items():
            lines.append(f"  {k:20s}: {v:.4f}")
        if self.cv_scores:
            lines.append(f"  {'CV (mean ± std)':20s}: {self.cv_mean:.4f} ± {self.cv_std:.4f}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════
# STEP 1 — DATA LOADING
# ══════════════════════════════════════════════════════

def load_classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Iris dataset — 150 samples, 4 features, 3 classes."""
    iris = load_iris(as_frame=True)
    X: pd.DataFrame = iris.data
    y: pd.Series = pd.Series(iris.target_names[iris.target], name="species")
    print(f"[Data] Iris: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"[Data] Classes: {y.unique().tolist()}")
    print(f"[Data] Distribution:\n{y.value_counts().to_string()}")
    return X, y


def load_regression_data() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic regression dataset."""
    X, y = make_regression(n_samples=500, n_features=5, noise=10.0, random_state=42)
    print(f"\n[Data] Regression: {X.shape[0]} samples × {X.shape[1]} features")
    return X, y


# ══════════════════════════════════════════════════════
# STEP 2 — PREPROCESSING
# ══════════════════════════════════════════════════════

def preprocess_classification(
    X: pd.DataFrame, y: pd.Series
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    StandardScaler → zero mean, unit variance.
    LabelEncoder  → string class names → integers.

    Fit scaler on TRAINING data only; transform both train & test.
    Fitting on test data = data leakage (like peeking at the exam answers).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"\n[Preprocess] Scaled: mean≈0, std≈1 per feature")
    print(f"[Preprocess] Label map: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    return X_scaled, y_encoded, X.columns.tolist()


# ══════════════════════════════════════════════════════
# STEP 3 — TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════

def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Holdout method: keep test set completely unseen until final evaluation.
    stratify=y → preserves class proportions in both splits.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\n[Split] Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════
# STEP 4 — TRAINING
# ══════════════════════════════════════════════════════

def train_logistic_regression(
    X_train: np.ndarray, y_train: np.ndarray
) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print(f"\n[Train] LogisticRegression on {X_train.shape[0]} samples")
    return model


def train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray
) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(f"[Train] RandomForest (100 trees) on {X_train.shape[0]} samples")
    return model


def train_linear_regression(
    X_train: np.ndarray, y_train: np.ndarray
) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"\n[Train] LinearRegression on {X_train.shape[0]} samples")
    return model


# ══════════════════════════════════════════════════════
# STEP 5 — EVALUATION
# ══════════════════════════════════════════════════════

def evaluate_classifier(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Classifier",
) -> MLResult:
    """
    Classification metrics:
    ─────────────────────────────────────────────────
    Accuracy  = (TP + TN) / total        — overall % correct
    Precision = TP / (TP + FP)           — of predicted positives, how many are real?
    Recall    = TP / (TP + FN)           — of actual positives, how many did we catch?
    F1        = 2 × (P × R) / (P + R)   — harmonic mean (balances P and R)
    AUC-ROC   = area under ROC curve     — 1.0 perfect, 0.5 random

    Use F1 (not accuracy) when classes are imbalanced.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    metrics: dict[str, float] = {
        "accuracy":        accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro"),
        "recall_macro":    recall_score(y_test, y_pred, average="macro"),
        "f1_macro":        f1_score(y_test, y_pred, average="macro"),
    }

    if y_prob is not None:
        metrics["auc_roc"] = roc_auc_score(
            y_test, y_prob, multi_class="ovr", average="macro"
        )

    # Cross-validation on training data (5-fold stratified)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

    result = MLResult(
        model_name=model_name,
        task_type="classification",
        metrics=metrics,
        cv_scores=cv_scores.tolist(),
    )

    print(result.summary())
    print(f"\n  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred)}")
    return result


def evaluate_regressor(
    model: LinearRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "LinearRegression",
) -> MLResult:
    """
    Regression metrics:
    ─────────────────────────────────────────────────
    MSE   = mean squared error        — penalises large errors heavily
    RMSE  = √MSE                      — same unit as target variable
    R²    = % variance explained      — 1.0 perfect, 0.0 = no better than mean
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    metrics: dict[str, float] = {
        "mse":      mse,
        "rmse":     mse ** 0.5,
        "r2_score": r2_score(y_test, y_pred),
    }

    result = MLResult(model_name=model_name, task_type="regression", metrics=metrics)
    print(result.summary())
    return result


# ══════════════════════════════════════════════════════
# SKLEARN PIPELINE — prevents data leakage
# ══════════════════════════════════════════════════════

def build_pipeline() -> Pipeline:
    """
    Pipeline chains preprocessing + model into a single object.
    - fit()      → scaler fits on train data only
    - predict()  → scaler transforms test data using train stats
    Java analogy: interceptor chain or Builder pattern.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, random_state=42)),
    ])


# ══════════════════════════════════════════════════════
# DEMO RUNNERS
# ══════════════════════════════════════════════════════

def run_classification_demo() -> tuple[MLResult, MLResult]:
    print("\n" + "=" * 55)
    print("CLASSIFICATION — Iris Dataset")
    print("=" * 55)

    X, y = load_classification_data()
    X_scaled, y_enc, features = preprocess_classification(X, y)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y_enc)

    lr = train_logistic_regression(X_train, y_train)
    rf = train_random_forest(X_train, y_train)

    lr_result = evaluate_classifier(lr, X_train, X_test, y_train, y_test, "LogisticRegression")
    rf_result = evaluate_classifier(rf, X_train, X_test, y_train, y_test, "RandomForest")

    winner = (
        "RandomForest"
        if rf_result.metrics["f1_macro"] > lr_result.metrics["f1_macro"]
        else "LogisticRegression"
    )
    print(f"\n[Winner] {winner}")
    return lr_result, rf_result


def run_regression_demo() -> MLResult:
    print("\n" + "=" * 55)
    print("REGRESSION — Synthetic Dataset")
    print("=" * 55)

    X, y = load_regression_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    reg = train_linear_regression(X_train, y_train)
    return evaluate_regressor(reg, X_test, y_test)


if __name__ == "__main__":
    lr_res, rf_res = run_classification_demo()
    reg_res        = run_regression_demo()
