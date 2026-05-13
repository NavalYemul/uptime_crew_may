"""
visualizations.py — 5 Matplotlib / Seaborn Charts
===================================================
Covers: confusion matrix, ROC curve, feature distributions,
        2D embedding scatter, model comparison bar chart.

Run:  python -m day1.visualizations
"""

from __future__ import annotations

import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="husl")
PALETTE = sns.color_palette("husl", 10)


# ══════════════════════════════════════════════════════
# VIZ 1 — CONFUSION MATRIX HEATMAP
# ══════════════════════════════════════════════════════

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Confusion Matrix",
    save_path: str | None = None,
) -> None:
    """
    Rows = actual class, Columns = predicted class.
    Diagonal = correct predictions (want these high).
    Off-diagonal = misclassifications (want these low).

    Reading it:
      cm[i, j] = how many samples of class i were predicted as class j.
    """
    cm = confusion_matrix(y_true, y_pred)
    acc = np.trace(cm) / cm.sum()

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names or list(range(cm.shape[1])),
        yticklabels=class_names or list(range(cm.shape[0])),
        linewidths=0.5, linecolor="white", ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"{title}\nAccuracy = {acc:.2%}", fontsize=14, fontweight="bold", pad=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════
# VIZ 2 — ROC CURVE (One-vs-Rest)
# ══════════════════════════════════════════════════════

def plot_roc_curve(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "ROC Curve (One-vs-Rest)",
    save_path: str | None = None,
) -> None:
    """
    ROC = Receiver Operating Characteristic.
    Plots True Positive Rate (Recall) vs False Positive Rate.

    AUC = Area Under Curve.
      AUC = 1.0 → perfect model
      AUC = 0.5 → random guessing (diagonal dashed line)
      AUC = 0.0 → perfectly wrong (flip predictions → perfect model)

    One-vs-Rest: each class gets its own curve.
    """
    n_classes = len(np.unique(y_test))
    y_bin     = label_binarize(y_test, classes=list(range(n_classes)))
    y_score   = model.predict_proba(X_test)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        label   = class_names[i] if class_names else f"Class {i}"
        ax.plot(fpr, tpr, linewidth=2.2,
                label=f"{label}  (AUC = {roc_auc:.3f})",
                color=PALETTE[i])

    # Random baseline
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Random (AUC = 0.50)", alpha=0.6)
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color="gray")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════
# VIZ 3 — FEATURE DISTRIBUTIONS BY CLASS
# ══════════════════════════════════════════════════════

def plot_feature_distributions(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | None = None,
    class_names: list[str] | None = None,
    title: str = "Feature Distributions by Class",
    save_path: str | None = None,
) -> None:
    """
    Violin plots — show spread, median, and density of each feature per class.
    Use this to:
      • understand which features separate classes best
      • spot outliers
      • check if classes are linearly separable
    """
    import pandas as pd

    cols = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
    df   = pd.DataFrame(X, columns=cols)
    df["class"] = [class_names[c] if class_names else str(c) for c in y]

    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for col, ax in zip(cols, axes):
        sns.violinplot(
            data=df, x="class", y=col,
            palette="husl", ax=ax, inner="box", linewidth=0.8,
        )
        ax.set_title(col.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15, labelsize=8)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════
# VIZ 4 — 2D EMBEDDING / VECTOR SPACE SCATTER
# ══════════════════════════════════════════════════════

def plot_2d_embeddings(
    vectors: np.ndarray,
    labels: list[str],
    color_ids: list[int] | None = None,
    title: str = "2D Vector Space (PCA)",
    save_path: str | None = None,
) -> None:
    """
    Projects high-dimensional vectors to 2D via PCA and scatter-plots them.
    Similar items cluster together — validates that embeddings capture semantics.
    """
    if vectors.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(vectors)
        explained = pca.explained_variance_ratio_.sum()
        subtitle = f"PCA explains {explained:.1%} of variance"
    else:
        coords  = vectors
        subtitle = ""

    fig, ax = plt.subplots(figsize=(10, 7))
    c = color_ids if color_ids is not None else list(range(len(labels)))
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=c, cmap="tab10", s=110,
        alpha=0.85, edgecolors="white", linewidths=0.7,
    )

    for i, lbl in enumerate(labels):
        ax.annotate(
            lbl, (coords[i, 0], coords[i, 1]),
            fontsize=7.5, textcoords="offset points",
            xytext=(5, 4), alpha=0.9,
        )

    ax.axhline(0, color="lightgray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="lightgray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Component 1", fontsize=11)
    ax.set_ylabel("Component 2", fontsize=11)
    ax.set_title(f"{title}\n{subtitle}", fontsize=13, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════
# VIZ 5 — MODEL COMPARISON BAR CHART
# ══════════════════════════════════════════════════════

def plot_model_comparison(
    model_metrics: dict[str, dict[str, float]],
    title: str = "Model Comparison",
    save_path: str | None = None,
) -> None:
    """
    Grouped bar chart comparing multiple models across multiple metrics.
    Quick visual to pick the best model for deployment.
    """
    models  = list(model_metrics.keys())
    metrics = sorted({m for vals in model_metrics.values() for m in vals})

    x     = np.arange(len(metrics))
    width = 0.75 / len(models)

    fig, ax = plt.subplots(figsize=(max(9, len(metrics) * 2.2), 5))

    for i, model in enumerate(models):
        vals   = [model_metrics[model].get(m, 0.0) for m in metrics]
        offset = (i - len(models) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width, label=model, alpha=0.88, color=PALETTE[i])

        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.yaxis.grid(True, alpha=0.35)
    ax.set_axisbelow(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════
# ALL-IN-ONE DEMO
# ══════════════════════════════════════════════════════

def run_all_visualizations() -> None:
    """Trains two models on Iris, then generates all 5 charts."""
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names   = iris.target_names.tolist()
    feature_names = iris.feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y,
    )
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_s, y_train)
    rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train_s, y_train)

    y_pred_lr = lr.predict(X_test_s)
    y_pred_rf = rf.predict(X_test_s)

    print("[Viz 1] Confusion Matrix")
    plot_confusion_matrix(y_test, y_pred_lr, class_names, "Confusion Matrix — Logistic Regression")

    print("[Viz 2] ROC Curve")
    plot_roc_curve(lr, X_test_s, y_test, class_names)

    print("[Viz 3] Feature Distributions")
    plot_feature_distributions(X, y, list(feature_names), class_names)

    print("[Viz 4] 2D Embedding Scatter")
    short_labels = [f"{class_names[c]}_{i}" for i, c in enumerate(y_test[:30])]
    plot_2d_embeddings(X_test_s[:30], short_labels, y_test[:30].tolist(),
                       "Iris Test Set — PCA 2D Projection")

    print("[Viz 5] Model Comparison")
    def _metrics(y_true, y_pred):
        return {
            "accuracy":  accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="macro"),
            "recall":    recall_score(y_true, y_pred, average="macro"),
            "f1_score":  f1_score(y_true, y_pred, average="macro"),
        }

    plot_model_comparison(
        {
            "LogisticRegression": _metrics(y_test, y_pred_lr),
            "RandomForest":       _metrics(y_test, y_pred_rf),
        },
        "Model Comparison — Iris Dataset",
    )


if __name__ == "__main__":
    run_all_visualizations()
