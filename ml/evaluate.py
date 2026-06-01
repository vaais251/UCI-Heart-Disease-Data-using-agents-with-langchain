"""
CardioTriage AI — Phase A: Reusable evaluation helpers.

Shared by every modeling step so all models are judged identically.
Provides metric computation, a tidy printout, and two plots
(confusion matrix, ROC curve).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

IMAGES_DIR = Path(__file__).parent.parent / "images"


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Return the standard classification metrics as a dict.

    Specificity is not built into scikit-learn, so we derive it from the
    confusion matrix: of all truly-healthy patients, the fraction we cleared.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),       # = sensitivity
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "f1": f1_score(y_true, y_pred),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    return metrics


def print_metrics(name: str, metrics: dict) -> None:
    """Pretty-print one model's metrics."""
    print(f"\n--- {name} ---")
    for key, value in metrics.items():
        print(f"  {key:12s}: {value:.3f}")


def plot_confusion_matrix(y_true, y_pred, title: str, filename: str) -> None:
    """Save a labeled 2x2 confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["No disease", "Disease"],
        yticklabels=["No disease", "Disease"],
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    IMAGES_DIR.mkdir(exist_ok=True)
    plt.savefig(IMAGES_DIR / filename, dpi=120)
    plt.close()


def plot_roc(curves: list[tuple[str, np.ndarray, np.ndarray]], title: str,
             filename: str) -> None:
    """Save a ROC plot. `curves` is a list of (label, y_true, y_proba)."""
    plt.figure(figsize=(6, 5))
    for label, y_true, y_proba in curves:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance (0.50)")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate (recall)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    IMAGES_DIR.mkdir(exist_ok=True)
    plt.savefig(IMAGES_DIR / filename, dpi=120)
    plt.close()
