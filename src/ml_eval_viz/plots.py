from __future__ import annotations
from typing import Iterable, Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics as skm

def _to_numpy(x: Iterable) -> np.ndarray:
    return np.asarray(list(x))

def _maybe_tight():
    try:
        plt.tight_layout()
    except Exception:
        pass

def plot_confusion_matrix(
    y_true: Iterable,
    y_pred: Iterable,
    labels: Optional[Sequence] = None,
    normalize: Optional[str] = None,
    title: Optional[str] = None,
):
    cm = skm.confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    disp = skm.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(values_format=".2f" if normalize else "d", colorbar=True)
    if title:
        plt.title(title)
    _maybe_tight()
    plt.show()

def plot_roc(
    y_true: Iterable,
    y_score: Iterable,
    pos_label: Optional[int] = None,
    title: Optional[str] = None,
):
    y_true = _to_numpy(y_true)
    y_score = _to_numpy(y_score)
    fpr, tpr, _ = skm.roc_curve(y_true, y_score, pos_label=pos_label)
    auc_val = skm.auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc_val:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title or "ROC Curve")
    plt.legend(loc="lower right")
    _maybe_tight()
    plt.show()

def plot_pr(
    y_true: Iterable,
    y_score: Iterable,
    pos_label: Optional[int] = None,
    title: Optional[str] = None,
):
    y_true = _to_numpy(y_true)
    y_score = _to_numpy(y_score)
    precision, recall, _ = skm.precision_recall_curve(y_true, y_score, pos_label=pos_label)
    auc_val = skm.auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f"PR AUC = {auc_val:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title or "Precision-Recall Curve")
    plt.legend(loc="lower left")
    _maybe_tight()
    plt.show()

def plot_residuals(
    y_true: Iterable,
    y_pred: Iterable,
    title: Optional[str] = None,
):
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    residuals = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals (y_true - y_pred)")
    plt.title(title or "Residuals vs Predicted")
    _maybe_tight()
    plt.show()

def plot_feature_importance(
    model,
    feature_names: Sequence[str],
    top_n: int = 20,
    title: Optional[str] = None,
):
    """Plot feature importances for tree-based models or coefficients for linear models."""
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        importances = np.abs(coef) if coef.ndim == 1 else np.mean(np.abs(coef), axis=0)
    else:
        raise ValueError("Model has neither 'feature_importances_' nor 'coef_'.")

    idx = np.argsort(importances)[::-1][:top_n]
    plt.figure()
    plt.bar(range(len(idx)), importances[idx])
    labels = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in idx]
    plt.xticks(range(len(idx)), labels, rotation=45, ha="right")
    plt.title(title or "Feature Importance")
    plt.ylabel("Importance (abs coef or gain)")
    _maybe_tight()
    plt.show()