from __future__ import annotations
from typing import Iterable, Optional, Dict, Any
import numpy as np
from sklearn import metrics as skm

def _to_numpy(x: Iterable) -> np.ndarray:
    return np.asarray(list(x))

def classification_metrics(
    y_true: Iterable,
    y_pred: Optional[Iterable] = None,
    y_prob: Optional[Iterable] = None,
    average: str = "weighted",
    labels: Optional[Iterable] = None,
) -> Dict[str, Any]:
    """Compute classification metrics.

    Works in two modes:
    - Label-based (if y_pred provided): accuracy, precision, recall, f1
    - Probability-only (if y_prob provided): AUC(ROC/PR), plus proper scoring (log_loss, Brier)

    For binary:
      y_prob can be a 1D array of positive-class probabilities.
    For multiclass:
      y_prob should be shape (n_samples, n_classes) with rows summing to 1.
    """
    y_true = _to_numpy(y_true)
    out: Dict[str, Any] = {"support": int(y_true.shape[0])}

    # Label-based metrics only if y_pred is provided
    if y_pred is not None:
        y_pred = _to_numpy(y_pred)
        out.update({
            "accuracy": float(skm.accuracy_score(y_true, y_pred)),
            "precision": float(skm.precision_score(y_true, y_pred, average=average, zero_division=0)),
            "recall": float(skm.recall_score(y_true, y_pred, average=average, zero_division=0)),
            "f1": float(skm.f1_score(y_true, y_pred, average=average, zero_division=0)),
        })

    # Probability / score-based metrics
    if y_prob is not None:
        y_prob = _to_numpy(y_prob)
        try:
            if y_prob.ndim == 1:
                # Binary: positive-class probability or decision scores
                out["auc_roc"] = float(skm.roc_auc_score(y_true, y_prob))
                precision, recall, _ = skm.precision_recall_curve(y_true, y_prob)
                out["auc_pr"] = float(skm.auc(recall, precision))
                # Proper scoring rules only when y_prob is a probability in [0,1]
                # If you pass raw scores (e.g., decision_function), log_loss/Brier may not be meaningful.
                # Here we assume probabilities for binary when 1D.
                out["log_loss"] = float(skm.log_loss(y_true, y_prob, labels=[0, 1]))
                out["brier"] = float(skm.brier_score_loss(y_true, y_prob))
            else:
                # Multiclass: OVR averaging for ROC AUC, macro-avg PR AUC
                out["auc_roc_ovr"] = float(skm.roc_auc_score(y_true, y_prob, multi_class="ovr", average=average))
                pr_aucs = []
                n_classes = y_prob.shape[1]
                for i in range(n_classes):
                    bin_true = (y_true == i).astype(int)
                    precision, recall, _ = skm.precision_recall_curve(bin_true, y_prob[:, i])
                    pr_aucs.append(skm.auc(recall, precision))
                out["auc_pr"] = float(np.mean(pr_aucs))
                out["log_loss"] = float(skm.log_loss(y_true, y_prob))
        except Exception as e:
            out["auc_error"] = str(e)

    return out

def regression_metrics(
    y_true: Iterable,
    y_pred: Iterable,
) -> Dict[str, float]:
    """Compute common regression metrics."""
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    mae = skm.mean_absolute_error(y_true, y_pred)
    mse = skm.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = skm.r2_score(y_true, y_pred)
    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "support": int(y_true.shape[0]),
    }