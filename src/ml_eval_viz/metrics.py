from __future__ import annotations
from typing import Iterable, Optional, Dict, Any
import numpy as np
from sklearn import metrics as skm

def _to_numpy(x: Iterable) -> np.ndarray:
    return np.asarray(list(x))

def classification_metrics(
    y_true: Iterable,
    y_pred: Iterable,
    y_prob: Optional[Iterable] = None,
    average: str = "weighted",
    labels: Optional[Iterable] = None,
) -> Dict[str, Any]:
    """Compute common classification metrics.
    Parameters
    ----------
    y_true : iterable of shape (n_samples,)
    y_pred : iterable of shape (n_samples,)
    y_prob : iterable of shape (n_samples,) or (n_samples, n_classes), optional
        Probabilities or decision scores for ROC/PR AUC.
    average : str
        Averaging method for precision/recall/f1 ('micro','macro','weighted').
    labels : iterable, optional
        Class labels (for confusion matrix consistency across runs).
    Returns
    -------
    dict with accuracy, precision, recall, f1, auc_roc, auc_pr, and support.
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    out = {
        "accuracy": float(skm.accuracy_score(y_true, y_pred)),
        "precision": float(skm.precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(skm.recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(skm.f1_score(y_true, y_pred, average=average, zero_division=0)),
        "support": int(y_true.shape[0]),
    }
    if y_prob is not None:
        y_prob = _to_numpy(y_prob)
        # binary: y_prob is scores for positive class; multiclass: one-vs-rest
        try:
            if y_prob.ndim == 1:
                out["auc_roc"] = float(skm.roc_auc_score(y_true, y_prob))
                precision, recall, _ = skm.precision_recall_curve(y_true, y_prob)
                out["auc_pr"] = float(skm.auc(recall, precision))
            else:
                out["auc_roc_ovr"] = float(skm.roc_auc_score(y_true, y_prob, multi_class="ovr", average=average))
                # For PR AUC in multiclass, average over classes
                pr_aucs = []
                # assume classes are 0..n-1 for binarization; map labels otherwise as needed
                for i in range(y_prob.shape[1]):
                    bin_true = (y_true == i).astype(int)
                    precision, recall, _ = skm.precision_recall_curve(bin_true, y_prob[:, i])
                    pr_aucs.append(skm.auc(recall, precision))
                out["auc_pr"] = float(np.mean(pr_aucs))
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