from __future__ import annotations
from typing import Any, Dict, Iterable, Optional
from .metrics import classification_metrics
from .plots import plot_confusion_matrix, plot_roc, plot_pr

def evaluate_classifier(
    model,
    X: Any,
    y_true: Iterable,
    *, 
    show_plots: bool = True,
    average: str = "weighted",
    labels: Optional[Iterable] = None,
) -> Dict[str, Any]:
    """Fits (if needed) and evaluates a classifier with quick visuals.
    
    If model is already fitted, it will just predict.
    Returns a metrics dict (good for logging / MLflow).
    """
    # Predict
    try:
        y_pred = model.predict(X)
    except Exception:
        model.fit(X, y_true)
        y_pred = model.predict(X)

    y_score = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)
        # binary convert to score for positive class
        if getattr(y_proba, "ndim", 1) == 2 and y_proba.shape[1] == 2:
            y_score = y_proba[:, 1]
        else:
            y_score = y_proba
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X)

    report = classification_metrics(y_true, y_pred, y_prob=y_score, average=average, labels=labels)

    if show_plots:
        plot_confusion_matrix(y_true, y_pred, labels=labels)
        if y_score is not None and (getattr(y_score, "ndim", 1) == 1):
            plot_roc(y_true, y_score)
            plot_pr(y_true, y_score)

    return report