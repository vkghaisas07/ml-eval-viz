# from __future__ import annotations
# from typing import Any, Dict, Iterable, Optional
# from .metrics import classification_metrics
# from .plots import plot_confusion_matrix, plot_roc, plot_pr

# def evaluate_classifier(
#     model,
#     X: Any,
#     y_true: Iterable,
#     *, 
#     show_plots: bool = True,
#     average: str = "weighted",
#     labels: Optional[Iterable] = None,
# ) -> Dict[str, Any]:
#     """Fits (if needed) and evaluates a classifier with quick visuals.
    
#     If model is already fitted, it will just predict.
#     Returns a metrics dict (good for logging / MLflow).
#     """
#     # Predict
#     try:
#         y_pred = model.predict(X)
#     except Exception:
#         model.fit(X, y_true)
#         y_pred = model.predict(X)

#     y_score = None
#     if hasattr(model, "predict_proba"):
#         y_proba = model.predict_proba(X)
#         # binary convert to score for positive class
#         if getattr(y_proba, "ndim", 1) == 2 and y_proba.shape[1] == 2:
#             y_score = y_proba[:, 1]
#         else:
#             y_score = y_proba
#     elif hasattr(model, "decision_function"):
#         y_score = model.decision_function(X)

#     report = classification_metrics(y_true, y_pred, y_prob=y_score, average=average, labels=labels)

#     if show_plots:
#         plot_confusion_matrix(y_true, y_pred, labels=labels)
#         if y_score is not None and (getattr(y_score, "ndim", 1) == 1):
#             plot_roc(y_true, y_score)
#             plot_pr(y_true, y_score)

#     return report



from __future__ import annotations
from typing import Any, Dict, Iterable, Optional
from .metrics import classification_metrics
from .plots import plot_confusion_matrix, plot_roc, plot_pr

def _ensure_fitted(model, X, y_true):
    # Many sklearn-compatible estimators expose `classes_` after fit.
    if not hasattr(model, "classes_"):
        model.fit(X, y_true)

def evaluate_classifier(
    model,
    X: Any,
    y_true: Iterable,
    *, 
    show_plots: bool = True,
    average: str = "weighted",
    labels: Optional[Iterable] = None,
    proba_only: bool = True,   # <-- default to proba-only
) -> Dict[str, Any]:
    """Evaluate a classifier with quick visuals.

    - If proba_only=True, never calls model.predict(). It only uses predict_proba (or decision_function).
    - If proba_only=False, behaves like before (will compute y_pred when needed).

    Returns a metrics dict (good for logging / MLflow).
    """
    # Fit if needed
    _ensure_fitted(model, X, y_true)

    y_pred = None
    y_score = None

    # Prefer probabilities
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)
        # Binary: compress to positive-class probability
        if getattr(y_proba, "ndim", 1) == 2 and y_proba.shape[1] == 2:
            y_score = y_proba[:, 1]
        else:
            y_score = y_proba
    elif hasattr(model, "decision_function"):
        # Fall back to raw decision scores if no predict_proba is available
        y_score = model.decision_function(X)

    # Only compute hard labels if explicitly allowed
    if not proba_only:
        try:
            y_pred = model.predict(X)
        except Exception:
            y_pred = None

    report = classification_metrics(
        y_true=y_true,
        y_pred=y_pred if not proba_only else None,
        y_prob=y_score,
        average=average,
        labels=labels,
    )

    if show_plots:
        # Confusion matrix only makes sense with hard labels
        if (not proba_only) and (y_pred is not None):
            plot_confusion_matrix(y_true, y_pred, labels=labels)

        # ROC/PR curves are score-based and only for binary if 1D scores
        if y_score is not None and (getattr(y_score, "ndim", 1) == 1):
            plot_roc(y_true, y_score)
            plot_pr(y_true, y_score)

    return report