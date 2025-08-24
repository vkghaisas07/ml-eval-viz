# read version from installed package
#from importlib.metadata import version
#__version__ = version("ml_eval_viz")

from .metrics import classification_metrics, regression_metrics
from .plots import plot_confusion_matrix, plot_roc, plot_pr, plot_residuals, plot_feature_importance
from .sklearn_helpers import evaluate_classifier

__all__ = [
    "classification_metrics",
    "regression_metrics",
    "plot_confusion_matrix",
    "plot_roc",
    "plot_pr",
    "plot_residuals",
    "plot_feature_importance",
    "evaluate_classifier",
]