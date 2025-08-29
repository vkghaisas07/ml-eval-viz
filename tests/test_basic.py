from ml_eval_viz.metrics import classification_metrics, regression_metrics
import math

def test_classification_metrics_runs():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    y_score = [0.1, 0.8, 0.3, 0.2]
    m = classification_metrics(y_true, y_pred, y_prob=y_score)
    assert 'accuracy' in m and 0.0 <= m['accuracy'] <= 1.0
    assert 'auc_roc' in m and 0.0 <= m['auc_roc'] <= 1.0
    assert 'log_loss' in m and m['log_loss'] >= 0.0
    assert 'brier' in m and 0.0 <= m['brier'] <= 1.0

def test_classification_metrics_proba_only():
    # Proba-only path: no hard predictions
    y_true = [0, 1, 1, 0]
    y_score = [0.1, 0.8, 0.3, 0.2]
    m = classification_metrics(y_true, y_prob=y_score)  # no y_pred
    assert 'accuracy' not in m
    assert 'auc_roc' in m and 0.0 <= m['auc_roc'] <= 1.0
    assert 'auc_pr' in m and 0.0 <= m['auc_pr'] <= 1.0
    assert 'log_loss' in m and m['log_loss'] >= 0.0
    assert 'brier' in m and 0.0 <= m['brier'] <= 1.0

def test_regression_metrics_runs():
    y_true = [3.0, 4.0, 5.0]
    y_pred = [2.9, 4.1, 5.2]
    m = regression_metrics(y_true, y_pred)
    assert 'rmse' in m and m['rmse'] >= 0.0
    # check value roughly matches expected RMSE
    expected_rmse = ((0.1**2 + 0.1**2 + 0.2**2) / 3.0) ** 0.5
    assert math.isclose(m['rmse'], expected_rmse, rel_tol=1e-9)