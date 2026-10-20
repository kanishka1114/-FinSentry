import numpy as np
from src.metrics import point_metrics, ranking_metrics, tune_threshold


def test_point_metrics_confusion_counts():
    m = point_metrics([0, 0, 1, 1], [0, 1, 1, 0])
    assert (m["true_negative"], m["false_positive"], m["false_negative"], m["true_positive"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["false_alerts"] == 1


def test_ranking_metrics_budget_and_baseline():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    m = ranking_metrics(y, score, alert_budget=2)
    assert m["precision_at_budget"] == 0.5 and m["recall_at_budget"] == 0.5
    assert m["pr_auc_baseline"] == 0.2


def test_ranking_metrics_undefined_without_fraud():
    m = ranking_metrics(np.zeros(5, dtype=int), np.linspace(0, 1, 5), alert_budget=100)
    assert m["pr_auc"] is None and m["roc_auc"] is None and m["recall_at_budget"] is None
    assert m["alert_budget"] == 5


def test_tune_threshold_uses_validation_and_falls_back():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    threshold, source, f1 = tune_threshold(y, s)
    assert 0.3 < threshold <= 0.8 and f1 == 1.0 and source == "max_f1_on_validation"
    assert tune_threshold(np.zeros(3, dtype=int), s[:3])[0] == 0.5
