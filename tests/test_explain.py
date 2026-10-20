import numpy as np
import pandas as pd
import pytest
from src.data_loader import load_data, prepare_features, build_features
from src.explain import (explain_transaction, explain_logistic, explain_forest, transformed_mean,
                         global_feature_importance)
from src.train import build_models

FAST_RF = {"n_estimators": 15, "max_depth": 6, "min_samples_leaf": 5, "max_samples": 0.5, "n_jobs": 1}
FORBIDDEN = ("name", "balance", "isFlagged", "step")


@pytest.fixture
def fitted(paysim_csv):
    df, _ = load_data(paysim_csv)
    X, y = prepare_features(df)
    models = build_models(0, FAST_RF, {"max_iter": 1000, "C": 1.0})
    for pipe in models.values():
        pipe.fit(X, y)
    return models, X


def test_logistic_contributions_are_exact_in_log_odds(fitted):
    models, X = fitted
    pipe = models["Logistic Regression"]
    bg = transformed_mean(pipe, X)
    for i in (0, 5, 123):
        row = X.iloc[[i]]
        ex = explain_logistic(pipe, row, bg)
        assert ex["space"] == "log-odds"
        assert ex["output"] == pytest.approx(pipe.decision_function(row)[0], abs=1e-8)
        assert ex["base"] + ex["by_feature"]["contribution"].sum() == pytest.approx(ex["output"])


def test_forest_contributions_sum_to_predicted_probability(fitted):
    models, X = fitted
    pipe = models["Random Forest"]
    for i in (1, 42, 999):
        row = X.iloc[[i]]
        ex = explain_forest(pipe, row)
        assert ex["space"] == "probability"
        assert ex["output"] == pytest.approx(pipe.predict_proba(row)[0, 1], abs=1e-9)
        assert ex["by_input"]["contribution"].sum() + ex["base"] == pytest.approx(ex["output"])


def test_explanations_only_mention_model_inputs(fitted):
    models, X = fitted
    raw = pd.DataFrame([{"step": 300, "type": "TRANSFER", "amount": 250_000.0}])
    feats = build_features(raw)
    bundles = {"Logistic Regression": {"pipeline": models["Logistic Regression"],
                                       "background_mean": transformed_mean(models["Logistic Regression"], X)},
               "Random Forest": {"pipeline": models["Random Forest"]}}
    for bundle in bundles.values():
        ex = explain_transaction(bundle, feats)
        assert set(ex["by_input"]["input"]) <= {"Amount", "Hour of day", "Transaction type"}
        for name in ex["by_feature"]["feature"]:
            assert not any(bad in name for bad in FORBIDDEN)


def test_logistic_bundle_without_background_raises(fitted):
    models, X = fitted
    with pytest.raises(ValueError, match="background_mean"):
        explain_transaction({"pipeline": models["Logistic Regression"]}, X.iloc[[0]])


def test_global_importance_still_works(fitted):
    models, _ = fitted
    for pipe in models.values():
        imp = global_feature_importance(pipe)
        assert not imp.empty and np.isfinite(imp["importance"]).all()
