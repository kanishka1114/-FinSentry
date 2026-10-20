"""Global importance and per-transaction feature contributions (no SHAP dependency).

Logistic Regression: exact additive contributions in log-odds space,
    contribution_j = coef_j * (x_j - mean_j over the model's training rows).
Random Forest: tree-path (Saabas) decomposition in probability space. Each split on the
    path from root to leaf credits the change in the node's fraud share to the split feature;
    averaged over trees, bias + contributions equals the predicted score exactly.
Both describe how THIS model arrived at its score; neither is a causal explanation.
"""
import numpy as np
import pandas as pd

EXPLANATION_NOTE = (
    "Contributions show how each model input moved this model's score relative to its "
    "baseline. They describe model behaviour, not causes of fraud, and inputs that are "
    "correlated (amount and log amount; the two hour encodings) share credit arbitrarily. "
    "Logistic Regression contributions are exact in log-odds. Random Forest uses tree-path "
    "(Saabas) contributions: exact in sum but known to over-credit features split near the "
    "leaves compared with TreeSHAP. Only the model inputs are shown; identifiers and balances "
    "are never loaded."
)
INPUT_GROUPS = {"amount": "Amount", "log_amount": "Amount",
                "hour_sin": "Hour of day", "hour_cos": "Hour of day"}


def _clean_names(pre):
    return [n.replace("num__", "").replace("cat__", "") for n in pre.get_feature_names_out()]


def input_group(feature):
    return "Transaction type" if feature.startswith("type_") else INPUT_GROUPS.get(feature, feature)


def global_feature_importance(pipeline):
    """Impurity importance (trees) or |coefficient| on scaled inputs (linear models)."""
    try:
        pre = pipeline.named_steps["preprocess"]
        model = pipeline.named_steps["model"]
        names = _clean_names(pre)
        if hasattr(model, "feature_importances_"):
            frame = pd.DataFrame({"feature": names, "importance": model.feature_importances_})
        elif hasattr(model, "coef_"):
            frame = pd.DataFrame({"feature": names, "importance": abs(model.coef_[0]),
                                  "coefficient": model.coef_[0]})
        else:
            return pd.DataFrame(columns=["feature", "importance"])
        return frame.sort_values("importance", ascending=False).reset_index(drop=True)
    except (AttributeError, KeyError, ValueError):
        return pd.DataFrame(columns=["feature", "importance"])


def _dense(z):
    return np.asarray(z.toarray() if hasattr(z, "toarray") else z, dtype=float)


def transformed_mean(pipeline, X):
    """Mean of the preprocessed training matrix (background for linear contributions)."""
    return _dense(pipeline.named_steps["preprocess"].transform(X)).mean(axis=0)


def explain_logistic(pipeline, X_row, background_mean):
    pre, model = pipeline.named_steps["preprocess"], pipeline.named_steps["model"]
    z = _dense(pre.transform(X_row))[0]
    coef, bg = model.coef_[0], np.asarray(background_mean, dtype=float)
    contrib = coef * (z - bg)
    base = float(model.intercept_[0] + coef @ bg)
    return _package(_clean_names(pre), z, contrib, base, base + contrib.sum(), "log-odds")


def explain_forest(pipeline, X_row):
    pre, model = pipeline.named_steps["preprocess"], pipeline.named_steps["model"]
    z = _dense(pre.transform(X_row))
    contrib, bias = np.zeros(z.shape[1]), 0.0
    for est in model.estimators_:
        tree = est.tree_
        values = tree.value[:, 0, :]
        frac = values[:, 1] / values.sum(axis=1)
        path = est.decision_path(z).indices
        bias += frac[path[0]]
        for parent, child in zip(path[:-1], path[1:]):
            contrib[tree.feature[parent]] += frac[child] - frac[parent]
    n = len(model.estimators_)
    contrib, bias = contrib / n, bias / n
    return _package(_clean_names(pre), z[0], contrib, bias, bias + contrib.sum(), "probability")


def _package(names, values, contrib, base, output, space):
    by_feature = pd.DataFrame({"feature": names, "model_input_value": values,
                               "contribution": contrib})
    by_feature["input"] = by_feature["feature"].map(input_group)
    by_input = (by_feature.groupby("input", as_index=False)["contribution"].sum()
                .assign(abs_c=lambda d: d["contribution"].abs())
                .sort_values("abs_c", ascending=False).drop(columns="abs_c")
                .reset_index(drop=True))
    by_feature = by_feature[by_feature["contribution"].abs() > 0].sort_values(
        "contribution", key=abs, ascending=False).reset_index(drop=True)
    return {"space": space, "base": float(base), "output": float(output),
            "by_input": by_input, "by_feature": by_feature}


def explain_transaction(bundle, X_row):
    """Explain one preprocessed-ready feature row with a saved model bundle."""
    pipeline = bundle["pipeline"]
    model = pipeline.named_steps["model"]
    if hasattr(model, "estimators_"):
        return explain_forest(pipeline, X_row)
    if hasattr(model, "coef_"):
        bg = bundle.get("background_mean")
        if bg is None:
            raise ValueError("Model bundle lacks background_mean; retrain to enable explanations.")
        return explain_logistic(pipeline, X_row, bg)
    raise ValueError("Unsupported model for explanations.")
