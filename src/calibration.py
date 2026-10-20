"""Probability calibration diagnostics. Calibrators are fit on validation data only."""
import numpy as np
from sklearn.linear_model import LogisticRegression

CALIBRATION_NOTE = (
    "Calibration asks whether a score of 0.2 means about 20% of such transactions are fraud. "
    "Class-weighted models deliberately inflate fraud scores, so raw scores are expected to be "
    "poorly calibrated even when ranking (PR-AUC) is good. A Platt (logistic) calibrator is fit "
    "on validation scores only and applied to the test period; it is monotonic, so it does not "
    "change rankings, PR-AUC or which transactions fall in the top-K. Under prevalence shift a "
    "calibrator fit on one period will be miscalibrated on another. With rare fraud, almost all "
    "rows sit in the lowest equal-width bin, so ECE is dominated by that bin and can look small "
    "while high-score bins are badly miscalibrated; read the reliability bins, not ECE alone. "
    "The 'constant' Brier baseline predicts the validation fraud rate for every row (the only "
    "rate known before the test period), so it is itself handicapped when prevalence shifts."
)
EPS = 1e-6


def brier(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2)) if len(y) else None


def reliability_bins(y, p, n_bins=10, min_rows=30):
    """Equal-width probability bins with observed fraud rate. ECE uses non-empty bins."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if len(y) == 0:
        return [], None
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        mean_pred, observed = float(p[mask].mean()), float(y[mask].mean())
        ece += n / len(y) * abs(observed - mean_pred)
        bins.append({"bin_low": float(edges[b]), "bin_high": float(edges[b + 1]), "rows": n,
                     "fraud": int(y[mask].sum()), "mean_predicted": mean_pred,
                     "observed_rate": observed, "enough_rows": n >= min_rows})
    return bins, float(ece)


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(y_val, s_val, min_fraud=5):
    """Return ({"a", "b"}, None) or (None, reason). Uses validation labels/scores only."""
    y_val = np.asarray(y_val).astype(int)
    n_pos = int(y_val.sum())
    if n_pos < min_fraud or n_pos == len(y_val):
        return None, f"needs >= {min_fraud} validation fraud rows and some legitimate rows (have {n_pos})"
    lr = LogisticRegression(C=1e4, max_iter=1000)
    lr.fit(_logit(s_val).reshape(-1, 1), y_val)
    return {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}, None


def apply_platt(params, s):
    z = params["a"] * _logit(s) + params["b"]
    return 1 / (1 + np.exp(-z))


def calibration_report(y_val, s_val, y_test, s_test, n_bins=10, min_fraud=5, min_rows=30):
    y_val, y_test = np.asarray(y_val).astype(int), np.asarray(y_test).astype(int)
    s_test = np.asarray(s_test, dtype=float)
    warnings = []
    if len(y_test) == 0:
        return {"computed": False, "reason": "empty test period", "note": CALIBRATION_NOTE}
    if y_test.sum() == 0:
        warnings.append("No fraud in test period: Brier score only measures how close scores are "
                        "to zero; reliability of fraud scores cannot be assessed.")
    elif y_test.sum() < min_fraud:
        warnings.append(f"Only {int(y_test.sum())} test fraud rows: calibration estimates are very noisy.")
    bins_raw, ece_raw = reliability_bins(y_test, s_test, n_bins, min_rows)
    val_prev = float(y_val.mean()) if len(y_val) else None
    out = {
        "computed": True, "test_rows": int(len(y_test)), "test_fraud": int(y_test.sum()),
        "brier_raw": brier(y_test, s_test),
        "brier_constant_baseline": brier(y_test, np.full(len(y_test), val_prev))
        if val_prev is not None else None,
        "constant_baseline_value": val_prev,
        "ece_raw": ece_raw, "bins_raw": bins_raw,
        "warnings": warnings, "note": CALIBRATION_NOTE,
    }
    params, reason = fit_platt(y_val, s_val, min_fraud)
    out["calibrator"] = {"method": "platt_on_validation", "params": params, "reason": reason}
    if params:
        cal = apply_platt(params, s_test)
        bins_cal, ece_cal = reliability_bins(y_test, cal, n_bins, min_rows)
        out.update({"brier_calibrated": brier(y_test, cal), "ece_calibrated": ece_cal,
                    "bins_calibrated": bins_cal})
    else:
        out.update({"brier_calibrated": None, "ece_calibrated": None, "bins_calibrated": []})
    return out
