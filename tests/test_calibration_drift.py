import numpy as np
from src.calibration import brier, reliability_bins, fit_platt, apply_platt, calibration_report
from src.drift import psi, psi_band, drift_report, score_summary


def inflated(seed=0, n=20_000):
    """True probability p, but the model reports sqrt(p) (over-confident fraud scores)."""
    rng = np.random.default_rng(seed)
    p = rng.beta(0.3, 20, n)
    y = (rng.random(n) < p).astype(int)
    return y, np.sqrt(p)


def test_brier_known_values():
    assert brier([0, 1], [0.0, 1.0]) == 0.0
    assert brier([0, 1], [0.5, 0.5]) == 0.25
    assert brier([], []) is None


def test_reliability_bins_counts_and_ece():
    y = np.array([0, 1] * 50)
    bins, ece = reliability_bins(y, np.full(100, 0.5), n_bins=10)
    assert len(bins) == 1 and bins[0]["rows"] == 100 and abs(ece) < 1e-12
    y, s = inflated()
    bins, ece = reliability_bins(y, s)
    assert sum(b["rows"] for b in bins) == len(y) and ece > 0.05
    assert all("enough_rows" in b for b in bins)


def test_platt_improves_brier_and_uses_validation_only():
    y_val, s_val = inflated(1)
    y_test, s_test = inflated(2)
    params, reason = fit_platt(y_val, s_val)
    assert reason is None
    assert brier(y_test, apply_platt(params, s_test)) < brier(y_test, s_test)
    a = calibration_report(y_val, s_val, y_test, s_test)
    b = calibration_report(y_val, s_val, 1 - y_test, s_test)   # test labels changed
    assert a["calibrator"]["params"] == b["calibrator"]["params"]
    assert a["brier_calibrated"] < a["brier_raw"]


def test_calibration_edge_cases():
    s = np.linspace(0, 1, 100)
    params, reason = fit_platt(np.zeros(100, dtype=int), s)
    assert params is None and "validation fraud" in reason
    rep = calibration_report(np.zeros(100, dtype=int), s, np.zeros(50, dtype=int), s[:50])
    assert rep["computed"] and rep["brier_calibrated"] is None
    assert any("No fraud" in w for w in rep["warnings"])
    empty = calibration_report(np.array([0, 1]), np.array([0.1, 0.9]), np.array([]), np.array([]))
    assert empty["computed"] is False


def test_psi_behaviour():
    rng = np.random.default_rng(0)
    a = rng.random(10_000)
    assert psi(a, a) < 1e-9
    assert psi(a, rng.random(10_000)) < 0.02
    assert psi(a, np.clip(a + 0.4, 0, 1)) > 0.25
    assert psi(np.full(100, 0.3), a) is None        # constant reference: undefined, not 0
    assert psi([], a) is None and psi(a, []) is None
    assert psi_band(None) == "undefined" and psi_band(0.05) == "small" and psi_band(0.3) == "large"


def test_drift_report_and_summary_edge_cases():
    rep = drift_report(np.array([0.1, 0.2, 0.3]), np.array([]))
    assert rep["psi"] is None and rep["current_summary"] is None
    assert score_summary([0.5])["p99"] == 0.5
