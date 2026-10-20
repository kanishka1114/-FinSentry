"""Label-free score drift diagnostics."""
import numpy as np

DRIFT_NOTE = (
    "Four different things can change over time: (1) prevalence shift — the fraud rate itself "
    "changes, which moves precision and the PR-AUC baseline even for an unchanged model; "
    "(2) ranking quality — whether fraud still scores above legitimate transactions (PR-AUC "
    "relative to its baseline, ROC-AUC); (3) calibration — whether score values match observed "
    "fraud rates (Brier, reliability); (4) threshold drift — the same threshold produces a "
    "different alert volume because the score distribution moved (PSI, alerts per 1,000). "
    "PSI bands (<0.1 small, 0.1–0.25 moderate, >0.25 large) are an industry heuristic, not a test."
)


def score_summary(s):
    s = np.asarray(s, dtype=float)
    if len(s) == 0:
        return None
    q = np.quantile(s, [0.5, 0.9, 0.99, 0.999])
    return {"rows": int(len(s)), "mean": float(s.mean()), "p50": float(q[0]),
            "p90": float(q[1]), "p99": float(q[2]), "p999": float(q[3]), "max": float(s.max())}


def psi(reference, current, n_bins=10, eps=1e-4):
    """Population stability index with bins from reference quantiles. None if undefined."""
    ref, cur = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    if len(ref) == 0 or len(cur) == 0:
        return None
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return None
    inner = edges[1:-1]
    p = np.bincount(np.searchsorted(inner, ref, side="right"), minlength=len(edges) - 1) / len(ref)
    q = np.bincount(np.searchsorted(inner, cur, side="right"), minlength=len(edges) - 1) / len(cur)
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def psi_band(value):
    if value is None:
        return "undefined"
    return "small" if value < 0.1 else "moderate" if value < 0.25 else "large"


def drift_report(reference_scores, current_scores, reference_label="validation",
                 current_label="test"):
    value = psi(reference_scores, current_scores)
    return {"reference": reference_label, "current": current_label, "psi": value,
            "psi_band": psi_band(value),
            "reference_summary": score_summary(reference_scores),
            "current_summary": score_summary(current_scores)}
