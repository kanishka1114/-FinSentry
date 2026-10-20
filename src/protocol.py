"""Final-fit protocol and threshold transfer to a refitted model."""
import numpy as np

PROTOCOL_STEPS = [
    "1. Split by whole simulation steps: train < validation < test (boundaries recorded).",
    "2. Fit each model on TRAIN only.",
    "3. Choose thresholds (max F1; alert budget) and fit the Platt calibrator on VALIDATION scores "
    "of the train-only model. No test rows are touched.",
    "4. final_fit=train_only (default): the train-only model is final; thresholds need no transfer.",
    "4. final_fit=refit (EXPERIMENTAL, not used for headline results): refit the same settings on "
    "TRAIN+VALIDATION. Thresholds are transferred by matching in-sample alert rates, and the "
    "transfer is accepted only if the train-only model's in-sample and out-of-sample (validation) "
    "alert rates agree within a tolerance. Otherwise the refit is rejected and the train-only model "
    "stays final. This rests on unverifiable assumptions: that the refit model's in-sample optimism "
    "matches the train-only model's, and that the alert rate chosen on validation is still "
    "appropriate after adding newer data. No unseen pre-test data remains to check either, and "
    "calibration of a refit model cannot be assessed, so diagnostics use the train-only model.",
    "5. Score the untouched TEST period once with the final model and thresholds.",
]
TEST_USAGE = ("Test labels and scores are used only for final reporting (metrics, intervals, "
              "calibration/drift diagnostics, preview). They never influence features, training, "
              "threshold selection, calibration or model choice. With the default uniform sampling "
              "no label from any period affects which rows are used; stratified sampling sizes each "
              "class from global label counts and is offered only for development.")


def alert_rate(scores, threshold):
    scores = np.asarray(scores, dtype=float)
    return float((scores >= threshold).mean()) if len(scores) else None


def rate_threshold(scores, rate):
    """Threshold giving roughly `rate` share of alerts on `scores`."""
    scores = np.asarray(scores, dtype=float)
    if rate <= 0:
        return float(np.nextafter(scores.max(), np.inf))
    return float(np.quantile(scores, 1 - min(rate, 1.0), method="higher"))


def transfer_threshold(old_train_scores, old_val_scores, threshold, new_insample_scores,
                       tolerance=2.0):
    """Map a threshold from a train-only model to a refitted model.

    The old model's in-sample alert rate (on its own training rows) is matched on the new
    model's in-sample scores. The match is only trusted when the old model's in-sample and
    out-of-sample (validation) alert rates are within a factor of `tolerance`; otherwise
    in-sample scores are too optimistic to transfer from.
    """
    r_in = alert_rate(old_train_scores, threshold)
    r_out = alert_rate(old_val_scores, threshold)
    info = {"old_threshold": float(threshold), "old_insample_alert_rate": r_in,
            "old_validation_alert_rate": r_out, "tolerance_factor": tolerance}
    if not r_in or not r_out:
        return None, {**info, "valid": False,
                      "reason": "zero alerts in-sample or on validation; rate cannot be matched"}
    ratio = r_in / r_out
    info["insample_vs_validation_ratio"] = ratio
    if not (1 / tolerance <= ratio <= tolerance):
        return None, {**info, "valid": False,
                      "reason": f"in-sample/validation alert-rate ratio {ratio:.2f} outside "
                                f"[{1 / tolerance:.2f}, {tolerance:.2f}]"}
    new_t = rate_threshold(new_insample_scores, r_in)
    return new_t, {**info, "valid": True, "new_threshold": new_t,
                   "new_insample_alert_rate": alert_rate(new_insample_scores, new_t)}
