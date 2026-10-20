# FinSentry — results summary (auto-generated)

_Generated 2026-10-20T11:53:34+00:00 from saved outputs._

> **PaySim is a synthetic simulation. These results describe models on simulated data and do not establish detection performance at any real bank or mobile-money provider.**

**Sources**

| File | Run id | Experiment id | Time (UTC) | Schema |
|---|---|---|---|---|
| experiment.json | — | exp-4ff3a723eb | 2026-10-20T11:48:40+00:00 | 4 |
| metrics.json | 87da10b3876c | exp-4ff3a723eb | 2026-10-20T11:48:45+00:00 | 4 |
| backtest.json | 9e4723665621 | exp-4ff3a723eb | 2026-10-20T11:53:34+00:00 | 4 |

**Consistency check:** passed (same configuration, current schema).

## A. Measured results

Every number in this section is copied from `outputs/metrics.json` or `outputs/backtest.json`. 'undefined' means the metric does not exist for that data (for example recall with no fraud); it is never replaced by zero.

### Key measured facts

- Test period: 149,629 rows, 627 fraud (0.42% prevalence = PR-AUC baseline); train prevalence 0.08%.
- Logistic Regression: PR-AUC 0.140 (33.4× baseline); at the max-F1 threshold precision 0.585, recall 0.061, 27 false alerts, 589 missed fraud, 0.43 alerts per 1,000.
- Logistic Regression alert budget: requested 5.0/1,000 on validation, realised 1.72/1,000 on test; precision 0.271, recall 0.112.
- Logistic Regression: block-bootstrap intervals are 1.1–2.6× the width of row-bootstrap intervals.
- Random Forest: PR-AUC 0.343 (81.9× baseline); at the max-F1 threshold precision 0.564, recall 0.301, 146 false alerts, 438 missed fraud, 2.24 alerts per 1,000.
- Random Forest alert budget: requested 5.0/1,000 on validation, realised 6.73/1,000 on test; precision 0.250, recall 0.402.
- Random Forest: block-bootstrap intervals are 1.2–3.3× the width of row-bootstrap intervals.
- Backtest Logistic Regression: PR-AUC ranged 0.004–0.171 across 5 windows (window baselines 0.0005–0.0059).
- Backtest Random Forest: PR-AUC ranged 0.037–0.368 across 5 windows (window baselines 0.0005–0.0059).

### A1. Single-split experiment

| Item | Value |
|---|---|
| Run id / time (UTC) | 87da10b3876c / 2026-10-20T11:48:45+00:00 |
| Command | `robustness.py --sample-rows 1000000 --seeds 42 123 2026` |
| Data | SAMPLE (uniform) of 1,000,000 rows |
| Rows / fraud used | 1,000,000 / 1,271 |
| Dataset SHA-256 | 16910f90577b0d98… |
| Seed | 2026 |
| Split | chronological |
| Final fit requested | train_only |

**Periods**

| Period | Steps | Rows | Fraud | Fraud rate |
|---|---|---|---|---|
| Train | 1–323 | 702,174 | 561 | 0.08% |
| Validation | 324–377 | 148,197 | 83 | 0.06% |
| Test | 378–742 | 149,629 | 627 | 0.42% |

**Final test results (max-F1 threshold for models)**

| Model | Final fit | Threshold | Precision | Recall | F1 | FP | FN | Alerts/1k | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| Rule-based | rule | n/a (fixed rule) | 0.016 | 0.681 | 0.031 | 26168 | 200 | 177.74 | undefined (baseline 0.0042) | undefined |
| Logistic Regression | train_only | 1.000 | 0.585 | 0.061 | 0.110 | 27 | 589 | 0.43 | 0.140 (baseline 0.0042) | 0.926 |
| Random Forest | train_only | 0.770 | 0.564 | 0.301 | 0.393 | 146 | 438 | 2.24 | 0.343 (baseline 0.0042) | 0.928 |

**Threshold strategies** (VALIDATION was used to choose; TEST is the result)

| Model | Strategy | Period | Threshold | Precision | Recall | FP | FN | Alerts/1k |
|---|---|---|---|---|---|---|---|---|
| Rule-based | rule | VALIDATION | n/a (fixed rule) | 0.002 | 0.735 | 28851 | 22 | 195.09 |
| Rule-based | rule | TEST | n/a (fixed rule) | 0.016 | 0.681 | 26168 | 200 | 177.74 |
| Logistic Regression | max_f1 | VALIDATION | 1.000 | 0.025 | 0.096 | 308 | 75 | 2.13 |
| Logistic Regression | max_f1 | TEST | 1.000 | 0.585 | 0.061 | 27 | 589 | 0.43 |
| Logistic Regression | alert_budget | VALIDATION | 0.995 | 0.016 | 0.145 | 729 | 71 | 5.00 |
| Logistic Regression | alert_budget | TEST | 0.995 | 0.271 | 0.112 | 188 | 557 | 1.72 |
| Random Forest | max_f1 | VALIDATION | 0.770 | 0.660 | 0.398 | 17 | 50 | 0.34 |
| Random Forest | max_f1 | TEST | 0.770 | 0.564 | 0.301 | 146 | 438 | 2.24 |
| Random Forest | alert_budget | VALIDATION | 0.601 | 0.053 | 0.470 | 702 | 44 | 5.00 |
| Random Forest | alert_budget | TEST | 0.601 | 0.250 | 0.402 | 755 | 375 | 6.73 |

**95% bootstrap intervals (estimates, not guarantees)**

| Model | Metric | Row bootstrap | Block bootstrap (steps) |
|---|---|---|---|
| Rule-based | rule.precision | 0.015 – 0.017 (n=300) | 0.010 – 0.038 (n=300) |
| Rule-based | rule.recall | 0.641 – 0.715 (n=300) | 0.643 – 0.724 (n=300) |
| Logistic Regression | max_f1.precision | 0.467 – 0.692 (n=300) | 0.391 – 0.788 (n=300) |
| Logistic Regression | max_f1.recall | 0.042 – 0.083 (n=300) | 0.038 – 0.081 (n=300) |
| Logistic Regression | pr_auc | 0.117 – 0.172 (n=300) | 0.097 – 0.224 (n=300) |
| Random Forest | max_f1.precision | 0.514 – 0.615 (n=300) | 0.467 – 0.692 (n=300) |
| Random Forest | max_f1.recall | 0.266 – 0.333 (n=300) | 0.258 – 0.361 (n=300) |
| Random Forest | pr_auc | 0.305 – 0.377 (n=300) | 0.290 – 0.417 (n=300) |

**Calibration and score drift (train-only model; validation → test)**

| Model | Brier raw | Brier calibrated | Brier constant | ECE raw | ECE calibrated | PSI | PSI band |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.11601 | 0.00415 | 0.00419 | 0.2022 | 0.0037 | 0.059 | small |
| Random Forest | 0.01989 | 0.00335 | 0.00419 | 0.0726 | 0.0028 | 0.005 | small |

**Warnings recorded during the run**

- Results come from a sample, not the full dataset.
- Fraud prevalence differs strongly between train and test periods; precision depends heavily on prevalence.

### A2. Temporal backtest

- Data: SAMPLE of 1,000,000 rows; scheme: expanding; windows: 5; run 9e4723665621 (2026-10-20T11:53:34+00:00); command `robustness.py --sample-rows 1000000 --seeds 42 123 2026`

| Window | Test steps | Rows | Fraud | Fraud rate (PR-AUC baseline) | LR PR-AUC | RF PR-AUC | RF score PSI | Enough fraud |
|---|---|---|---|---|---|---|---|---|
| 1 | 239–281 | 101,028 | 65 | 0.06% | 0.031 | 0.099 | 0.008 | yes |
| 2 | 282–323 | 100,998 | 68 | 0.07% | 0.007 | 0.060 | 0.004 | yes |
| 3 | 324–354 | 95,116 | 43 | 0.05% | 0.004 | 0.037 | 0.010 | yes |
| 4 | 355–398 | 102,968 | 81 | 0.08% | 0.052 | 0.368 | 0.005 | yes |
| 5 | 399–742 | 99,742 | 586 | 0.59% | 0.171 | 0.364 | 0.004 | yes |

**Across-window variation**

| Model | Metric | Windows | Mean | Std | Min | Max |
|---|---|---|---|---|---|---|
| Logistic Regression | pr_auc | 5 | 0.053 | 0.069 | 0.004 | 0.171 |
| Logistic Regression | pr_auc_lift | 5 | 32.535 | 24.280 | 9.208 | 65.563 |
| Logistic Regression | roc_auc | 5 | 0.929 | 0.013 | 0.907 | 0.942 |
| Logistic Regression | max_f1.precision | 5 | 0.147 | 0.255 | 0.006 | 0.600 |
| Logistic Regression | max_f1.recall | 5 | 0.194 | 0.154 | 0.056 | 0.395 |
| Logistic Regression | max_f1.f1 | 5 | 0.057 | 0.043 | 0.011 | 0.103 |
| Logistic Regression | max_f1.alerts_per_1000 | 5 | 9.402 | 10.541 | 0.551 | 26.326 |
| Logistic Regression | alert_budget.precision | 5 | 0.081 | 0.119 | 0.000 | 0.288 |
| Logistic Regression | alert_budget.recall | 5 | 0.171 | 0.121 | 0.000 | 0.338 |
| Logistic Regression | alert_budget.alerts_per_1000 | 5 | 6.833 | 10.001 | 1.241 | 24.654 |
| Logistic Regression | precision_at_budget | 5 | 0.130 | 0.186 | 0.000 | 0.450 |
| Logistic Regression | brier_raw | 5 | 0.112 | 0.012 | 0.097 | 0.124 |
| Logistic Regression | brier_calibrated | 5 | 0.002 | 0.002 | 0.000 | 0.006 |
| Logistic Regression | ece_raw | 5 | 0.198 | 0.012 | 0.184 | 0.210 |
| Logistic Regression | psi | 5 | 0.085 | 0.070 | 0.003 | 0.173 |
| Random Forest | pr_auc | 5 | 0.186 | 0.166 | 0.037 | 0.368 |
| Random Forest | pr_auc_lift | 5 | 170.754 | 169.884 | 61.951 | 468.376 |
| Random Forest | roc_auc | 5 | 0.930 | 0.019 | 0.899 | 0.952 |
| Random Forest | max_f1.precision | 5 | 0.514 | 0.425 | 0.014 | 1.000 |
| Random Forest | max_f1.recall | 5 | 0.161 | 0.116 | 0.023 | 0.311 |
| Random Forest | max_f1.f1 | 5 | 0.196 | 0.183 | 0.025 | 0.403 |
| Random Forest | max_f1.alerts_per_1000 | 5 | 1.902 | 2.448 | 0.011 | 5.624 |
| Random Forest | alert_budget.precision | 5 | 0.095 | 0.110 | 0.021 | 0.284 |
| Random Forest | alert_budget.recall | 5 | 0.410 | 0.142 | 0.163 | 0.519 |
| Random Forest | alert_budget.alerts_per_1000 | 5 | 7.384 | 4.532 | 2.849 | 14.317 |
| Random Forest | precision_at_budget | 5 | 0.308 | 0.384 | 0.020 | 0.960 |
| Random Forest | brier_raw | 5 | 0.019 | 0.004 | 0.013 | 0.023 |
| Random Forest | brier_calibrated | 5 | 0.002 | 0.002 | 0.000 | 0.005 |
| Random Forest | ece_raw | 5 | 0.071 | 0.007 | 0.060 | 0.077 |
| Random Forest | psi | 5 | 0.006 | 0.003 | 0.004 | 0.010 |

**Backtest warnings**

- Backtest uses a sample, not the full dataset.

## B. Interpretation guide (explanatory text, not measurements)

**Evaluation protocol**

- 1. Split by whole simulation steps: train < validation < test (boundaries recorded).
- 2. Fit each model on TRAIN only.
- 3. Choose thresholds (max F1; alert budget) and fit the Platt calibrator on VALIDATION scores of the train-only model. No test rows are touched.
- 4. final_fit=train_only (default): the train-only model is final; thresholds need no transfer.
- 4. final_fit=refit (EXPERIMENTAL, not used for headline results): refit the same settings on TRAIN+VALIDATION. Thresholds are transferred by matching in-sample alert rates, and the transfer is accepted only if the train-only model's in-sample and out-of-sample (validation) alert rates agree within a tolerance. Otherwise the refit is rejected and the train-only model stays final. This rests on unverifiable assumptions: that the refit model's in-sample optimism matches the train-only model's, and that the alert rate chosen on validation is still appropriate after adding newer data. No unseen pre-test data remains to check either, and calibration of a refit model cannot be assessed, so diagnostics use the train-only model.
- 5. Score the untouched TEST period once with the final model and thresholds.

- Test labels and scores are used only for final reporting (metrics, intervals, calibration/drift diagnostics, preview). They never influence features, training, threshold selection, calibration or model choice. With the default uniform sampling no label from any period affects which rows are used; stratified sampling sizes each class from global label counts and is offered only for development.

**Thresholds.** No threshold is universally best. 'Max F1' weights missed fraud and false alerts equally; an alert budget fixes reviewer workload instead. The right choice depends on the relative cost of a missed fraud versus a false alert and on review capacity. All thresholds are chosen on validation data; test labels are never used for tuning.

**Row bootstrap.** Stratified row bootstrap of the test period: fraud and legitimate rows are resampled separately so every replicate keeps the observed fraud count. It reflects sampling noise in which test rows were observed, for a fixed trained model and threshold. It does NOT capture time dependence, drift, threshold-selection or retraining variability; see the temporal backtest for that.

**Block bootstrap.** Moving-block bootstrap over simulation steps: contiguous blocks of whole steps are resampled, so transactions from the same hour (and neighbouring hours) stay together. This keeps short-range time dependence that the row bootstrap destroys, and the fraud count varies between replicates. Like the row bootstrap it is conditional on the fitted model and the already-chosen thresholds. Intervals are estimates, not guarantees: they assume the test period is roughly stationary, cannot reflect drift beyond the test period, and become unreliable with few blocks or few fraud cases (some replicates may contain no fraud, making recall/PR-AUC undefined for them; see n_valid).

**Calibration.** Calibration asks whether a score of 0.2 means about 20% of such transactions are fraud. Class-weighted models deliberately inflate fraud scores, so raw scores are expected to be poorly calibrated even when ranking (PR-AUC) is good. A Platt (logistic) calibrator is fit on validation scores only and applied to the test period; it is monotonic, so it does not change rankings, PR-AUC or which transactions fall in the top-K. Under prevalence shift a calibrator fit on one period will be miscalibrated on another. With rare fraud, almost all rows sit in the lowest equal-width bin, so ECE is dominated by that bin and can look small while high-score bins are badly miscalibrated; read the reliability bins, not ECE alone. The 'constant' Brier baseline predicts the validation fraud rate for every row (the only rate known before the test period), so it is itself handicapped when prevalence shifts.

**Drift.** Four different things can change over time: (1) prevalence shift — the fraud rate itself changes, which moves precision and the PR-AUC baseline even for an unchanged model; (2) ranking quality — whether fraud still scores above legitimate transactions (PR-AUC relative to its baseline, ROC-AUC); (3) calibration — whether score values match observed fraud rates (Brier, reliability); (4) threshold drift — the same threshold produces a different alert volume because the score distribution moved (PSI, alerts per 1,000). PSI bands (<0.1 small, 0.1–0.25 moderate, >0.25 large) are an industry heuristic, not a test.

**Explanations.** Contributions show how each model input moved this model's score relative to its baseline. They describe model behaviour, not causes of fraud, and inputs that are correlated (amount and log amount; the two hour encodings) share credit arbitrarily. Logistic Regression contributions are exact in log-odds. Random Forest uses tree-path (Saabas) contributions: exact in sum but known to over-credit features split near the leaves compared with TreeSHAP. Only the model inputs are shown; identifiers and balances are never loaded.

**Limitations**

- PaySim is synthetic; results are not evidence of real-world fraud-detection performance and do not establish deployment performance at a real bank.
- Intervals are conditional on the fitted model and the chosen thresholds; they omit threshold-selection and retraining uncertainty.
- With rare fraud, ECE is dominated by the lowest score bin; reliability bins are more informative.
- The optional refit protocol transfers thresholds under unverifiable assumptions and is excluded from headline results.
- Sample runs contain few fraud cases; many metrics have wide intervals.
- Bootstrap intervals are estimates under a stationarity assumption, not guarantees. The row bootstrap ignores time dependence; the block bootstrap keeps short-range dependence but cannot represent drift outside the test period.
- Thresholds chosen on validation may not transfer when prevalence or score distributions shift.
- Feature contributions describe model behaviour, not causes of fraud.
- Only amount, transaction type and hour of day are model inputs; balances and identifiers are excluded by design, which limits achievable performance.

## C. Reproduce

```bash
robustness.py --sample-rows 1000000 --seeds 42 123 2026
streamlit run app.py
```

Library versions: python 3.13.3, numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, joblib 1.6.0, streamlit 1.66.0, plotly 7.1.0
