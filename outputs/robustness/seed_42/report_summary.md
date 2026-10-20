# FinSentry — results summary (auto-generated)

_Generated 2026-10-20T11:43:48+00:00 from saved outputs._

> **PaySim is a synthetic simulation. These results describe models on simulated data and do not establish detection performance at any real bank or mobile-money provider.**

**Sources**

| File | Run id | Experiment id | Time (UTC) | Schema |
|---|---|---|---|---|
| experiment.json | — | exp-7feac00e3b | 2026-10-20T11:38:30+00:00 | 4 |
| metrics.json | ff5e307d217d | exp-7feac00e3b | 2026-10-20T11:38:35+00:00 | 4 |
| backtest.json | c87a3e7b4877 | exp-7feac00e3b | 2026-10-20T11:43:48+00:00 | 4 |

**Consistency check:** passed (same configuration, current schema).

## A. Measured results

Every number in this section is copied from `outputs/metrics.json` or `outputs/backtest.json`. 'undefined' means the metric does not exist for that data (for example recall with no fraud); it is never replaced by zero.

### Key measured facts

- Test period: 150,442 rows, 627 fraud (0.42% prevalence = PR-AUC baseline); train prevalence 0.08%.
- Logistic Regression: PR-AUC 0.153 (36.6× baseline); at the max-F1 threshold precision 0.356, recall 0.100, 114 false alerts, 564 missed fraud, 1.18 alerts per 1,000.
- Logistic Regression alert budget: requested 5.0/1,000 on validation, realised 2.09/1,000 on test; precision 0.263, recall 0.132.
- Logistic Regression: block-bootstrap intervals are 0.9–2.9× the width of row-bootstrap intervals.
- Random Forest: PR-AUC 0.350 (84.1× baseline); at the max-F1 threshold precision 0.649, recall 0.260, 88 false alerts, 464 missed fraud, 1.67 alerts per 1,000.
- Random Forest alert budget: requested 5.0/1,000 on validation, realised 4.55/1,000 on test; precision 0.349, recall 0.381.
- Random Forest: block-bootstrap intervals are 1.1–2.5× the width of row-bootstrap intervals.
- Backtest Logistic Regression: PR-AUC ranged 0.005–0.186 across 5 windows (window baselines 0.0005–0.0058).
- Backtest Random Forest: PR-AUC ranged 0.077–0.359 across 5 windows (window baselines 0.0005–0.0058).

### A1. Single-split experiment

| Item | Value |
|---|---|
| Run id / time (UTC) | ff5e307d217d / 2026-10-20T11:38:35+00:00 |
| Command | `robustness.py --sample-rows 1000000 --seeds 42 123 2026` |
| Data | SAMPLE (uniform) of 1,000,000 rows |
| Rows / fraud used | 1,000,000 / 1,279 |
| Dataset SHA-256 | 16910f90577b0d98… |
| Seed | 42 |
| Split | chronological |
| Final fit requested | train_only |

**Periods**

| Period | Steps | Rows | Fraud | Fraud rate |
|---|---|---|---|---|
| Train | 1–323 | 701,467 | 573 | 0.08% |
| Validation | 324–377 | 148,091 | 79 | 0.05% |
| Test | 378–742 | 150,442 | 627 | 0.42% |

**Final test results (max-F1 threshold for models)**

| Model | Final fit | Threshold | Precision | Recall | F1 | FP | FN | Alerts/1k | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| Rule-based | rule | n/a (fixed rule) | 0.016 | 0.667 | 0.031 | 26120 | 209 | 176.40 | undefined (baseline 0.0042) | undefined |
| Logistic Regression | train_only | 0.998 | 0.356 | 0.100 | 0.157 | 114 | 564 | 1.18 | 0.153 (baseline 0.0042) | 0.932 |
| Random Forest | train_only | 0.767 | 0.649 | 0.260 | 0.371 | 88 | 464 | 1.67 | 0.350 (baseline 0.0042) | 0.934 |

**Threshold strategies** (VALIDATION was used to choose; TEST is the result)

| Model | Strategy | Period | Threshold | Precision | Recall | FP | FN | Alerts/1k |
|---|---|---|---|---|---|---|---|---|
| Rule-based | rule | VALIDATION | n/a (fixed rule) | 0.002 | 0.620 | 29265 | 30 | 197.95 |
| Rule-based | rule | TEST | n/a (fixed rule) | 0.016 | 0.667 | 26120 | 209 | 176.40 |
| Logistic Regression | max_f1 | VALIDATION | 0.998 | 0.018 | 0.114 | 499 | 70 | 3.43 |
| Logistic Regression | max_f1 | TEST | 0.998 | 0.356 | 0.100 | 114 | 564 | 1.18 |
| Logistic Regression | alert_budget | VALIDATION | 0.994 | 0.012 | 0.114 | 732 | 70 | 5.00 |
| Logistic Regression | alert_budget | TEST | 0.994 | 0.263 | 0.132 | 232 | 544 | 2.09 |
| Random Forest | max_f1 | VALIDATION | 0.767 | 0.548 | 0.215 | 14 | 62 | 0.21 |
| Random Forest | max_f1 | TEST | 0.767 | 0.649 | 0.260 | 88 | 464 | 1.67 |
| Random Forest | alert_budget | VALIDATION | 0.577 | 0.040 | 0.380 | 711 | 49 | 5.00 |
| Random Forest | alert_budget | TEST | 0.577 | 0.349 | 0.381 | 446 | 388 | 4.55 |

**95% bootstrap intervals (estimates, not guarantees)**

| Model | Metric | Row bootstrap | Block bootstrap (steps) |
|---|---|---|---|
| Rule-based | rule.precision | 0.015 – 0.017 (n=300) | 0.009 – 0.037 (n=300) |
| Rule-based | rule.recall | 0.628 – 0.701 (n=300) | 0.635 – 0.710 (n=300) |
| Logistic Regression | max_f1.precision | 0.279 – 0.423 (n=300) | 0.238 – 0.540 (n=300) |
| Logistic Regression | max_f1.recall | 0.074 – 0.121 (n=300) | 0.076 – 0.124 (n=300) |
| Logistic Regression | pr_auc | 0.120 – 0.182 (n=300) | 0.105 – 0.223 (n=300) |
| Random Forest | max_f1.precision | 0.590 – 0.700 (n=300) | 0.547 – 0.755 (n=300) |
| Random Forest | max_f1.recall | 0.223 – 0.290 (n=300) | 0.214 – 0.310 (n=300) |
| Random Forest | pr_auc | 0.306 – 0.385 (n=300) | 0.301 – 0.411 (n=300) |

**Calibration and score drift (train-only model; validation → test)**

| Model | Brier raw | Brier calibrated | Brier constant | ECE raw | ECE calibrated | PSI | PSI band |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.12282 | 0.00413 | 0.00416 | 0.2102 | 0.0037 | 0.044 | small |
| Random Forest | 0.01901 | 0.00342 | 0.00416 | 0.0703 | 0.0031 | 0.005 | small |

**Warnings recorded during the run**

- Results come from a sample, not the full dataset.
- Fraud prevalence differs strongly between train and test periods; precision depends heavily on prevalence.

### A2. Temporal backtest

- Data: SAMPLE of 1,000,000 rows; scheme: expanding; windows: 5; run c87a3e7b4877 (2026-10-20T11:43:48+00:00); command `robustness.py --sample-rows 1000000 --seeds 42 123 2026`

| Window | Test steps | Rows | Fraud | Fraud rate (PR-AUC baseline) | LR PR-AUC | RF PR-AUC | RF score PSI | Enough fraud |
|---|---|---|---|---|---|---|---|---|
| 1 | 239–281 | 101,165 | 80 | 0.08% | 0.033 | 0.215 | 0.013 | yes |
| 2 | 282–323 | 100,653 | 66 | 0.07% | 0.006 | 0.089 | 0.006 | yes |
| 3 | 324–354 | 95,345 | 47 | 0.05% | 0.005 | 0.077 | 0.005 | yes |
| 4 | 355–398 | 102,720 | 74 | 0.07% | 0.051 | 0.352 | 0.005 | yes |
| 5 | 399–742 | 100,468 | 585 | 0.58% | 0.186 | 0.359 | 0.009 | yes |

**Across-window variation**

| Model | Metric | Windows | Mean | Std | Min | Max |
|---|---|---|---|---|---|---|
| Logistic Regression | pr_auc | 5 | 0.056 | 0.075 | 0.005 | 0.186 |
| Logistic Regression | pr_auc_lift | 5 | 32.570 | 25.835 | 8.828 | 71.191 |
| Logistic Regression | roc_auc | 5 | 0.928 | 0.009 | 0.914 | 0.935 |
| Logistic Regression | max_f1.precision | 5 | 0.239 | 0.402 | 0.005 | 0.950 |
| Logistic Regression | max_f1.recall | 5 | 0.142 | 0.117 | 0.032 | 0.340 |
| Logistic Regression | max_f1.f1 | 5 | 0.063 | 0.053 | 0.010 | 0.130 |
| Logistic Regression | max_f1.alerts_per_1000 | 5 | 8.809 | 13.677 | 0.199 | 32.209 |
| Logistic Regression | alert_budget.precision | 5 | 0.081 | 0.123 | 0.000 | 0.296 |
| Logistic Regression | alert_budget.recall | 5 | 0.151 | 0.089 | 0.000 | 0.227 |
| Logistic Regression | alert_budget.alerts_per_1000 | 5 | 7.201 | 10.585 | 1.175 | 26.060 |
| Logistic Regression | precision_at_budget | 5 | 0.148 | 0.220 | 0.000 | 0.530 |
| Logistic Regression | brier_raw | 5 | 0.119 | 0.013 | 0.101 | 0.130 |
| Logistic Regression | brier_calibrated | 5 | 0.002 | 0.002 | 0.000 | 0.006 |
| Logistic Regression | ece_raw | 5 | 0.207 | 0.013 | 0.189 | 0.221 |
| Logistic Regression | psi | 5 | 0.099 | 0.071 | 0.005 | 0.186 |
| Random Forest | pr_auc | 5 | 0.218 | 0.136 | 0.077 | 0.359 |
| Random Forest | pr_auc_lift | 5 | 222.606 | 166.384 | 61.725 | 488.082 |
| Random Forest | roc_auc | 5 | 0.914 | 0.025 | 0.880 | 0.937 |
| Random Forest | max_f1.precision | 5 | 0.553 | 0.411 | 0.022 | 1.000 |
| Random Forest | max_f1.recall | 5 | 0.217 | 0.120 | 0.021 | 0.333 |
| Random Forest | max_f1.f1 | 5 | 0.217 | 0.177 | 0.042 | 0.424 |
| Random Forest | max_f1.alerts_per_1000 | 5 | 2.574 | 4.118 | 0.010 | 9.786 |
| Random Forest | alert_budget.precision | 5 | 0.104 | 0.118 | 0.020 | 0.306 |
| Random Forest | alert_budget.recall | 5 | 0.381 | 0.131 | 0.170 | 0.486 |
| Random Forest | alert_budget.alerts_per_1000 | 5 | 7.065 | 5.538 | 2.737 | 16.294 |
| Random Forest | precision_at_budget | 5 | 0.314 | 0.381 | 0.050 | 0.970 |
| Random Forest | brier_raw | 5 | 0.018 | 0.004 | 0.010 | 0.021 |
| Random Forest | brier_calibrated | 5 | 0.002 | 0.002 | 0.000 | 0.005 |
| Random Forest | ece_raw | 5 | 0.067 | 0.011 | 0.050 | 0.074 |
| Random Forest | psi | 5 | 0.008 | 0.003 | 0.005 | 0.013 |

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
