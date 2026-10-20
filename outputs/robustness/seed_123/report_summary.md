# FinSentry — results summary (auto-generated)

_Generated 2026-10-20T11:48:40+00:00 from saved outputs._

> **PaySim is a synthetic simulation. These results describe models on simulated data and do not establish detection performance at any real bank or mobile-money provider.**

**Sources**

| File | Run id | Experiment id | Time (UTC) | Schema |
|---|---|---|---|---|
| experiment.json | — | exp-ce67704e29 | 2026-10-20T11:43:48+00:00 | 4 |
| metrics.json | e64db6ba64f4 | exp-ce67704e29 | 2026-10-20T11:43:53+00:00 | 4 |
| backtest.json | 224d1838ad60 | exp-ce67704e29 | 2026-10-20T11:48:40+00:00 | 4 |

**Consistency check:** passed (same configuration, current schema).

## A. Measured results

Every number in this section is copied from `outputs/metrics.json` or `outputs/backtest.json`. 'undefined' means the metric does not exist for that data (for example recall with no fraud); it is never replaced by zero.

### Key measured facts

- Test period: 149,730 rows, 655 fraud (0.44% prevalence = PR-AUC baseline); train prevalence 0.08%.
- Logistic Regression: PR-AUC 0.145 (33.1× baseline); at the max-F1 threshold precision 0.818, recall 0.027, 4 false alerts, 637 missed fraud, 0.15 alerts per 1,000.
- Logistic Regression alert budget: requested 5.0/1,000 on validation, realised 1.92/1,000 on test; precision 0.272, recall 0.119.
- Logistic Regression: block-bootstrap intervals are 1.1–2.7× the width of row-bootstrap intervals.
- Random Forest: PR-AUC 0.348 (79.5× baseline); at the max-F1 threshold precision 0.775, recall 0.200, 38 false alerts, 524 missed fraud, 1.13 alerts per 1,000.
- Random Forest alert budget: requested 5.0/1,000 on validation, realised 4.56/1,000 on test; precision 0.370, recall 0.386.
- Random Forest: block-bootstrap intervals are 1.0–3.0× the width of row-bootstrap intervals.
- Backtest Logistic Regression: PR-AUC ranged 0.005–0.174 across 5 windows (window baselines 0.0005–0.0061).
- Backtest Random Forest: PR-AUC ranged 0.024–0.388 across 5 windows (window baselines 0.0005–0.0061).

### A1. Single-split experiment

| Item | Value |
|---|---|
| Run id / time (UTC) | e64db6ba64f4 / 2026-10-20T11:43:53+00:00 |
| Command | `robustness.py --sample-rows 1000000 --seeds 42 123 2026` |
| Data | SAMPLE (uniform) of 1,000,000 rows |
| Rows / fraud used | 1,000,000 / 1,339 |
| Dataset SHA-256 | 16910f90577b0d98… |
| Seed | 123 |
| Split | chronological |
| Final fit requested | train_only |

**Periods**

| Period | Steps | Rows | Fraud | Fraud rate |
|---|---|---|---|---|
| Train | 1–323 | 702,141 | 588 | 0.08% |
| Validation | 324–377 | 148,129 | 96 | 0.06% |
| Test | 378–743 | 149,730 | 655 | 0.44% |

**Final test results (max-F1 threshold for models)**

| Model | Final fit | Threshold | Precision | Recall | F1 | FP | FN | Alerts/1k | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| Rule-based | rule | n/a (fixed rule) | 0.017 | 0.696 | 0.033 | 26237 | 199 | 178.27 | undefined (baseline 0.0044) | undefined |
| Logistic Regression | train_only | 1.000 | 0.818 | 0.027 | 0.053 | 4 | 637 | 0.15 | 0.145 (baseline 0.0044) | 0.937 |
| Random Forest | train_only | 0.864 | 0.775 | 0.200 | 0.318 | 38 | 524 | 1.13 | 0.348 (baseline 0.0044) | 0.939 |

**Threshold strategies** (VALIDATION was used to choose; TEST is the result)

| Model | Strategy | Period | Threshold | Precision | Recall | FP | FN | Alerts/1k |
|---|---|---|---|---|---|---|---|---|
| Rule-based | rule | VALIDATION | n/a (fixed rule) | 0.002 | 0.740 | 28820 | 25 | 195.04 |
| Rule-based | rule | TEST | n/a (fixed rule) | 0.017 | 0.696 | 26237 | 199 | 178.27 |
| Logistic Regression | max_f1 | VALIDATION | 1.000 | 0.034 | 0.031 | 85 | 93 | 0.59 |
| Logistic Regression | max_f1 | TEST | 1.000 | 0.818 | 0.027 | 4 | 637 | 0.15 |
| Logistic Regression | alert_budget | VALIDATION | 0.996 | 0.012 | 0.094 | 732 | 87 | 5.00 |
| Logistic Regression | alert_budget | TEST | 0.996 | 0.272 | 0.119 | 209 | 577 | 1.92 |
| Random Forest | max_f1 | VALIDATION | 0.864 | 0.407 | 0.229 | 32 | 74 | 0.36 |
| Random Forest | max_f1 | TEST | 0.864 | 0.775 | 0.200 | 38 | 524 | 1.13 |
| Random Forest | alert_budget | VALIDATION | 0.658 | 0.051 | 0.396 | 703 | 58 | 5.00 |
| Random Forest | alert_budget | TEST | 0.658 | 0.370 | 0.386 | 430 | 402 | 4.56 |

**95% bootstrap intervals (estimates, not guarantees)**

| Model | Metric | Row bootstrap | Block bootstrap (steps) |
|---|---|---|---|
| Rule-based | rule.precision | 0.016 – 0.018 (n=300) | 0.011 – 0.041 (n=300) |
| Rule-based | rule.recall | 0.660 – 0.734 (n=300) | 0.654 – 0.735 (n=300) |
| Logistic Regression | max_f1.precision | 0.667 – 0.952 (n=300) | 0.643 – 0.951 (n=300) |
| Logistic Regression | max_f1.recall | 0.017 – 0.040 (n=300) | 0.014 – 0.043 (n=300) |
| Logistic Regression | pr_auc | 0.122 – 0.173 (n=300) | 0.101 – 0.220 (n=300) |
| Random Forest | max_f1.precision | 0.706 – 0.838 (n=300) | 0.653 – 0.910 (n=300) |
| Random Forest | max_f1.recall | 0.169 – 0.231 (n=300) | 0.166 – 0.227 (n=300) |
| Random Forest | pr_auc | 0.307 – 0.382 (n=300) | 0.290 – 0.423 (n=300) |

**Calibration and score drift (train-only model; validation → test)**

| Model | Brier raw | Brier calibrated | Brier constant | ECE raw | ECE calibrated | PSI | PSI band |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.11808 | 0.00433 | 0.00437 | 0.2055 | 0.0038 | 0.056 | small |
| Random Forest | 0.01944 | 0.00377 | 0.00437 | 0.0720 | 0.0034 | 0.001 | small |

**Warnings recorded during the run**

- Results come from a sample, not the full dataset.
- Fraud prevalence differs strongly between train and test periods; precision depends heavily on prevalence.

### A2. Temporal backtest

- Data: SAMPLE of 1,000,000 rows; scheme: expanding; windows: 5; run 224d1838ad60 (2026-10-20T11:48:40+00:00); command `robustness.py --sample-rows 1000000 --seeds 42 123 2026`

| Window | Test steps | Rows | Fraud | Fraud rate (PR-AUC baseline) | LR PR-AUC | RF PR-AUC | RF score PSI | Enough fraud |
|---|---|---|---|---|---|---|---|---|
| 1 | 239–281 | 101,565 | 70 | 0.07% | 0.034 | 0.087 | 0.002 | yes |
| 2 | 282–323 | 100,477 | 67 | 0.07% | 0.006 | 0.085 | 0.003 | yes |
| 3 | 324–354 | 95,052 | 52 | 0.05% | 0.005 | 0.024 | 0.006 | yes |
| 4 | 355–398 | 102,976 | 89 | 0.09% | 0.046 | 0.337 | 0.004 | yes |
| 5 | 399–743 | 99,831 | 610 | 0.61% | 0.174 | 0.388 | 0.008 | yes |

**Across-window variation**

| Model | Metric | Windows | Mean | Std | Min | Max |
|---|---|---|---|---|---|---|
| Logistic Regression | pr_auc | 5 | 0.053 | 0.070 | 0.005 | 0.174 |
| Logistic Regression | pr_auc_lift | 5 | 29.780 | 21.255 | 8.584 | 53.065 |
| Logistic Regression | roc_auc | 5 | 0.930 | 0.018 | 0.903 | 0.950 |
| Logistic Regression | max_f1.precision | 5 | 0.151 | 0.278 | 0.005 | 0.646 |
| Logistic Regression | max_f1.recall | 5 | 0.200 | 0.234 | 0.057 | 0.607 |
| Logistic Regression | max_f1.f1 | 5 | 0.050 | 0.048 | 0.009 | 0.124 |
| Logistic Regression | max_f1.alerts_per_1000 | 5 | 11.178 | 11.088 | 0.512 | 25.190 |
| Logistic Regression | alert_budget.precision | 5 | 0.080 | 0.111 | 0.000 | 0.272 |
| Logistic Regression | alert_budget.recall | 5 | 0.167 | 0.114 | 0.000 | 0.299 |
| Logistic Regression | alert_budget.alerts_per_1000 | 5 | 6.893 | 9.801 | 1.441 | 24.344 |
| Logistic Regression | precision_at_budget | 5 | 0.144 | 0.226 | 0.000 | 0.540 |
| Logistic Regression | brier_raw | 5 | 0.115 | 0.012 | 0.101 | 0.128 |
| Logistic Regression | brier_calibrated | 5 | 0.002 | 0.002 | 0.001 | 0.006 |
| Logistic Regression | ece_raw | 5 | 0.203 | 0.012 | 0.190 | 0.215 |
| Logistic Regression | psi | 5 | 0.088 | 0.068 | 0.004 | 0.175 |
| Random Forest | pr_auc | 5 | 0.184 | 0.166 | 0.024 | 0.388 |
| Random Forest | pr_auc_lift | 5 | 150.357 | 138.776 | 44.756 | 389.629 |
| Random Forest | roc_auc | 5 | 0.926 | 0.009 | 0.917 | 0.940 |
| Random Forest | max_f1.precision | 5 | 0.333 | 0.392 | 0.000 | 0.893 |
| Random Forest | max_f1.recall | 5 | 0.163 | 0.108 | 0.000 | 0.270 |
| Random Forest | max_f1.f1 | 5 | 0.170 | 0.167 | 0.000 | 0.369 |
| Random Forest | max_f1.alerts_per_1000 | 5 | 2.606 | 4.632 | 0.011 | 10.848 |
| Random Forest | alert_budget.precision | 5 | 0.109 | 0.128 | 0.018 | 0.327 |
| Random Forest | alert_budget.recall | 5 | 0.376 | 0.178 | 0.058 | 0.472 |
| Random Forest | alert_budget.alerts_per_1000 | 5 | 7.027 | 5.910 | 1.452 | 16.521 |
| Random Forest | precision_at_budget | 5 | 0.294 | 0.386 | 0.010 | 0.950 |
| Random Forest | brier_raw | 5 | 0.019 | 0.004 | 0.013 | 0.023 |
| Random Forest | brier_calibrated | 5 | 0.002 | 0.002 | 0.001 | 0.005 |
| Random Forest | ece_raw | 5 | 0.071 | 0.006 | 0.060 | 0.076 |
| Random Forest | psi | 5 | 0.004 | 0.002 | 0.002 | 0.008 |

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
