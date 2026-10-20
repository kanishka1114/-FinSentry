# FinSentry: Explainable Machine Learning for Mobile-Money Fraud Monitoring

## Abstract
This project compares a rule-based baseline, Logistic Regression and Random Forest on synthetic PaySim transactions. It evaluates precision, recall, F1, PR-AUC and false-alert counts and presents model outputs in Streamlit. Synthetic results are not evidence of real-world fraud detection performance.

## 1. Introduction
Digital payment providers need ways to prioritize transactions for further review. Imbalanced labels, leakage, false positives and explainability are important concerns.

## 2. Problem Statement
Build an educational prototype that flags transactions labeled fraudulent in PaySim and explains model behavior. It is not a production decision system.

## 3. Dataset
Source: https://www.kaggle.com/datasets/ealaxi/paysim1  
License on dataset listing: CC BY-SA 4.0.  
Citation: Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016), "PaySim: A financial mobile money simulator for fraud detection," EMSS.

Record the exact CSV version, rows used, fraud prevalence, transaction types, and whether full data or a sample was used.

## 4. Methodology
Describe cleaning, feature engineering, split, model settings and threshold. Only `step`, `type`, `amount` and `isFraud` are loaded; identifiers, balance fields and `isFlaggedFraud` are excluded to reduce memorization and shortcut leakage. Raw `step` is used only for the chronological train/validation/test split; models see hour-of-day encodings instead. Thresholds are tuned on validation data (max F1) and the test period is evaluated once. Copy sampling settings, split boundaries, per-period prevalence, seed and library versions from `outputs/metrics.json`.

## 5. Models
1. Rule-based heuristic baseline
2. Class-weighted Logistic Regression
3. Class-weighted Random Forest

## 6. Evaluation
Report precision, recall, F1, PR-AUC, ROC-AUC, confusion matrix counts and false alerts. Explain why accuracy alone is misleading under severe class imbalance.

Reproduce with the recommended experiment. It writes `outputs/experiment.json` (configuration, data fingerprint, run ids, versions), `metrics.json`, `backtest.json` and `report_summary.md`, all sharing one experiment id:

```bash
python -m src.experiment --sample-rows 1000000   # uniform sample, seed 42, chronological split + 5-window backtest
```

Copy numbers only from a report whose consistency check passed. State that the data was a uniform 1M-row sample of synthetic PaySim and that full-data results were not measured.

Describe both threshold strategies (max F1 on validation; alert budget of N per 1,000 on validation), and state that test labels were never used for tuning. Explain that the preferred threshold depends on the cost of a missed fraud relative to a false alert and on review capacity.

**Protocol.** State the final-fit mode (`train_only` or `refit`), whether any refit was rejected and why (`models.<name>.final_fit`), and the exact step boundaries (`protocol.step_boundaries`). Mention the leakage test that flips all test labels.

**Uncertainty.** Report both the stratified row bootstrap and the moving-block bootstrap over steps (block length, number of blocks, replicates, seed). Explain that the row bootstrap ignores time dependence, that the block bootstrap keeps short-range dependence but assumes stationarity within the test period, and that both are estimates, not guarantees. Note replicates without fraud (`n_valid`).

**Calibration and drift.** Report Brier (raw, Platt-calibrated, constant baseline), ECE and the reliability diagram; the calibrator was fit on validation only. Report PSI and prevalence by period/window. Distinguish prevalence shift, ranking quality, calibration and threshold drift.

`outputs/report_summary.md` contains ready-made tables for all of the above; copy only measured values from section A.

## 7. Results
Fill this table only after running the experiment. Do not invent values. State whether the run used a sample or full data, and the split method. PR-AUC baseline equals the test fraud prevalence.

| Model | Threshold | Precision | Recall | F1 | PR-AUC (baseline) | Precision@100 | False alerts |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rule-based | n/a | — | — | — | n/a | n/a | — |
| Logistic Regression | — | — | — | — | — | — | — |
| Random Forest | — | — | — | — | — | — | — |

Add 95% bootstrap intervals from `metrics.json` (`models.<name>.ci.intervals`) next to precision, recall and PR-AUC.

### 7.1 Threshold strategies (test period)
| Model | Strategy | Threshold | Precision | Recall | False alerts | Missed fraud | Alerts per 1,000 |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Max F1 | — | — | — | — | — | — |
| Logistic Regression | Alert budget | — | — | — | — | — | — |
| Random Forest | Max F1 | — | — | — | — | — | — |
| Random Forest | Alert budget | — | — | — | — | — | — |

Note whether the realised test alert rate matched the validation budget, and why it might not.

### 7.2 Temporal backtest
| Window | Test steps | Test rows | Test fraud | Fraud rate (baseline) | LR PR-AUC | RF PR-AUC | Enough fraud? |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | — | — | — | — | — | — | — |

Report mean, standard deviation, minimum and maximum across windows (`backtest.json` → `summary`). Discuss whether performance is stable over time and how much of any change is explained by the changing fraud rate.

### 7.3 Uncertainty, calibration and drift
| Model | Metric | Row bootstrap 95% | Block bootstrap 95% | Width ratio |
|---|---|---|---|---:|
| Random Forest | Precision (max F1) | — | — | — |

| Model | Brier raw | Brier calibrated | Brier constant | ECE raw | ECE calibrated | PSI val→test |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | — | — | — | — | — | — |
| Random Forest | — | — | — | — | — | — |

### 7.4 Robustness across samples (multi-seed study)
Run `python -m src.robustness --sample-rows 1000000 --seeds 42 123 2026` and copy values from section A of `outputs/robustness_summary.md`. Use only a study whose status is `complete` and in which every seed is aggregated. Measured study `rob-de5ef78fad` (3 of 3 runs completed):

| Model | Test PR-AUC per seed (42 / 123 / 2026) | Mean (SD) | Max-F1 precision range | Max-F1 false-alert range |
|---|---|---|---|---|
| Logistic Regression | 0.153 / 0.145 / 0.140 | 0.146 (0.006) | 0.356–0.818 | 4–114 |
| Random Forest | 0.350 / 0.348 / 0.343 | 0.347 (0.004) | 0.564–0.775 | 38–146 |

Report between-seed variation (different samples) separately from temporal variation (backtest windows within a seed). In this study the typical across-window SD of Random Forest PR-AUC was 0.156, against 0.019 between seeds. Random Forest ranked above Logistic Regression in all 3 seeds and all 15 seed-windows. Say that three seeds give a descriptive spread, not a confidence interval, and do not pool per-run bootstrap intervals. Discuss why threshold-dependent metrics (precision, false alerts) vary more than PR-AUC.

## 8. Explainability and Dashboard
Describe the ten dashboard pages (including the read-only Robustness Study page). Global importance and per-transaction contributions (exact log-odds contributions for Logistic Regression; tree-path contributions for Random Forest) summarise model behaviour, not causality. Show one worked example from 'Try a Transaction'.

## 9. Limitations and Ethics
The dataset is synthetic. All measured results come from 1M-row uniform samples, not the full 6.36M rows. Synthetic PaySim results do not establish performance at a real bank. A random split does not test future-time generalization, and simulated labels may not match real fraud. Alerts are not proof and must not be used for financial decisions.

## 10. Conclusion and Future Work
Summarize measured results after training. Future extensions: nested temporal validation for threshold transfer, TreeSHAP comparison, pre-transaction balance features as a documented experiment, and full-data runs on larger hardware.

## References
1. Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016). *PaySim: A financial mobile money simulator for fraud detection*. EMSS.
2. Kaggle, *Synthetic Financial Datasets For Fraud Detection*. https://www.kaggle.com/datasets/ealaxi/paysim1
3. Creative Commons, *Attribution-ShareAlike 4.0 International*. https://creativecommons.org/licenses/by-sa/4.0/
