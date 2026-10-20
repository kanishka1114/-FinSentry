# FinSentry — multi-seed robustness summary (auto-generated)

_Generated 2026-10-20T11:54:31+00:00 from outputs/robustness/robustness.json._

> **PaySim is a synthetic simulation. These results describe models on simulated data and do not establish detection performance at any real bank or mobile-money provider.** Every run uses an independent uniform sample of 1,000,000 rows, never the full dataset; full-data results were not measured.

Study `rob-de5ef78fad` · status **complete** · seeds 42, 123, 2026

## Runs

| Seed | Status | Aggregated | Experiment | Train run | Backtest run | Data SHA-256 | Seconds | Problems |
|---|---|---|---|---|---|---|---|---|
| 42 | complete | yes | exp-7feac00e3b | ff5e307d217d | c87a3e7b4877 | 16910f90577b | 318 | none |
| 123 | complete | yes | exp-ce67704e29 | e64db6ba64f4 | 224d1838ad60 | 16910f90577b | 293 | none |
| 2026 | complete | yes | exp-4ff3a723eb | 87da10b3876c | 9e4723665621 | 16910f90577b | 293 | none |

Total wall time 904 s; process peak RSS 750.7 MB (measured with getrusage for the whole study process).

Boundaries (single split, steps):

| Seed | Train | Validation | Test |
|---|---|---|---|
| 42 | 1–323 | 324–377 | 378–742 |
| 123 | 1–323 | 324–377 | 378–743 |
| 2026 | 1–323 | 324–377 | 378–742 |

## A. Measured results

Every number below is copied or computed arithmetically from the per-seed JSON outputs. SD is the sample SD across seeds (n shown). Spread across seeds is descriptive (n = number of completed seeds). With three seeds it is not a confidence interval for the population of possible samples, and per-run bootstrap intervals are not pooled.

### A1. Single chronological split — between-seed variation

Test fraud prevalence (= PR-AUC baseline) per seed: 42: 0.42%, 123: 0.44%, 2026: 0.42% (mean 0.42%).

| Model | Metric | seed 42 | seed 123 | seed 2026 | Mean | SD | Range |
|---|---|---|---|---|---|---|---|
| Rule-based | Precision (rule) | 0.016 | 0.017 | 0.016 | 0.016 | 0.001 | 0.016 – 0.017 |
| Rule-based | Recall (rule) | 0.667 | 0.696 | 0.681 | 0.681 | 0.015 | 0.667 – 0.696 |
| Rule-based | False alerts (rule) | 26,120 | 26,237 | 26,168 | 26175.0 | 58.8 | 26,120 – 26,237 |
| Rule-based | Missed fraud (rule) | 209 | 199 | 200 | 202.7 | 5.5 | 199 – 209 |
| Rule-based | Alerts / 1,000 (rule) | 176.40 | 178.27 | 177.74 | 177.47 | 0.97 | 176.40 – 178.27 |
| Logistic Regression | PR-AUC | 0.153 | 0.145 | 0.140 | 0.146 | 0.006 | 0.140 – 0.153 |
| Logistic Regression | Precision (max F1) | 0.356 | 0.818 | 0.585 | 0.586 | 0.231 | 0.356 – 0.818 |
| Logistic Regression | Recall (max F1) | 0.100 | 0.027 | 0.061 | 0.063 | 0.037 | 0.027 – 0.100 |
| Logistic Regression | False alerts (max F1) | 114 | 4 | 27 | 48.3 | 58.0 | 4 – 114 |
| Logistic Regression | Missed fraud (max F1) | 564 | 637 | 589 | 596.7 | 37.1 | 564 – 637 |
| Logistic Regression | Alerts / 1,000 (max F1) | 1.18 | 0.15 | 0.43 | 0.59 | 0.53 | 0.15 – 1.18 |
| Logistic Regression | Precision (alert budget) | 0.263 | 0.272 | 0.271 | 0.269 | 0.005 | 0.263 – 0.272 |
| Logistic Regression | Recall (alert budget) | 0.132 | 0.119 | 0.112 | 0.121 | 0.011 | 0.112 – 0.132 |
| Logistic Regression | False alerts (alert budget) | 232 | 209 | 188 | 209.7 | 22.0 | 188 – 232 |
| Logistic Regression | Missed fraud (alert budget) | 544 | 577 | 557 | 559.3 | 16.6 | 544 – 577 |
| Logistic Regression | Alerts / 1,000 (alert budget) | 2.09 | 1.92 | 1.72 | 1.91 | 0.18 | 1.72 – 2.09 |
| Random Forest | PR-AUC | 0.350 | 0.348 | 0.343 | 0.347 | 0.004 | 0.343 – 0.350 |
| Random Forest | Precision (max F1) | 0.649 | 0.775 | 0.564 | 0.663 | 0.106 | 0.564 – 0.775 |
| Random Forest | Recall (max F1) | 0.260 | 0.200 | 0.301 | 0.254 | 0.051 | 0.200 – 0.301 |
| Random Forest | False alerts (max F1) | 88 | 38 | 146 | 90.7 | 54.0 | 38 – 146 |
| Random Forest | Missed fraud (max F1) | 464 | 524 | 438 | 475.3 | 44.1 | 438 – 524 |
| Random Forest | Alerts / 1,000 (max F1) | 1.67 | 1.13 | 2.24 | 1.68 | 0.56 | 1.13 – 2.24 |
| Random Forest | Precision (alert budget) | 0.349 | 0.370 | 0.250 | 0.323 | 0.064 | 0.250 – 0.370 |
| Random Forest | Recall (alert budget) | 0.381 | 0.386 | 0.402 | 0.390 | 0.011 | 0.381 – 0.402 |
| Random Forest | False alerts (alert budget) | 446 | 430 | 755 | 543.7 | 183.2 | 430 – 755 |
| Random Forest | Missed fraud (alert budget) | 388 | 402 | 375 | 388.3 | 13.5 | 375 – 402 |
| Random Forest | Alerts / 1,000 (alert budget) | 4.55 | 4.56 | 6.73 | 5.28 | 1.25 | 4.55 – 6.73 |

### A2. Walk-forward backtest — window PR-AUC vs that window's fraud-rate baseline

| Seed | Window | Test steps | Rows | Fraud | Baseline (prevalence) | LR PR-AUC | LR × baseline | RF PR-AUC | RF × baseline | Enough fraud |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 1 | 239–281 | 101,165 | 80 | 0.0008 | 0.033 | 41.4 | 0.215 | 271.5 | yes |
| 42 | 2 | 282–323 | 100,653 | 66 | 0.0007 | 0.006 | 8.8 | 0.089 | 135.6 | yes |
| 42 | 3 | 324–354 | 95,345 | 47 | 0.0005 | 0.005 | 9.4 | 0.077 | 156.2 | yes |
| 42 | 4 | 355–398 | 102,720 | 74 | 0.0007 | 0.051 | 71.2 | 0.352 | 488.1 | yes |
| 42 | 5 | 399–742 | 100,468 | 585 | 0.0058 | 0.186 | 32.0 | 0.359 | 61.7 | yes |
| 123 | 1 | 239–281 | 101,565 | 70 | 0.0007 | 0.034 | 49.6 | 0.087 | 126.7 | yes |
| 123 | 2 | 282–323 | 100,477 | 67 | 0.0007 | 0.006 | 9.2 | 0.085 | 127.1 | yes |
| 123 | 3 | 324–354 | 95,052 | 52 | 0.0005 | 0.005 | 8.6 | 0.024 | 44.8 | yes |
| 123 | 4 | 355–398 | 102,976 | 89 | 0.0009 | 0.046 | 53.1 | 0.337 | 389.6 | yes |
| 123 | 5 | 399–743 | 99,831 | 610 | 0.0061 | 0.174 | 28.5 | 0.388 | 63.6 | yes |
| 2026 | 1 | 239–281 | 101,028 | 65 | 0.0006 | 0.031 | 48.0 | 0.099 | 153.4 | yes |
| 2026 | 2 | 282–323 | 100,998 | 68 | 0.0007 | 0.007 | 10.8 | 0.060 | 88.8 | yes |
| 2026 | 3 | 324–354 | 95,116 | 43 | 0.0005 | 0.004 | 9.2 | 0.037 | 81.2 | yes |
| 2026 | 4 | 355–398 | 102,968 | 81 | 0.0008 | 0.052 | 65.6 | 0.368 | 468.4 | yes |
| 2026 | 5 | 399–742 | 99,742 | 586 | 0.0059 | 0.171 | 29.2 | 0.364 | 62.0 | yes |

Window boundaries are set by row share within each sample, so window k covers similar but not identical steps in different seeds.

### A3. Temporal vs between-seed variation (backtest)

Temporal: SD across the windows of one seed. Between-seed: SD of each seed's window mean. They answer different questions and are not combined.

| Model | Metric | seed 42: window mean ± SD [min–max] | seed 123: window mean ± SD [min–max] | seed 2026: window mean ± SD [min–max] | Typical temporal SD | Between-seed SD of mean |
|---|---|---|---|---|---|---|
| Rule-based | rule.precision | 0.006 ± 0.009 [0.001–0.023] | 0.007 ± 0.010 [0.002–0.024] | 0.006 ± 0.009 [0.002–0.023] | 0.009 | 0.000 |
| Rule-based | rule.recall | 0.634 ± 0.078 [0.553–0.738] | 0.686 ± 0.058 [0.627–0.775] | 0.687 ± 0.045 [0.615–0.728] | 0.060 | 0.030 |
| Rule-based | rule.false_positive | 18298.80 ± 1275.59 [16,880–19,275] | 18221.00 ± 1009.28 [17,011–19,037] | 18200.00 ± 1144.77 [16,922–19,059] | 1143.21 | 52.05 |
| Rule-based | rule.false_negative | 57.80 ± 75.10 [21–192] | 54.80 ± 74.56 [16–188] | 53.60 ± 75.29 [12–188] | 74.98 | 2.16 |
| Rule-based | rule.alerts_per_1000 | 184.18 ± 14.21 [167.87–202.43] | 183.67 ± 12.06 [167.93–199.43] | 183.37 ± 12.64 [167.89–199.97] | 12.97 | 0.41 |
| Logistic Regression | pr_auc | 0.056 ± 0.075 [0.005–0.186] | 0.053 ± 0.070 [0.005–0.174] | 0.053 ± 0.069 [0.004–0.171] | 0.071 | 0.002 |
| Logistic Regression | pr_auc_lift | 32.57 ± 25.84 [8.83–71.19] | 29.78 ± 21.26 [8.58–53.07] | 32.54 ± 24.28 [9.21–65.56] | 23.79 | 1.60 |
| Logistic Regression | max_f1.precision | 0.239 ± 0.402 [0.005–0.950] | 0.151 ± 0.278 [0.005–0.646] | 0.147 ± 0.255 [0.006–0.600] | 0.312 | 0.052 |
| Logistic Regression | max_f1.recall | 0.142 ± 0.117 [0.032–0.340] | 0.200 ± 0.234 [0.057–0.607] | 0.194 ± 0.154 [0.056–0.395] | 0.168 | 0.032 |
| Logistic Regression | max_f1.false_positive | 841.00 ± 1302.16 [1–3,055] | 1091.60 ± 1107.31 [23–2,540] | 903.40 ± 1005.89 [22–2,490] | 1138.45 | 130.46 |
| Logistic Regression | max_f1.false_negative | 158.40 ± 228.36 [31–566] | 154.60 ± 231.47 [35–568] | 150.20 ± 225.53 [29–553] | 228.45 | 4.10 |
| Logistic Regression | max_f1.alerts_per_1000 | 8.81 ± 13.68 [0.20–32.21] | 11.18 ± 11.09 [0.51–25.19] | 9.40 ± 10.54 [0.55–26.33] | 11.77 | 1.23 |
| Logistic Regression | alert_budget.precision | 0.081 ± 0.123 [0.000–0.296] | 0.080 ± 0.111 [0.000–0.272] | 0.081 ± 0.119 [0.000–0.288] | 0.118 | 0.001 |
| Logistic Regression | alert_budget.recall | 0.151 ± 0.089 [0.000–0.227] | 0.167 ± 0.114 [0.000–0.299] | 0.171 ± 0.121 [0.000–0.338] | 0.108 | 0.011 |
| Logistic Regression | alert_budget.alerts_per_1000 | 7.20 ± 10.58 [1.17–26.06] | 6.89 ± 9.80 [1.44–24.34] | 6.83 ± 10.00 [1.24–24.65] | 10.13 | 0.20 |
| Random Forest | pr_auc | 0.218 ± 0.136 [0.077–0.359] | 0.184 ± 0.166 [0.024–0.388] | 0.186 ± 0.166 [0.037–0.368] | 0.156 | 0.019 |
| Random Forest | pr_auc_lift | 222.61 ± 166.38 [61.72–488.08] | 150.36 ± 138.78 [44.76–389.63] | 170.75 ± 169.88 [61.95–468.38] | 158.35 | 37.25 |
| Random Forest | max_f1.precision | 0.553 ± 0.411 [0.022–1.000] | 0.333 ± 0.392 [0.000–0.893] | 0.514 ± 0.425 [0.014–1.000] | 0.409 | 0.118 |
| Random Forest | max_f1.recall | 0.217 ± 0.120 [0.021–0.333] | 0.163 ± 0.108 [0.000–0.270] | 0.161 ± 0.116 [0.023–0.311] | 0.115 | 0.032 |
| Random Forest | max_f1.false_positive | 218.20 ± 417.51 [0–963] | 229.00 ± 472.57 [1–1,074] | 147.80 ± 236.86 [0–560] | 375.65 | 44.10 |
| Random Forest | max_f1.false_negative | 129.40 ± 173.81 [44–440] | 144.60 ± 194.86 [51–493] | 125.00 ± 156.16 [42–404] | 174.94 | 10.28 |
| Random Forest | max_f1.alerts_per_1000 | 2.57 ± 4.12 [0.01–9.79] | 2.61 ± 4.63 [0.01–10.85] | 1.90 ± 2.45 [0.01–5.62] | 3.73 | 0.40 |
| Random Forest | alert_budget.precision | 0.104 ± 0.118 [0.020–0.306] | 0.109 ± 0.128 [0.018–0.327] | 0.095 ± 0.110 [0.021–0.284] | 0.119 | 0.007 |
| Random Forest | alert_budget.recall | 0.381 ± 0.131 [0.170–0.486] | 0.376 ± 0.178 [0.058–0.472] | 0.410 ± 0.142 [0.163–0.519] | 0.150 | 0.019 |
| Random Forest | alert_budget.alerts_per_1000 | 7.07 ± 5.54 [2.74–16.29] | 7.03 ± 5.91 [1.45–16.52] | 7.38 ± 4.53 [2.85–14.32] | 5.33 | 0.20 |

Window fraud prevalence per seed: 42: 0.08%, 0.07%, 0.05%, 0.07%, 0.58%; 123: 0.07%, 0.07%, 0.05%, 0.09%, 0.61%; 2026: 0.06%, 0.07%, 0.05%, 0.08%, 0.59%

### A4. Ranking and trade-off consistency

- Single split: Random Forest had the higher test PR-AUC in all 3 seeds.
- Backtest: Random Forest had the higher PR-AUC in 15 of 15 seed-windows, Logistic Regression in 0.
- Logistic Regression PR-AUC exceeded its window's fraud-rate baseline in 15 of 15 seed-windows.
- Random Forest PR-AUC exceeded its window's fraud-rate baseline in 15 of 15 seed-windows.
- Logistic Regression: typical temporal SD of window PR-AUC (0.071) is larger than the between-seed SD of the window mean (0.002).
- Random Forest: typical temporal SD of window PR-AUC (0.156) is larger than the between-seed SD of the window mean (0.019).
- Logistic Regression: in every seed the alert-budget threshold traded precision for recall relative to max F1 (consistent operational trade-off).
- Random Forest: in every seed the alert-budget threshold traded precision for recall relative to max F1 (consistent operational trade-off).

**Warnings recorded in the runs**

- Backtest uses a sample, not the full dataset.
- Fraud prevalence differs strongly between train and test periods; precision depends heavily on prevalence.
- Results come from a sample, not the full dataset.

## B. Interpretation (explanatory text, not measurements)

- Between-seed variation mixes two sources: which rows the uniform sample drew and the Random Forest's random seed. It says how much a headline number could move with a different 1M-row sample of the same synthetic data, not how a model would perform on other data.
- Temporal variation is usually the larger effect here because the fraud rate and score distribution change over simulated time; PR-AUC must be read against each window's own baseline.
- Three seeds give a rough stability check. SDs from n = 3 are themselves very uncertain; they are not confidence intervals and per-run bootstrap intervals were not pooled.
- Where the findings say 'inconclusive', the evidence does not support a stable ranking or trade-off claim.
- Re-running a seed reproduces the sample, split, Logistic Regression and PR-AUC exactly. Random Forest prediction with n_jobs > 1 sums tree probabilities in thread order, so scores can differ in the last floating-point digit and a transaction scoring exactly at a threshold may flip (a difference of about one alert). With n_jobs=1 runs are bit-identical.
- PaySim is a synthetic simulation. These results describe models on simulated data and do not establish detection performance at any real bank or mobile-money provider.

## C. Reproduce

```bash
python -m src.robustness --sample-rows 1000000 --seeds 42 123 2026
```

Library versions: python 3.13.3, platform macOS-14.6-arm64-arm-64bit-Mach-O, numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, joblib 1.6.0, streamlit 1.66.0, plotly 7.1.0
