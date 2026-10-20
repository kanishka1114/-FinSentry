# FinSentry: project overview and viva guide

All numbers below come from experiment `exp-0781f6ac1d` (`python -m src.experiment --sample-rows 1000000`, seed 42). If the experiment is re-run, update them from `outputs/report_summary.md`.

## One-paragraph overview
FinSentry asks how well simple, explainable models can rank mobile-money transactions by fraud risk when they are evaluated honestly over time. It compares a transparent rule with Logistic Regression and Random Forest on the synthetic PaySim dataset.

"Honestly" means several things:
- Data is split chronologically.
- Thresholds are chosen on validation data only.
- The test period is scored once.
- Baselines are prevalence-aware.
- Uncertainty is estimated with a time-aware bootstrap.
- A walk-forward backtest shows variation over time.
- Calibration and drift diagnostics are included.

Results are shown in a 10-page Streamlit dashboard and an auto-generated report. PaySim is synthetic, so the results describe the method, not performance at a real bank.

## Problem
- Fraud is rare: 1,279 of the 1,000,000 sampled rows (0.13%). Accuracy is therefore meaningless, because flagging nothing scores 99.87%.
- The useful questions are about ranking and workload:
  - How much fraud is in the top alerts?
  - How many false alerts must a reviewer handle?
  - How much fraud is missed?

## Data
- PaySim simulates 743 hourly steps of mobile-money activity (6,362,620 rows). Fraud occurs only in TRANSFER and CASH_OUT.
- Only `step`, `type`, `amount` and `isFraud` are loaded, in chunks, with compact data types.
- Balances, account identifiers and `isFlaggedFraud` are excluded. Balance fields in PaySim are known to leak the label, and identifiers would let a model memorise accounts.
- The sample is **uniform**: rows are chosen without looking at labels. A stratified option exists for development but uses global label counts, so the report flags it.

## Features and models
- Features:
  - `amount` and `log_amount`.
  - Hour of day as `hour_sin` / `hour_cos`. Raw `step` is never a feature, to avoid a time shortcut.
  - One-hot transaction type.
- Rule baseline: 1 point each for a risky type, amount ≥ 200k and amount ≥ 1M. A transaction is flagged at 2 or more points. The rule has no score, so it has no PR-AUC.
- Logistic Regression: standardised inputs with `class_weight="balanced"`. Fully interpretable.
- Random Forest: 100 trees, depth 12, `max_samples=0.5`, `class_weight="balanced_subsample"`. Sized for an 8 GB laptop.

## Evaluation protocol
1. Split by whole steps: train is steps 1–323, validation 324–377, test 378–742 (about 70/15/15 by rows).
2. Fit on train only.
3. Choose thresholds on validation only:
   - **max F1**;
   - an **alert budget** of 5 alerts per 1,000 transactions.
   The Platt calibrator is also fit on validation.
4. Keep the train-only model as final. Refit on train+validation is available but experimental.
5. Score the test period once.

Test labels are used only for reporting. `tests/test_leakage.py` flips every test label and checks that nothing the model or threshold depends on changes.

## Key measured findings
- **Prevalence shift**: fraud is 0.08% in train, 0.05% in validation and 0.42% in test. PR-AUC must be compared with its baseline (the test prevalence, 0.0042), and precision is not comparable across periods.
- **Random Forest is the strongest ranker.**
  - Test PR-AUC is 0.350 (84× baseline), with a block-bootstrap 95% interval of 0.301–0.411.
  - At the max-F1 threshold: precision 0.652, recall 0.260, 87 false alerts, 464 missed fraud, 1.66 alerts per 1,000.
- **Logistic Regression**: PR-AUC 0.153 (0.105–0.223). At max F1: precision 0.356, recall 0.100.
- **The rule finds 67% of fraud but raises 176 alerts per 1,000** (26,120 false alerts, precision 0.016). That review load is unworkable.
- **The threshold strategy changes the trade-off.** The Random Forest alert budget (5 per 1,000 on validation) gave 4.55 per 1,000 on test, with recall 0.381 but precision 0.349. Neither threshold is "best"; the choice depends on the cost of a missed fraud versus review capacity.
- **Performance varies strongly over time.** Backtest Random Forest PR-AUC ranged from 0.077 to 0.359 across 5 windows (Logistic Regression 0.005–0.186). A single split hides this.
- **Uncertainty.** Block-bootstrap intervals were up to 2.9× wider than row-bootstrap intervals. Treating rows as independent understates uncertainty.
- **Calibration.** Class weighting inflates raw scores: the Logistic Regression raw Brier score is 0.123. After Platt scaling on validation it is 0.0041, which is only about equal to a constant prediction (0.0042). The calibrator fixes the scale, but under prevalence shift it cannot be trusted across periods.

## Robustness across samples (study `rob-de5ef78fad`)
The full experiment was repeated on three independent uniform 1M-row samples (seeds 42, 123 and 2026). All 3 runs completed, taking 905 s in total with 0.75 GB peak RSS.
- **The ranking is stable.** Random Forest test PR-AUC was 0.350 / 0.348 / 0.343, against 0.153 / 0.145 / 0.140 for Logistic Regression. Random Forest also won in all 15 backtest seed-windows, and both models beat every window's fraud-rate baseline.
- **Thresholded numbers are less stable.** Random Forest max-F1 precision ranged 0.564–0.775 and false alerts 38–146. Logistic Regression precision ranged 0.356–0.818. So "precision 0.65" should be presented as one sample's value, not a fixed property.
- **The trade-off is consistent.** In every seed the alert budget gave higher recall and lower precision than max F1.
- **Time dominates.** Across backtest windows the SD of Random Forest PR-AUC was about 0.156. Between seeds the SD of the window mean was 0.019.
- **Three seeds are a sanity check, not a confidence interval.** Everything is still a sample of synthetic data.

## Likely viva questions
- **How robust are the results?** See the section above. The model ranking held in every sample and every window. Precision and false-alert counts at the max-F1 threshold moved noticeably between samples, because that threshold is fit on fewer than 100 validation fraud cases.
- **Why doesn't seed 42 in the study exactly match the primary run?** Everything matched except one Random Forest false alert (88 vs 87). Parallel Random Forest prediction sums tree outputs in thread order, so one transaction exactly at the threshold flipped. Single-threaded runs are bit-identical.
- **Why not accuracy or a random split?** Accuracy rewards flagging nothing. A random split mixes future and past rows, which inflates results and hides drift.
- **Why is validation precision so much lower than test precision?** Validation prevalence is about 8× lower than test prevalence. Precision scales with prevalence even for an unchanged model.
- **Why is the max-F1 test alert rate (1.66 per 1,000) higher than on validation (0.21)?** The test period contains more fraud and its score distribution shifted. A fixed threshold therefore gives a different workload, which is threshold drift.
- **Are the confidence intervals guarantees?** No. They are conditional on the trained model and the chosen thresholds, and they assume rough stationarity within the test period. The backtest shows the larger variation over time.
- **Why does the Random Forest explanation start at a 50% baseline?** Balanced class weights make the weighted root node about 50/50. Contributions explain movement from that baseline. They describe the model, not causes of fraud.
- **Why not use balance features?** In PaySim they almost reveal the label, so results would look excellent but mean little. Testing them needs a separate leakage analysis.
- **Would this work at a real bank?** That is unknown. The data is synthetic, the feature set is tiny, and no full-data or real-data experiment was run.

## Limitations
- Synthetic data.
- A 1M-row uniform sample, not the full data. Full-data results are not measured.
- Only three inputs.
- Few fraud cases in some windows.
- Intervals are estimates.
- Thresholds may not transfer under drift.
- The calibrator is affected by prevalence shift.
- Explanations are not causal.
- Not for production or financial decisions.

## Future work
- Nested temporal validation for refit thresholds.
- A TreeSHAP comparison.
- A balance-feature study with leakage analysis.
- Full-data runs on bigger hardware, more seeds, and a more stable threshold rule.
- Per-period recalibration.

## Demo order (about 5 minutes)
A timed script with talking points is in [`DEMO_GUIDE.md`](DEMO_GUIDE.md).
1. Overview: synthetic-data note, KPI cards, PR-AUC against its baseline, period fraud rates, final test table.
2. Thresholds & Alerts: max F1 vs alert budget, the instability note, cost explorer.
3. Temporal Backtest: PR-AUC per window against each window's baseline.
4. Robustness Study: three samples agree on the ranking; time matters more than the sample.
5. Try a Transaction: score a large TRANSFER and explain the contributions.
6. Report Summary: download the current report.
