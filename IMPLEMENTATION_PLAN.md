# FinSentry implementation plan

## Completed
1. Foundation: memory-safe loading, reproducible sampling (label-free uniform by default; stratified for development), `--max-step`, full-data guard. Done.
2. Trustworthy evaluation: chronological split, no raw step feature, validation-only thresholds, single test evaluation. Done.
3. Honest reporting: prevalence baselines, top-K, rule as point metrics, run metadata and dataset fingerprint. Done.
4. Dashboard: labels, comparison, PR curves, confusion matrices, enriched preview, manual scoring. Done.
5. Reliability: walk-forward backtest, max-F1 vs alert-budget thresholds, row bootstrap. Done.
6. Research upgrade: moving-block bootstrap over steps, documented final-fit protocol with checked threshold transfer, leakage tests that flip test labels, calibration (Brier, ECE, reliability, Platt on validation), drift (PSI, score quantiles, alert volume), per-transaction explanations, auto-generated report. Done.
7. Release audit:
   - `python -m src.experiment` is now the single recommended workflow. It records `experiment.json` and tags all outputs with one experiment id.
   - Uniform sampling is the default.
   - F1 is undefined (not 0) when there is no fraud.
   - The refit protocol is marked experimental.
   - The ECE and bootstrap-conditioning caveats are documented.
   - Stale and mixed outputs are detected in both the report and the dashboard.
   - The report is always rebuilt from the current JSON.
   - Dashboard smoke tests cover all 10 pages.
   - Metric-consistency tests check that confusion counts, precision, recall, alerts and top-K agree.
   - Measured 1M-row results are in the README and the report.
   - Done.

8. Multi-seed robustness study:
   - `python -m src.robustness` runs the unchanged experiment on independent uniform samples, each in its own folder (`outputs/robustness/seed_<seed>/`).
   - Only complete, consistent runs are aggregated.
   - It reports between-seed and temporal variation separately, plus ranking and trade-off checks.
   - Executed for seeds 42, 123 and 2026 on 1M-row samples (study `rob-de5ef78fad`, 905 s, 0.75 GB peak RSS). All 3 runs completed.
   - Tests cover seed reproducibility, isolation, incomplete and failed runs, aggregation arithmetic and the summary text.
   - Done.
9. Dashboard presentation and demo:
   - Dark theme (`.streamlit/config.toml`) and presentation-only helpers in `src/ui.py`: brand header, KPI cards with definitions, number formatting that keeps "undefined" distinct from 0, a shared Plotly template and chart subtitles.
   - A read-only Robustness Study page and robustness context on Overview and Thresholds & Alerts. Every page labels the data as synthetic PaySim.
   - No metric, model or output changed. `docs/DEMO_GUIDE.md` added; `tests/test_ui.py` covers formatting, HTML escaping, the robustness loaders and the Robustness Study page.
   - Done.

## Remaining (future work, not started)
- Deterministic Random Forest prediction (`n_jobs=1` at scoring time) if bit-identical re-runs are required. Currently re-runs of a seed can differ by about one alert at the max-F1 threshold.
- More seeds, and a more stable threshold rule than validation max F1, which moved noticeably between samples.
- Nested temporal validation so a refit model's thresholds can be checked on unseen pre-test data.
- TreeSHAP comparison for Random Forest explanations (optional dependency).
- Balance-feature experiment with an explicit leakage analysis.
- Full-data (6.36M rows) runs on larger hardware. No full-data results exist yet.
- Prevalence-aware calibration (for example, re-estimating the calibrator per window) to deal with the large validation-to-test prevalence shift.
